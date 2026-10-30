"""
RL-adapted Connectome-Constrained CfC Policy for Locomotion Control.

This module provides an RL-specific connectome RNN that does NOT modify
the original ``larva/src/continuous_rnn.py``. It reuses the same
connectome loading utilities and mathematical framework (CTRNN/LTC/CfC
gating), but is tailored for:

- Proprioceptive input (46D) instead of visual features
- 6D tanh-bounded correction output instead of 2D speed commands
- Integration with stable-baselines3 as a custom feature extractor

The model can be used in two modes:
  - **Feedforward** (SB3 training): each observation processed independently
  - **Recurrent** (online evaluation): hidden state persists across steps
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Dict

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Ensure larva/src is importable
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from connectome import load_connectivity_data


class ConnectomeCfCRL(nn.Module):
    """
    Connectome-constrained CfC for RL locomotion correction.

    Same mathematical framework as ``ConnectomeCfC`` in continuous_rnn.py
    but with RL-appropriate I/O:
        Input:  46D proprioceptive observation → Internal neurons
        Output: 6D tanh-bounded correction magnitudes (one per leg)

    Parameters
    ----------
    W_init : np.ndarray, shape (N, N)
        Connectome adjacency in SIO order.
    obs_dim : int
        Observation dimension (default 46).
    action_dim : int
        Action dimension (default 6, one per leg).
    sensory_dim, internal_dim, output_dim : int
        SIO partition sizes.
    mode : str
        Dynamics: 'ctrnn', 'ltc', or 'cfc'.
    dt : float
        Integration timestep.
    tau_init : float
        Initial time constant (seconds).
    use_lora : bool
        Enable LoRA adaptation.
    lora_rank : int
        LoRA rank.
    trainable_W : bool
        Make connectome weights directly trainable.
    """

    VALID_MODES = ('ctrnn', 'ltc', 'cfc')

    def __init__(
        self,
        W_init: np.ndarray,
        obs_dim: int = 46,
        action_dim: int = 6,
        sensory_dim: int = 430,
        internal_dim: int = 2304,
        output_dim: int = 218,
        mode: Literal['ctrnn', 'ltc', 'cfc'] = 'cfc',
        dt: float = 1e-4,
        tau_init: float = 0.01,
        tau_min: float = 1e-4,
        tau_max: float = 1.0,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        trainable_W: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()
        assert mode in self.VALID_MODES
        N = sensory_dim + internal_dim + output_dim
        assert W_init.shape == (N, N), f"W shape {W_init.shape} != ({N},{N})"

        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = N
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mode = mode
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.noise_std = noise_std

        # --- Connectome ---
        self.register_buffer('W_init_buf',
                             torch.tensor(W_init, dtype=torch.float32))
        self.register_buffer('sparsity_mask',
                             (torch.tensor(W_init, dtype=torch.float32) != 0).float())
        W_tensor = torch.tensor(W_init, dtype=torch.float32)
        if trainable_W:
            self.W = nn.Parameter(W_tensor)
        else:
            self.register_buffer('W', W_tensor)

        # --- LoRA ---
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 0
        if use_lora:
            self.lora_A = nn.Parameter(torch.empty(N, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(lora_rank, N))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        # --- Dynamics parameters ---
        if mode == 'ctrnn':
            self.log_tau = nn.Parameter(
                torch.full((N,), float(np.log(tau_init))))
        elif mode == 'ltc':
            self.log_tau_base = nn.Parameter(
                torch.full((N,), float(np.log(tau_init))))
            self.ltc_w_mod = nn.Parameter(torch.zeros(N))
            self.ltc_b_mod = nn.Parameter(torch.zeros(N))
            self.ltc_gamma = nn.Parameter(torch.ones(N) * 0.5)
        elif mode == 'cfc':
            self.cfc_speed_w_drive = nn.Parameter(torch.randn(N) * 0.1)
            self.cfc_speed_w_state = nn.Parameter(torch.randn(N) * 0.1)
            self.cfc_speed_bias = nn.Parameter(torch.full((N,), 2.0))
            self.cfc_offset_w_drive = nn.Parameter(torch.randn(N) * 0.1)
            self.cfc_offset_w_state = nn.Parameter(torch.randn(N) * 0.1)
            self.cfc_offset_bias = nn.Parameter(torch.zeros(N))

        # --- Input projection: obs → Internal neurons ---
        self.input_proj = nn.Linear(obs_dim, internal_dim)
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)

        # --- Output readout: O neurons → corrections ---
        self.output_layer = nn.Linear(output_dim, action_dim)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ConnectomeCfCRL [{mode.upper()}] | S={sensory_dim} I={internal_dim} "
              f"O={output_dim} | obs={obs_dim}→I, action={action_dim} | "
              f"trainable params: {n_train}")

    def get_effective_W(self) -> torch.Tensor:
        W_eff = self.W
        if self.use_lora:
            W_eff = W_eff + self.lora_scaling * (self.lora_A @ self.lora_B)
        return W_eff * self.sparsity_mask

    def _compute_gate(self, x_state, r, drive):
        if self.mode == 'ctrnn':
            tau = torch.clamp(torch.exp(self.log_tau), self.tau_min, self.tau_max)
            return torch.clamp(self.dt / tau, 0.0, 1.0)
        elif self.mode == 'ltc':
            tau_base = torch.clamp(
                torch.exp(self.log_tau_base), self.tau_min, self.tau_max)
            modulation = self.ltc_gamma * torch.sigmoid(
                self.ltc_w_mod * drive + self.ltc_b_mod)
            tau_eff = torch.clamp(tau_base * (1 + modulation),
                                  self.tau_min, self.tau_max)
            return torch.clamp(self.dt / tau_eff, 0.0, 1.0)
        else:  # cfc
            speed = (self.cfc_speed_w_drive * drive
                     + self.cfc_speed_w_state * r
                     + self.cfc_speed_bias)
            offset = (self.cfc_offset_w_drive * drive
                      + self.cfc_offset_w_state * r
                      + self.cfc_offset_bias)
            return torch.sigmoid(speed * self.dt + offset)

    def step(
        self,
        x_state: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single integration step.

        Parameters
        ----------
        x_state : (batch, N) membrane potentials
        obs : (batch, obs_dim) proprioceptive observation

        Returns
        -------
        x_next : (batch, N)
        action : (batch, action_dim) tanh-bounded corrections
        """
        S, I, O = self.sensory_dim, self.internal_dim, self.output_dim
        W_eff = self.get_effective_W()

        r = torch.tanh(x_state)
        drive = r @ W_eff

        I_ext = torch.zeros_like(x_state)
        I_ext[:, S:S+I] = self.input_proj(obs)

        full_drive = drive + I_ext
        x_target = torch.tanh(full_drive)

        gate = self._compute_gate(x_state, r, full_drive)
        x_next = (1.0 - gate) * x_state + gate * x_target

        if self.training and self.noise_std > 0:
            x_next = x_next + self.noise_std * torch.randn_like(x_next)

        r_next = torch.tanh(x_next)
        r_output = r_next[:, S+I:S+I+O]
        action = torch.tanh(self.output_layer(r_output))

        return x_next, action

    def init_state(self, batch_size=1, device=None):
        if device is None:
            device = self.W.device
        return 0.01 * torch.randn(batch_size, self.total_dim, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Feedforward mode for SB3 training: one-step from zero state.

        Runs the RNN for a small number of settling steps so the
        connectome dynamics have time to produce a meaningful response.
        """
        B = obs.shape[0]
        x = self.init_state(B, obs.device)
        n_settle = 3
        for _ in range(n_settle):
            x, action = self.step(x, obs)
        return action

    def get_config(self) -> dict:
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'sensory_dim': self.sensory_dim,
            'internal_dim': self.internal_dim,
            'output_dim': self.output_dim,
            'mode': self.mode,
            'dt': self.dt,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'use_lora': self.use_lora,
            'lora_rank': self.lora_rank,
            'trainable_W': isinstance(self.W, nn.Parameter),
            'noise_std': self.noise_std,
        }

    def save(self, path: str):
        torch.save({
            'config': self.get_config(),
            'state_dict': self.state_dict(),
            'W_init': self.W_init_buf.cpu().numpy(),
        }, path)

    @classmethod
    def load(cls, path: str, device='cpu') -> 'ConnectomeCfCRL':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = cls(W_init=ckpt['W_init'], **cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model.to(device)


# ====================================================================
# SB3 Feature Extractor wrapper
# ====================================================================

class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
    """
    Stable-baselines3 custom feature extractor that uses the connectome
    CfC RNN to process proprioceptive observations.

    The ``features_dim`` output is the action_dim (6) because the
    connectome RNN directly produces the correction signal. A thin
    linear actor head in SB3 will sit on top, but we initialise it
    close to identity so the connectome output dominates.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        connectome_path: str,
        annotation_path: str,
        mode: str = 'cfc',
        features_dim: int = 64,
        use_lora: bool = True,
        lora_rank: int = 16,
        dt: float = 1e-4,
    ):
        super().__init__(observation_space, features_dim)

        data = load_connectivity_data(
            connectome_path, annotation_path, sensory_type='all')
        W = data['W'].astype(np.float32)

        self.rnn = ConnectomeCfCRL(
            W_init=W,
            obs_dim=observation_space.shape[0],
            action_dim=features_dim,
            sensory_dim=len(data['sensory_ids']),
            internal_dim=len(data['internal_ids']),
            output_dim=len(data['output_ids']),
            mode=mode,
            dt=dt,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )
        self.output_proj = nn.Linear(self.rnn.action_dim, features_dim)
        nn.init.orthogonal_(self.output_proj.weight, gain=1.0)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self.rnn(observations)
        return self.output_proj(raw)


def build_connectome_rnn(
    connectome_csv: str = None,
    annotation_csv: str = None,
    obs_dim: int = 46,
    action_dim: int = 6,
    mode: str = 'cfc',
    dt: float = 1e-4,
    use_lora: bool = True,
    lora_rank: int = 16,
) -> ConnectomeCfCRL:
    """Convenience factory: load connectome and build the RL model."""
    if connectome_csv is None:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data")
        connectome_csv = os.path.join(base, "ad_connectivity_matrix.csv")
        annotation_csv = os.path.join(base, "science.add9330_data_s2.csv")

    data = load_connectivity_data(
        connectome_csv, annotation_csv, sensory_type='all')
    W = data['W'].astype(np.float32)

    return ConnectomeCfCRL(
        W_init=W,
        obs_dim=obs_dim,
        action_dim=action_dim,
        sensory_dim=len(data['sensory_ids']),
        internal_dim=len(data['internal_ids']),
        output_dim=len(data['output_ids']),
        mode=mode,
        dt=dt,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )

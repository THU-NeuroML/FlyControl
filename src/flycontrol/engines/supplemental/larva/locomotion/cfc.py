"""Connectome-constrained CfC dynamics core.

Four time-state modes share the same update rule:

    r       = tanh(x)
    drive   = r @ W_eff + I_ext
    target  = tanh(drive)
    gate    = mode_gate(x, r, drive)
    x_next  = (1 - gate) * x + gate * target

mode_gate:

    rate    gate = clamp(dt / tau, 0, 1)             # tau shared & fixed
    ctrnn   gate = clamp(dt / tau, 0, 1)             # tau learnable per neuron
    ltc     gate = clamp(dt / tau_eff, 0, 1)         # tau modulated by drive
    cfc     gate = sigmoid(speed * dt + offset)      # closed-form

The ``rate`` mode is the classic "vanilla rate model" baseline: a single
network-wide time constant, frozen at ``tau_init``, registered as a
buffer rather than a parameter. It exists explicitly so the paper can
attribute the contribution of *learnable per-neuron* time constants
against a connectome whose spectral radius (~0.39) implies very short
intrinsic memory.

Inputs of dimension ``input_dim`` are projected into the Internal
neurons. Output is a linear readout of the Output neurons; no squashing
is applied here so this module can serve as a generic building block for
a Gaussian policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .connectome import ConnectomeData


VALID_MODES = ("rate", "ctrnn", "ltc", "cfc")
GateMode = Literal["rate", "ctrnn", "ltc", "cfc"]
InputTarget = Literal["sensory", "internal", "all"]
CfcInit = Literal["legacy", "heterogeneous_tau"]


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return torch.log(x) - torch.log1p(-x)


class ConnectomeCfC(nn.Module):
    """Recurrent core operating on a fixed connectome adjacency."""

    def __init__(
        self,
        connectome: ConnectomeData,
        input_dim: int,
        output_dim: int,
        mode: GateMode = "cfc",
        dt: float = 1e-2,
        tau_init: float = 0.05,
        tau_min: float = 1e-3,
        tau_max: float = 1.0,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        trainable_W: bool = False,
        noise_std: float = 0.0,
        input_target: InputTarget = "internal",
        cfc_init: CfcInit = "legacy",
        cfc_tau_log_std: float = 0.5,
        cfc_gate_mod_init: float = 0.01,
    ) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        N = connectome.N
        self.S = connectome.sensory_dim
        self.I = connectome.internal_dim
        self.O = connectome.output_dim
        assert self.S + self.I + self.O == N

        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.dt = float(dt)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.noise_std = float(noise_std)
        self.input_target = input_target
        if input_target == "sensory":
            self.input_start = 0
            self.input_width = self.S
        elif input_target == "internal":
            self.input_start = self.S
            self.input_width = self.I
        elif input_target == "all":
            self.input_start = 0
            self.input_width = N
        else:
            raise ValueError("input_target must be sensory, internal, or all")

        self.is_sparse = connectome.is_sparse
        if self.is_sparse:
            if trainable_W:
                raise ValueError("trainable_W is not supported for sparse connectomes")
            if use_lora:
                raise ValueError("LoRA would densify a sparse connectome; disable it")
            assert connectome.edge_pre is not None
            assert connectome.edge_post is not None
            assert connectome.edge_weight is not None
            self.register_buffer(
                "edge_pre", torch.as_tensor(connectome.edge_pre, dtype=torch.long))
            self.register_buffer(
                "edge_post", torch.as_tensor(connectome.edge_post, dtype=torch.long))
            self.register_buffer(
                "edge_weight", torch.as_tensor(connectome.edge_weight, dtype=torch.float32))
            self.use_lora = False
        else:
            if connectome.W is None:
                raise ValueError("dense connectomes require W")
            W = torch.tensor(connectome.W, dtype=torch.float32)
            self.register_buffer("sparsity_mask", (W != 0).float())
            if trainable_W:
                self.W = nn.Parameter(W)
            else:
                self.register_buffer("W", W)

            self.use_lora = bool(use_lora)
        if self.use_lora:
            self.lora_rank = int(lora_rank)
            self.lora_scaling = lora_alpha / lora_rank
            self.lora_A = nn.Parameter(torch.empty(N, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(lora_rank, N))
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        if mode == "rate":
            tau_val = float(np.clip(tau_init, tau_min, tau_max))
            self.register_buffer(
                "rate_tau", torch.tensor(tau_val, dtype=torch.float32))
        elif mode == "ctrnn":
            self.log_tau = nn.Parameter(
                torch.full((N,), float(np.log(tau_init))))
        elif mode == "ltc":
            self.log_tau_base = nn.Parameter(
                torch.full((N,), float(np.log(tau_init))))
            self.ltc_w_mod = nn.Parameter(torch.zeros(N))
            self.ltc_b_mod = nn.Parameter(torch.zeros(N))
            self.ltc_gamma = nn.Parameter(torch.full((N,), 0.5))
        else:  # cfc
            if cfc_init == "legacy":
                self.cfc_speed_w_drive = nn.Parameter(torch.randn(N) * 0.1)
                self.cfc_speed_w_state = nn.Parameter(torch.randn(N) * 0.1)
                self.cfc_speed_bias = nn.Parameter(torch.full((N,), 2.0))
                self.cfc_offset_w_drive = nn.Parameter(torch.randn(N) * 0.1)
                self.cfc_offset_w_state = nn.Parameter(torch.randn(N) * 0.1)
                self.cfc_offset_bias = nn.Parameter(torch.zeros(N))
            elif cfc_init == "heterogeneous_tau":
                # Start as a heterogeneous rate model with the same mean tau as
                # the rate baseline. Learning then modulates the gate from state
                # and drive, instead of spending early updates undoing a bad
                # "fast-rate" initialization.
                log_tau = np.log(tau_init) + np.random.randn(N) * float(cfc_tau_log_std)
                tau = torch.clamp(torch.tensor(np.exp(log_tau), dtype=torch.float32),
                                  tau_min, tau_max)
                init_gate = torch.clamp(torch.tensor(self.dt, dtype=torch.float32) / tau,
                                        1e-4, 1.0 - 1e-4)
                self.cfc_speed_w_drive = nn.Parameter(
                    torch.randn(N) * float(cfc_gate_mod_init))
                self.cfc_speed_w_state = nn.Parameter(
                    torch.randn(N) * float(cfc_gate_mod_init))
                self.cfc_speed_bias = nn.Parameter(torch.zeros(N))
                self.cfc_offset_w_drive = nn.Parameter(
                    torch.randn(N) * float(cfc_gate_mod_init))
                self.cfc_offset_w_state = nn.Parameter(
                    torch.randn(N) * float(cfc_gate_mod_init))
                self.cfc_offset_bias = nn.Parameter(_logit(init_gate))
            else:
                raise ValueError(
                    "cfc_init must be 'legacy' or 'heterogeneous_tau'")

        self.input_proj = nn.Linear(input_dim, self.input_width)
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)

        self.output_layer = nn.Linear(self.O, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

        # Evaluation-only causal interventions.  These are deliberately not
        # buffers or parameters, so they never enter a training checkpoint.
        self._silenced_neuron_mask: Optional[torch.Tensor] = None
        self._recurrent_gain: float = 1.0

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    def _W_eff(self) -> torch.Tensor:
        if self.is_sparse:
            indices = torch.stack([self.edge_pre, self.edge_post], dim=0)
            return torch.sparse_coo_tensor(
                indices, self.edge_weight, (self.N, self.N),
                device=self.edge_weight.device).coalesce()
        W = self.W
        if self.use_lora:
            W = W + self.lora_scaling * (self.lora_A @ self.lora_B)
        return W * self.sparsity_mask

    def _recurrent_drive(self, r: torch.Tensor) -> torch.Tensor:
        if not self.is_sparse:
            return r @ self._W_eff()

        drive = torch.zeros_like(r)
        messages = r.index_select(1, self.edge_pre) * self.edge_weight
        drive.index_add_(1, self.edge_post, messages)
        return drive

    def set_silenced_neurons(self, indices: Iterable[int] | torch.Tensor) -> None:
        """Clamp selected rate units to zero during subsequent forward steps.

        Silencing is applied before recurrent transmission and after state
        integration.  Thus a lesioned unit sends no recurrent signal, cannot
        relay external input, and cannot be read out.  This is an evaluation
        intervention; it does not mutate connectivity or checkpoint tensors.
        """
        values = torch.as_tensor(
            list(indices) if not isinstance(indices, torch.Tensor) else indices,
            dtype=torch.long,
            device=self.input_proj.weight.device,
        ).flatten()
        if values.numel() and (
            int(values.min()) < 0 or int(values.max()) >= self.N
        ):
            raise IndexError(f"silenced neuron index outside [0, {self.N})")
        mask = torch.zeros(
            self.N, dtype=torch.bool, device=self.input_proj.weight.device
        )
        if values.numel():
            mask[values] = True
        self._silenced_neuron_mask = mask

    def clear_interventions(self) -> None:
        self._silenced_neuron_mask = None
        self._recurrent_gain = 1.0

    @contextmanager
    def intervention(
        self,
        *,
        silenced_neurons: Iterable[int] | torch.Tensor | None = None,
        recurrent_gain: float = 1.0,
    ) -> Iterator["ConnectomeCfC"]:
        """Temporarily apply a causal evaluation intervention."""
        previous_mask = self._silenced_neuron_mask
        previous_gain = self._recurrent_gain
        if not np.isfinite(recurrent_gain):
            raise ValueError("recurrent_gain must be finite")
        self._recurrent_gain = float(recurrent_gain)
        if silenced_neurons is not None:
            self.set_silenced_neurons(silenced_neurons)
        try:
            yield self
        finally:
            self._silenced_neuron_mask = previous_mask
            self._recurrent_gain = previous_gain

    def _gate(self, x: torch.Tensor, r: torch.Tensor,
              drive: torch.Tensor) -> torch.Tensor:
        if self.mode == "rate":
            return torch.clamp(self.dt / self.rate_tau, 0.0, 1.0)
        if self.mode == "ctrnn":
            tau = torch.clamp(torch.exp(self.log_tau),
                              self.tau_min, self.tau_max)
            return torch.clamp(self.dt / tau, 0.0, 1.0)
        if self.mode == "ltc":
            tau_base = torch.clamp(torch.exp(self.log_tau_base),
                                   self.tau_min, self.tau_max)
            mod = self.ltc_gamma * torch.sigmoid(
                self.ltc_w_mod * drive + self.ltc_b_mod)
            tau_eff = torch.clamp(tau_base * (1.0 + mod),
                                  self.tau_min, self.tau_max)
            return torch.clamp(self.dt / tau_eff, 0.0, 1.0)
        speed = (self.cfc_speed_w_drive * drive
                 + self.cfc_speed_w_state * r
                 + self.cfc_speed_bias)
        offset = (self.cfc_offset_w_drive * drive
                  + self.cfc_offset_w_state * r
                  + self.cfc_offset_bias)
        return torch.sigmoid(speed * self.dt + offset)

    def init_state(self, batch: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.edge_weight.device if self.is_sparse else self.W.device
        return 0.01 * torch.randn(batch, self.N, device=device)

    def step(self, x: torch.Tensor, inputs: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One integration step.

        Parameters
        ----------
        x       (B, N) membrane potentials
        inputs  (B, input_dim) external drive into internal neurons
        """
        mask = self._silenced_neuron_mask
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(0), 0.0)
        r = torch.tanh(x)
        drive = self._recurrent_gain * self._recurrent_drive(r)
        I_ext = torch.zeros_like(x)
        start = self.input_start
        I_ext[:, start:start + self.input_width] = self.input_proj(inputs)
        full_drive = drive + I_ext
        target = torch.tanh(full_drive)
        gate = self._gate(x, r, full_drive)
        x_next = (1.0 - gate) * x + gate * target
        if mask is not None:
            x_next = x_next.masked_fill(mask.unsqueeze(0), 0.0)
        if self.training and self.noise_std > 0.0:
            x_next = x_next + self.noise_std * torch.randn_like(x_next)
        out = self.output_layer(torch.tanh(x_next)[:, self.S + self.I:])
        return x_next, out

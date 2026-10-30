"""Policy registry: MLP, LSTM, CfC, Connectome-CfC.

A ``Policy`` exposes a tiny interface that hides whether internal state
matters:

    init_state(batch)                  -> hidden state tensor
    actor(obs, hidden) -> mean, h_next # logits for Normal(mean, sigma)
    value(obs)         -> scalar       # critic uses observation only

The PPO loop unrolls ``actor`` along time so hidden state is preserved
across the rollout but reset at episode boundaries. The critic is
intentionally state-less and shared: actor structure is the research
variable, critic is just a tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .cfc import ConnectomeCfC, GateMode
from .connectome import ConnectomeData


# ----------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------


def _activation(name: str) -> type[nn.Module]:
    """Resolve a small, explicit activation registry.

    Ant plasticity experiments use ReLU in the reference implementation,
    while the existing FlyGym locomotion experiments use Tanh.  Keeping the
    choice explicit prevents an Ant config from silently changing the legacy
    locomotion baselines.
    """
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
    }
    try:
        return activations[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {name!r}; choose from {sorted(activations)}"
        ) from exc


def mlp(in_dim: int, hidden: list[int], out_dim: int,
        act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ValueHead(nn.Module):
    """State-less MLP critic on a flat observation vector."""

    def __init__(self, obs_dim: int, hidden: list[int],
                 activation: str = "tanh") -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1, act=_activation(activation))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ----------------------------------------------------------------------
# Actor variants
# ----------------------------------------------------------------------


class _BaseActor(nn.Module):
    """Common contract: outputs raw mean (real-valued); PPO squashes."""

    has_state: bool = False
    state_dim: int = 0

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, 0, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MlpActor(_BaseActor):
    has_state = False

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int],
                 activation: str = "tanh") -> None:
        super().__init__()
        self.net = mlp(
            obs_dim, hidden, action_dim, act=_activation(activation))

    def forward(self, obs, hidden):
        return self.net(obs), hidden


class LstmActor(_BaseActor):
    has_state = True

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128,
                 mlp_hidden: Optional[list[int]] = None) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(obs_dim, hidden_size)
        self.hidden_size = hidden_size
        head_in = hidden_size
        if mlp_hidden:
            self.head = mlp(head_in, mlp_hidden, action_dim)
        else:
            self.head = nn.Linear(head_in, action_dim)
        # state layout: concat[h, c]
        self.state_dim = 2 * hidden_size

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.state_dim, device=device)

    def forward(self, obs, hidden):
        h, c = hidden.split(self.hidden_size, dim=-1)
        h, c = self.cell(obs, (h.contiguous(), c.contiguous()))
        return self.head(h), torch.cat([h, c], dim=-1)


class CfcActor(_BaseActor):
    """Dense (no connectome mask) CfC; uses the ConnectomeCfC math with a
    fully-connected adjacency. This is the "vanilla CfC" baseline that
    isolates the connectome prior contribution."""

    has_state = True

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int,
                 mode: GateMode = "cfc", dt: float = 1e-2,
                 tau_init: float = 0.05,
                 cfc_init: str = "legacy",
                 cfc_tau_log_std: float = 0.5,
                 cfc_gate_mod_init: float = 0.01) -> None:
        super().__init__()
        # Fake connectome: dense, balanced SIO partition (1/3 each).
        third = max(1, hidden_size // 3)
        N = third * 3
        W = np.random.default_rng(0).standard_normal((N, N)).astype(np.float32) * 0.1
        fake = ConnectomeData(
            W=W,
            sensory_dim=third,
            internal_dim=third,
            output_dim=third,
            sensory_ids=tuple(range(third)),
            internal_ids=tuple(range(third, 2 * third)),
            output_ids=tuple(range(2 * third, 3 * third)),
        )
        self.core = ConnectomeCfC(
            connectome=fake,
            input_dim=obs_dim,
            output_dim=action_dim,
            mode=mode,
            dt=dt,
            tau_init=tau_init,
            cfc_init=cfc_init,
            cfc_tau_log_std=cfc_tau_log_std,
            cfc_gate_mod_init=cfc_gate_mod_init,
            use_lora=False,
            trainable_W=True,
        )
        self.state_dim = N

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.core.init_state(batch, device)

    def forward(self, obs, hidden):
        h_next, mean = self.core.step(hidden, obs)
        return mean, h_next


class ConnectomeCfcActor(_BaseActor):
    """Connectome-constrained CfC actor.

    ``data`` is the ConnectomeData (possibly randomized) the policy will
    be tied to. Sparsity mask is enforced inside ConnectomeCfC.
    """

    has_state = True

    def __init__(self, data: ConnectomeData, obs_dim: int, action_dim: int,
                 mode: GateMode = "cfc", dt: float = 1e-2,
                 tau_init: float = 0.05,
                 use_lora: bool = True, lora_rank: int = 16,
                 input_target: str = "internal",
                 cfc_init: str = "legacy",
                 cfc_tau_log_std: float = 0.5,
                 cfc_gate_mod_init: float = 0.01) -> None:
        super().__init__()
        self.core = ConnectomeCfC(
            connectome=data,
            input_dim=obs_dim,
            output_dim=action_dim,
            mode=mode,
            dt=dt,
            tau_init=tau_init,
            use_lora=use_lora,
            lora_rank=lora_rank,
            input_target=input_target,
            cfc_init=cfc_init,
            cfc_tau_log_std=cfc_tau_log_std,
            cfc_gate_mod_init=cfc_gate_mod_init,
            trainable_W=False,
        )
        self.state_dim = self.core.N

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.core.init_state(batch, device)

    def forward(self, obs, hidden):
        h_next, mean = self.core.step(hidden, obs)
        return mean, h_next


class ConnectomeRateCore(nn.Module):
    """Pure sparse fixed-time-constant rate dynamics for an adult graph."""

    mode = "rate"

    def __init__(
        self, data: ConnectomeData, input_dim: int, output_dim: int,
        *, dt: float, tau: float, input_target: str,
        readout_hidden: int | None = None,
        internal_substeps: int = 1,
        input_adapter_rank: int | None = None,
        input_mixing_seed: int = 0,
        fixed_readout_by_root_id: Mapping[int | str, Sequence[float]] | None = None,
        initial_state_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not data.is_sparse:
            raise ValueError("connectome_rate requires a sparse connectome")
        if dt <= 0 or tau <= 0:
            raise ValueError("dt and tau must be positive")
        if not isinstance(internal_substeps, int) or internal_substeps <= 0:
            raise ValueError("internal_substeps must be a positive integer")
        if initial_state_std < 0.0:
            raise ValueError("initial_state_std must be non-negative")
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        self.N = data.N
        self.S = data.sensory_dim
        self.I = data.internal_dim
        self.O = data.output_dim
        if input_target == "sensory":
            self.input_start, self.input_width = 0, self.S
        elif input_target == "internal":
            self.input_start, self.input_width = self.S, self.I
        elif input_target == "all":
            self.input_start, self.input_width = 0, self.N
        else:
            raise ValueError("input_target must be sensory, internal, or all")
        self.register_buffer(
            "edge_pre", torch.as_tensor(data.edge_pre, dtype=torch.long))
        self.register_buffer(
            "edge_post", torch.as_tensor(data.edge_post, dtype=torch.long))
        self.register_buffer(
            "edge_weight", torch.as_tensor(
                data.edge_weight, dtype=torch.float32))
        self.register_buffer(
            "rate_tau", torch.tensor(float(tau), dtype=torch.float32))
        self.dt = float(dt)
        self.internal_substeps = internal_substeps
        self.initial_state_std = float(initial_state_std)
        self._recurrent_gain = 1.0
        self._external_input_gain = 1.0
        self._neural_readout_gain = 1.0
        self._deterministic_reduction = False
        self.register_buffer("_deterministic_edge_order", None, persistent=False)
        self.register_buffer("_deterministic_post_lengths", None, persistent=False)
        if input_adapter_rank is None:
            self.input_proj = nn.Linear(input_dim, self.input_width)
            self.input_mixing = None
        else:
            rank = int(input_adapter_rank)
            if rank <= 0:
                raise ValueError("input_adapter_rank must be positive")
            self.input_proj = nn.Linear(input_dim, rank)
            rng = np.random.default_rng(int(input_mixing_seed))
            mixing = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32),
                size=(rank, self.input_width),
            ).astype(np.float32)
            mixing /= np.sqrt(float(rank))
            self.register_buffer(
                "input_mixing", torch.as_tensor(mixing, dtype=torch.float32)
            )
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        if fixed_readout_by_root_id is not None:
            if readout_hidden is not None:
                raise ValueError(
                    "fixed_readout_by_root_id cannot be combined with readout_hidden"
                )
            output_index = {
                int(root_id): index
                for index, root_id in enumerate(data.output_ids)
            }
            fixed = np.zeros((self.O, output_dim), dtype=np.float32)
            unknown: list[int] = []
            for raw_root_id, raw_weights in fixed_readout_by_root_id.items():
                root_id = int(raw_root_id)
                if root_id not in output_index:
                    unknown.append(root_id)
                    continue
                weights = np.asarray(raw_weights, dtype=np.float32)
                if weights.shape != (output_dim,):
                    raise ValueError(
                        "each fixed readout row must have shape "
                        f"({output_dim},), got {weights.shape} for {root_id}"
                    )
                fixed[output_index[root_id]] = weights
            if unknown:
                raise ValueError(
                    "fixed readout contains root IDs outside output_selector: "
                    f"{sorted(unknown)}"
                )
            if not np.any(fixed):
                raise ValueError("fixed readout must contain a non-zero coefficient")
            self.register_buffer(
                "fixed_output_readout",
                torch.as_tensor(fixed, dtype=torch.float32),
            )
            self.output_layer = None
        elif readout_hidden is None:
            self.fixed_output_readout = None
            self.output_layer = nn.Linear(self.O, output_dim)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
            nn.init.zeros_(self.output_layer.bias)
        else:
            self.fixed_output_readout = None
            if readout_hidden <= 0:
                raise ValueError("readout_hidden must be positive")
            self.output_layer = nn.Sequential(
                nn.Linear(self.O, int(readout_hidden)), nn.ReLU(),
                nn.Linear(int(readout_hidden), output_dim),
            )
            for module in self.output_layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)

    def init_state(
        self, batch: int, device: torch.device,
    ) -> torch.Tensor:
        if self.initial_state_std == 0.0:
            return torch.zeros(batch, self.N, device=device)
        return self.initial_state_std * torch.randn(batch, self.N, device=device)

    def step(
        self, state: torch.Tensor, inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        external = torch.zeros_like(state)
        start = self.input_start
        projected = self.input_proj(inputs)
        if self.input_mixing is not None:
            projected = projected @ self.input_mixing
        external[:, start:start + self.input_width] = self._external_input_gain * projected
        full_step_gate = torch.clamp(self.dt / self.rate_tau, 0.0, 1.0)
        # Preserve the configured full-step time constant while allowing a
        # current sensory signal to traverse graph edges within one control
        # step.  For one substep this is exactly the historical update.
        gate = 1.0 - torch.pow(
            1.0 - full_step_gate, 1.0 / self.internal_substeps)
        next_state = state
        for _ in range(self.internal_substeps):
            rates = torch.tanh(next_state)
            recurrent = self.recurrent_drive(rates)
            target = torch.tanh(self._recurrent_gain * recurrent + external)
            next_state = (1.0 - gate) * next_state + gate * target
        output_rates = self._neural_readout_gain * torch.tanh(next_state)[:, self.S + self.I:]
        if self.fixed_output_readout is not None:
            output = output_rates @ self.fixed_output_readout
        else:
            assert self.output_layer is not None
            output = self.output_layer(output_rates)
        return next_state, output

    def recurrent_drive(self, rates: torch.Tensor) -> torch.Tensor:
        messages = rates.index_select(1, self.edge_pre) * self.edge_weight
        if self._deterministic_reduction:
            if self._deterministic_edge_order is None or self._deterministic_post_lengths is None:
                raise RuntimeError("deterministic reduction was not initialized")
            ordered = messages.index_select(1, self._deterministic_edge_order)
            return torch.stack([torch.segment_reduce(row, reduce="sum", lengths=self._deterministic_post_lengths) for row in ordered])
        drive = torch.zeros_like(rates)
        drive.index_add_(1, self.edge_post, messages)
        return drive

    def enable_deterministic_inference(self) -> None:
        """Install fixed postsynaptic summation order after any graph override.

        Training retains its native sparse reduction. These nonpersistent
        buffers move with the module and do not change checkpoint structure.
        """
        post = self.edge_post.detach().cpu().numpy()
        self._deterministic_edge_order = torch.as_tensor(np.argsort(post, kind="stable"), device=self.edge_post.device)
        self._deterministic_post_lengths = torch.as_tensor(np.bincount(post, minlength=self.N), device=self.edge_post.device)
        self._deterministic_reduction = True


class ConnectomeRateActor(_BaseActor):
    """Adult-connectome actor with fixed synapses and explicit rate dynamics."""

    has_state = True

    def __init__(
        self, data: ConnectomeData, obs_dim: int, action_dim: int,
        dt: float = 1e-2, tau_init: float = 0.02,
        use_lora: bool = False, lora_rank: int = 16,
        input_target: str = "sensory", readout_hidden: int | None = None,
        goal_adapter_hidden: int | list[int] | tuple[int, ...] | None = None,
        goal_adapter_features: int = 4,
        goal_adapter_scale: float = 1.0,
        goal_adapter_motor_scale: float = 1.0,
        connectome_output_scale: float = 1.0,
        internal_substeps: int = 1,
        input_adapter_rank: int | None = None,
        input_mixing_seed: int = 0,
        fixed_readout_by_root_id: Mapping[int | str, Sequence[float]] | None = None,
        direct_skip_hidden: int | list[int] | tuple[int, ...] | None = None,
        direct_skip_scale: float = 1.0,
        direct_skip_zero_init: bool = True,
        initial_state_std: float = 0.01,
        group_dynamics: Mapping | None = None,
    ) -> None:
        super().__init__()
        if use_lora:
            raise ValueError(
                "connectome_rate keeps the adult recurrent graph frozen; "
                "LoRA is not supported")
        del lora_rank
        core_type = ConnectomeRateCore
        core_extra = {}
        self.dynamics_manifest = None
        if group_dynamics is not None:
            from .adaptive_rate import GroupAdaptiveRateCore, load_dynamics_groups
            core_type = GroupAdaptiveRateCore
            core_extra, self.dynamics_manifest = load_dynamics_groups(data, group_dynamics)
        self.core = core_type(
            data, obs_dim, action_dim, dt=dt, tau=tau_init,
            input_target=input_target, readout_hidden=readout_hidden,
            internal_substeps=internal_substeps,
            input_adapter_rank=input_adapter_rank,
            input_mixing_seed=input_mixing_seed,
            fixed_readout_by_root_id=fixed_readout_by_root_id,
            initial_state_std=initial_state_std,
            **core_extra,
        )
        self.goal_adapter_features = int(goal_adapter_features)
        self.goal_adapter_scale = float(goal_adapter_scale)
        if self.goal_adapter_scale < 0.0:
            raise ValueError("goal_adapter_scale must be non-negative")
        self.goal_adapter_motor_scale = float(goal_adapter_motor_scale)
        self.connectome_output_scale = float(connectome_output_scale)
        self._graph_output_gain = self.connectome_output_scale
        if self.goal_adapter_motor_scale < 0.0:
            raise ValueError("goal_adapter_motor_scale must be non-negative")
        if self.connectome_output_scale < 0.0:
            raise ValueError("connectome_output_scale must be non-negative")
        self.direct_skip_scale = float(direct_skip_scale)
        # Evaluation-only causal interventions may suppress the trainable
        # bypass without mutating its parameters.  Keep this as a plain
        # scalar (rather than a buffer) so historical checkpoints remain
        # state-dict compatible.
        self._direct_skip_gain = 1.0
        if self.direct_skip_scale < 0.0:
            raise ValueError("direct_skip_scale must be non-negative")
        self.direct_skip: nn.Module | None = None
        if direct_skip_hidden is not None:
            widths = (
                [int(direct_skip_hidden)]
                if isinstance(direct_skip_hidden, int)
                else [int(width) for width in direct_skip_hidden]
            )
            if any(width <= 0 for width in widths):
                raise ValueError("direct_skip_hidden must contain positive widths")
            layers: list[nn.Module] = []
            width_in = obs_dim
            for width_out in widths:
                layer = nn.Linear(width_in, width_out)
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
                layers.extend([layer, nn.Tanh()])
                width_in = width_out
            final = nn.Linear(width_in, action_dim)
            if direct_skip_zero_init:
                nn.init.zeros_(final.weight)
            else:
                nn.init.xavier_normal_(final.weight, gain=0.1)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.direct_skip = nn.Sequential(*layers)
        self.goal_adapter: nn.Module | None = None
        if goal_adapter_hidden is not None:
            widths = (
                [int(goal_adapter_hidden)]
                if isinstance(goal_adapter_hidden, int)
                else [int(width) for width in goal_adapter_hidden]
            )
            if not widths or any(width <= 0 for width in widths):
                raise ValueError(
                    "goal_adapter_hidden must contain positive widths")
            if not 0 < self.goal_adapter_features <= obs_dim:
                raise ValueError(
                    "goal_adapter_features must be in (0, obs_dim]")
            layers: list[nn.Module] = []
            width_in = self.core.O + self.goal_adapter_features
            for width_out in widths:
                layer = nn.Linear(width_in, width_out)
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
                layers.extend([layer, nn.ReLU()])
                width_in = width_out
            final = nn.Linear(width_in, action_dim)
            # A zero residual preserves the pretrained connectome gait exactly.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.goal_adapter = nn.Sequential(*layers)
        self.state_dim = self.core.N

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.core.init_state(batch, device)

    def forward(self, obs, hidden):
        hidden_next, mean = self.core.step(hidden, obs)
        mean = self._graph_output_gain * mean
        if self.direct_skip is not None:
            mean = (
                mean
                + self._direct_skip_gain
                * self.direct_skip_scale
                * self.direct_skip(obs)
            )
        if self.goal_adapter is not None:
            motor_rates = torch.tanh(hidden_next)[:, self.core.S + self.core.I:]
            motor_rates = self.goal_adapter_motor_scale * motor_rates
            adapter_input = torch.cat(
                [motor_rates, obs[:, -self.goal_adapter_features:]], dim=1)
            mean = mean + self.goal_adapter_scale * self.goal_adapter(
                adapter_input)
        return mean, hidden_next


class GoalRouterActor(_BaseActor):
    """Learn final-goal routing while preserving a locomotion actor.

    Inputs use the AntMaze world encoding ``[..., achieved_xy, goal_delta]``.
    The router predicts a residual in normalized goal-delta space and the
    wrapped actor receives the resulting learned local goal.  No maze map or
    external waypoint is consulted at inference time.
    """

    def __init__(
        self, base: _BaseActor, obs_dim: int,
        hidden: list[int] | tuple[int, ...],
        input_features: int | str = 4,
        smoothing: float = 1.0,
        codebook: list[list[float]] | tuple[tuple[float, float], ...]
        | None = None,
        codebook_latch_radius: float | None = None,
        codebook_active_pairs: list[list[list[float]]] | None = None,
        codebook_pair_match_radius: float = 2.0,
    ) -> None:
        super().__init__()
        widths = [int(width) for width in hidden]
        if obs_dim < 4:
            raise ValueError("goal routing requires four trailing goal features")
        if input_features == "all":
            router_input_dim = obs_dim
        elif (isinstance(input_features, int)
              and not isinstance(input_features, bool)
              and 4 <= input_features <= obs_dim):
            router_input_dim = input_features
        else:
            raise ValueError(
                "goal_router_input_features must be 'all' or an integer "
                "between 4 and obs_dim")
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("goal_router_hidden must contain positive widths")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("goal_router_smoothing must be in (0, 1]")
        if (codebook_latch_radius is not None
                and codebook_latch_radius <= 0.0):
            raise ValueError("goal_router_codebook_latch_radius must be positive")
        if codebook_latch_radius is not None and smoothing != 1.0:
            raise ValueError("codebook latching and goal smoothing cannot be combined")
        self.base = base
        self.goal_router_input_features = input_features
        self._goal_router_input_dim = router_input_dim
        self.goal_router_smoothing = float(smoothing)
        self.goal_router_codebook_latch_radius = codebook_latch_radius
        if codebook is None:
            self._goal_router_codebook_world = None
        else:
            anchors = torch.as_tensor(codebook, dtype=torch.float32)
            if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 1:
                raise ValueError(
                    "goal_router_codebook must be a non-empty Nx2 array")
            self._goal_router_codebook_world = anchors
        if codebook_latch_radius is not None and codebook is None:
            raise ValueError("codebook latching requires goal_router_codebook")
        if codebook_active_pairs is None:
            self._goal_router_codebook_active_pairs = None
        else:
            pairs = torch.as_tensor(
                codebook_active_pairs, dtype=torch.float32)
            if pairs.ndim != 3 or pairs.shape[1:] != (2, 2) or len(pairs) < 1:
                raise ValueError(
                    "goal_router_codebook_active_pairs must be a non-empty "
                    "Nx2x2 array")
            if codebook_latch_radius is None:
                raise ValueError(
                    "conditional codebook routing requires codebook latching")
            if codebook_pair_match_radius <= 0.0:
                raise ValueError(
                    "goal_router_codebook_pair_match_radius must be positive")
            self._goal_router_codebook_active_pairs = pairs
        self.goal_router_codebook_pair_match_radius = float(
            codebook_pair_match_radius)
        self._route_state_dim = (
            4 if self._goal_router_codebook_active_pairs is not None
            else 3 if smoothing != 1.0 or codebook_latch_radius is not None
            else 0)
        self._goal_router_normalizer: tuple[torch.Tensor, ...] | None = None
        self.goal_router = mlp(router_input_dim, widths, 2, act=nn.ReLU)
        final = next(
            module for module in reversed(self.goal_router)
            if isinstance(module, nn.Linear))
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.has_state = bool(base.has_state or self._route_state_dim)
        self.state_dim = int(base.state_dim) + self._route_state_dim

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        base_state = self.base.init_state(batch, device)
        if not self._route_state_dim:
            return base_state
        route_state = torch.zeros(batch, self._route_state_dim, device=device)
        return torch.cat([base_state, route_state], dim=1)

    def _snap_goal_to_codebook(
        self, obs: torch.Tensor, continuous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._goal_router_codebook_world is not None
        if self._goal_router_normalizer is None:
            raise RuntimeError(
                "goal-router codebook requires normalizer calibration")
        achieved_mean, achieved_scale, delta_mean, delta_scale = (
            value.to(device=obs.device, dtype=obs.dtype)
            for value in self._goal_router_normalizer)
        achieved_world = obs[:, -4:-2] * achieved_scale + achieved_mean
        anchors = self._goal_router_codebook_world.to(
            device=obs.device, dtype=obs.dtype)
        anchor_deltas = (
            anchors.unsqueeze(0) - achieved_world.unsqueeze(1)
            - delta_mean
        ) / delta_scale
        final_world = (
            obs[:, -2:] * delta_scale + delta_mean + achieved_world)
        candidate_world = torch.cat([
            anchors.unsqueeze(0).expand(len(obs), -1, -1),
            final_world[:, None, :],
        ], dim=1)
        candidates = torch.cat([
            anchor_deltas, obs[:, None, -2:],
        ], dim=1)
        distances = torch.linalg.vector_norm(
            candidates - continuous[:, None, :], dim=-1)
        indices = distances.argmin(dim=1)[:, None, None]
        selected = candidates.gather(
            1, indices.expand(-1, 1, 2)).squeeze(1)
        selected_world = candidate_world.gather(
            1, indices.expand(-1, 1, 2)).squeeze(1)
        # Discrete forward routing with a straight-through continuous gradient
        # keeps TD3/HER refinement possible without deployment-time planning.
        return continuous + (selected - continuous).detach(), selected_world

    def route_goal(self, obs: torch.Tensor) -> torch.Tensor:
        continuous = obs[:, -2:] + self.goal_router(
            self._router_observation(obs))
        if self._goal_router_codebook_world is None:
            return continuous
        selected, _ = self._snap_goal_to_codebook(obs, continuous)
        return selected

    def _router_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.goal_router_input_features == "all":
            return obs
        return obs[:, -int(self.goal_router_input_features):]

    def configure_goal_router_normalizer(
        self, mean: torch.Tensor, variance: torch.Tensor,
    ) -> None:
        if self._goal_router_codebook_world is None:
            return
        mean = torch.as_tensor(mean, dtype=torch.float32)
        variance = torch.as_tensor(variance, dtype=torch.float32)
        if mean.ndim != 1 or variance.shape != mean.shape or len(mean) < 4:
            raise ValueError("goal-router normalizer must contain four features")
        scale = torch.sqrt(variance + 1e-8)
        self._goal_router_normalizer = (
            mean[-4:-2], scale[-4:-2], mean[-2:], scale[-2:],
        )

    def forward(self, obs, hidden):
        base_hidden = hidden[:, :self.base.state_dim]
        continuous = obs[:, -2:] + self.goal_router(
            self._router_observation(obs))
        codebook_target = None
        if self._goal_router_codebook_world is not None:
            routed_goal, codebook_target = self._snap_goal_to_codebook(
                obs, continuous)
        else:
            routed_goal = continuous
        if self.goal_router_codebook_latch_radius is not None:
            assert self._goal_router_normalizer is not None
            assert codebook_target is not None
            route_state = hidden[:, self.base.state_dim:]
            previous = route_state[:, :2]
            if self._goal_router_codebook_active_pairs is None:
                active = torch.ones_like(route_state[:, 2:3])
                initialized = route_state[:, 2:3]
            else:
                previous_active = route_state[:, 2:3]
                initialized = route_state[:, 3:4]
            achieved_mean, achieved_scale, delta_mean, delta_scale = (
                value.to(device=obs.device, dtype=obs.dtype)
                for value in self._goal_router_normalizer)
            achieved_world = obs[:, -4:-2] * achieved_scale + achieved_mean
            reached = torch.linalg.vector_norm(
                achieved_world - previous, dim=-1, keepdim=True,
            ) <= self.goal_router_codebook_latch_radius
            switch = (initialized <= 0.5) | reached
            codebook_target = torch.where(switch, codebook_target, previous)
            selected_delta = (
                codebook_target - achieved_world - delta_mean) / delta_scale
            routed_codebook = (
                continuous + (selected_delta - continuous).detach())
            if self._goal_router_codebook_active_pairs is not None:
                pairs = self._goal_router_codebook_active_pairs.to(
                    device=obs.device, dtype=obs.dtype)
                final_world = (
                    obs[:, -2:] * delta_scale + delta_mean + achieved_world)
                start_matches = torch.linalg.vector_norm(
                    achieved_world[:, None, :] - pairs[None, :, 0, :],
                    dim=-1) <= self.goal_router_codebook_pair_match_radius
                goal_matches = torch.linalg.vector_norm(
                    final_world[:, None, :] - pairs[None, :, 1, :],
                    dim=-1) <= self.goal_router_codebook_pair_match_radius
                proposed_active = (start_matches & goal_matches).any(
                    dim=1, keepdim=True).to(obs.dtype)
                active = torch.where(
                    initialized > 0.5, previous_active, proposed_active)
                routed_goal = torch.where(
                    active > 0.5, routed_codebook, continuous)
            else:
                routed_goal = routed_codebook
        elif self._route_state_dim:
            previous = hidden[:, self.base.state_dim:self.base.state_dim + 2]
            initialized = hidden[:, self.base.state_dim + 2:self.base.state_dim + 3]
            blended = (
                self.goal_router_smoothing * routed_goal
                + (1.0 - self.goal_router_smoothing) * previous)
            routed_goal = torch.where(initialized > 0.5, blended, routed_goal)
        routed = torch.cat([obs[:, :-2], routed_goal], dim=-1)
        mean, base_next = self.base(routed, base_hidden)
        if not self._route_state_dim:
            return mean, base_next
        state_value = (
            codebook_target
            if self.goal_router_codebook_latch_radius is not None
            else routed_goal)
        if self._goal_router_codebook_active_pairs is not None:
            route_next = torch.cat([
                state_value, active, torch.ones_like(initialized),
            ], dim=1)
        else:
            route_next = torch.cat([
                state_value, torch.ones_like(initialized),
            ], dim=1)
        return mean, torch.cat([base_next, route_next], dim=1)


class RoleCorrectConnectomeRateActor(_BaseActor):
    """Frozen BANC rate core with cell-role-correct input and output masks."""

    has_state = True

    def __init__(
        self,
        data: ConnectomeData,
        obs_dim: int,
        action_dim: int,
        *,
        metadata_path: str,
        dt: float = 1e-2,
        tau_init: float = 2e-2,
        selector_hidden: int = 32,
        dn_amplitude_scale: float = 0.1,
        pool_feature_scale: float = 10.0,
        noise_std: float = 0.0,
        structured_target_branch: bool = False,
        context_selector_enabled: bool = True,
        trainable_sensory_gains: bool = True,
        trainable_target_gains: bool = True,
        trainable_readout_calibration: bool = True,
        minimal_calibration_parameters: bool = False,
        residual_centering: str = "none",
        motor_readout_mode: str = "leg_pool",
        normalized_turn_readout: bool = False,
        turn_ema_tau: float = 0.1,
        normalized_turn_epsilon: float = 1e-4,
        sensory_encoder_mode: str = "legacy_proprio",
        activity_mode: str = "centered_tanh",
        tonic_rate: float = 0.1,
    ) -> None:
        super().__init__()
        # Keep the general policy registry usable without importing FlyGym's
        # dm_control/rendering stack.  role_correct -> cpg -> flygym is needed
        # only when this actor is actually selected; Ant and other external
        # Gymnasium tasks must remain independent of that optional stack.
        from .role_correct import (
            FrozenSparseRateCore,
            RoleCorrectInputAdapter,
            RoleCorrectMotorReadout,
            build_role_correct_masks,
        )

        sensory_mask, dn_mask, motor_mask, inventory = build_role_correct_masks(
            data,
            metadata_path,
            obs_dim,
            include_dn_routing=(
                bool(context_selector_enabled) or bool(structured_target_branch)
            ),
            motor_readout_mode=motor_readout_mode,
            sensory_encoder_mode=sensory_encoder_mode,
        )
        self.core = FrozenSparseRateCore(
            data,
            dt=dt,
            tau=tau_init,
            noise_std=noise_std,
            activity_mode=activity_mode,
            tonic_rate=tonic_rate,
        )
        self.input_adapter = RoleCorrectInputAdapter(
            sensory_mask,
            dn_mask,
            selector_hidden=selector_hidden,
            dn_amplitude_scale=dn_amplitude_scale,
            structured_target_branch=structured_target_branch,
            context_selector_enabled=context_selector_enabled,
            trainable_sensory_gains=trainable_sensory_gains,
            trainable_target_gains=trainable_target_gains,
            sensory_encoder_mode=sensory_encoder_mode,
        )
        self.readout = RoleCorrectMotorReadout(
            motor_mask,
            action_dim,
            pool_feature_scale=pool_feature_scale,
            normalized_turn_readout=normalized_turn_readout,
            normalized_turn_epsilon=normalized_turn_epsilon,
            trainable_calibration=trainable_readout_calibration,
            minimal_calibration_parameters=minimal_calibration_parameters,
            residual_centering=residual_centering,
        )
        self.normalized_turn_readout = bool(normalized_turn_readout)
        self._graph_output_gain = 1.0
        if self.normalized_turn_readout:
            if action_dim not in {2, 3, 14}:
                raise ValueError("normalized turn readout requires a command action")
            if turn_ema_tau <= 0:
                raise ValueError("turn_ema_tau must be positive")
            alpha = 1.0 - np.exp(-float(dt) / float(turn_ema_tau))
            self.register_buffer(
                "turn_ema_alpha", torch.tensor(alpha, dtype=torch.float32)
            )
        self.state_dim = data.N + int(self.normalized_turn_readout)
        self.routing_manifest = inventory
        self.routing_manifest.update(
            {
                "selector_hidden": int(selector_hidden),
                "dn_amplitude_scale": float(dn_amplitude_scale),
                "pool_feature_scale": float(pool_feature_scale),
                "structured_target_branch": bool(structured_target_branch),
                "context_selector_enabled": bool(context_selector_enabled),
                "trainable_sensory_gains": bool(trainable_sensory_gains),
                "trainable_target_gains": bool(trainable_target_gains),
                "trainable_readout_calibration": bool(
                    trainable_readout_calibration
                ),
                "minimal_calibration_parameters": bool(
                    minimal_calibration_parameters
                ),
                "residual_centering": str(residual_centering),
                "motor_readout_mode": str(motor_readout_mode),
                "sensory_encoder_mode": str(sensory_encoder_mode),
                "activity_mode": str(activity_mode),
                "tonic_rate": float(tonic_rate),
                "dn_side_semantics": "soma_side_not_behavioral_action_sign",
                "normalized_turn_readout": bool(normalized_turn_readout),
                "turn_ema_tau": float(turn_ema_tau),
                "normalized_turn_epsilon": float(normalized_turn_epsilon),
                "action_dim": int(action_dim),
            }
        )

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        graph_state = self.core.init_state(batch, device)
        if not self.normalized_turn_readout:
            return graph_state
        return torch.cat(
            [graph_state, torch.zeros(batch, 1, device=device)], dim=1
        )

    def forward(self, obs, hidden):
        external, _dn_amplitudes = self.input_adapter(obs)
        graph_hidden = hidden[:, : self.core.N]
        graph_next = self.core.step(graph_hidden, external)
        output_rates = self._graph_output_gain * self.core.output_activity(
            graph_next
        )[:, self.core.S + self.core.I :]
        mean, motor_pools = self.readout(output_rates)
        if not self.normalized_turn_readout:
            return mean, graph_next
        turn_previous = hidden[:, self.core.N]
        turn_raw = self.readout.normalized_turn(motor_pools)
        turn_next = turn_previous + self.turn_ema_alpha * (
            turn_raw - turn_previous
        )
        mean = mean.clone()
        mean[:, 1] = torch.exp(self.readout.normalized_turn_log_gain) * turn_next
        return mean, torch.cat([graph_next, turn_next[:, None]], dim=1)


# ----------------------------------------------------------------------
# Policy: actor + value + Gaussian distribution
# ----------------------------------------------------------------------


@dataclass
class PolicyConfig:
    name: str
    obs_dim: int
    action_dim: int
    actor_kwargs: dict
    value_hidden: list[int]
    log_std_init: float = -0.5


class Policy(nn.Module):
    """Stochastic actor-critic with optional persistent hidden state."""

    def __init__(self, actor: _BaseActor, critic: ValueHead,
                 action_dim: int, log_std_init: float = -0.5) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(log_std_init)))

    @property
    def has_state(self) -> bool:
        return self.actor.has_state

    @property
    def state_dim(self) -> int:
        return self.actor.state_dim

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.actor.init_state(batch, device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mean, hidden_next, value)."""
        mean, h_next = self.actor(obs, hidden)
        value = self.critic(obs)
        return mean, h_next, value

    def distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: torch.Tensor,
            deterministic: bool = False,
            generator: torch.Generator | None = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, h_next, value = self.forward(obs, hidden)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            return action, log_prob, value, h_next
        dist = self.distribution(mean)
        if generator is None:
            raw = dist.sample()
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            raw = mean + dist.scale * noise
        log_prob = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw)
        # tanh log-prob correction
        log_prob = log_prob - torch.sum(
            torch.log1p(-action.pow(2) + 1e-6), dim=-1)
        return action, log_prob, value, h_next

    def evaluate(self, obs: torch.Tensor, hidden: torch.Tensor,
                 actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-prob, entropy, value for an executed (obs, action)."""
        mean, h_next, value = self.forward(obs, hidden)
        dist = self.distribution(mean)
        # invert tanh; clamp for numerical safety
        a = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6)
        raw = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        log_prob = dist.log_prob(raw).sum(-1)
        log_prob = log_prob - torch.sum(
            torch.log1p(-a.pow(2) + 1e-6), dim=-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value, h_next


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def build_policy(name: str, obs_dim: int, action_dim: int, *,
                 actor_kwargs: Optional[dict] = None,
                 value_hidden: Optional[list[int]] = None,
                 value_activation: str = "tanh",
                 log_std_init: float = -0.5,
                 connectome: Optional[ConnectomeData] = None) -> Policy:
    actor_kwargs = dict(actor_kwargs or {})
    value_hidden = value_hidden or [128, 128]

    if name == "mlp":
        hidden = actor_kwargs.get("hidden", [128, 128])
        actor: _BaseActor = MlpActor(
            obs_dim, action_dim, hidden,
            activation=actor_kwargs.get("activation", "tanh"))
    elif name == "lstm":
        actor = LstmActor(
            obs_dim, action_dim,
            hidden_size=actor_kwargs.get("hidden_size", 128),
            mlp_hidden=actor_kwargs.get("mlp_hidden"))
    elif name == "cfc":
        actor = CfcActor(
            obs_dim, action_dim,
            hidden_size=actor_kwargs.get("hidden_size", 192),
            mode=actor_kwargs.get("mode", "cfc"),
            dt=actor_kwargs.get("dt", 1e-2),
            tau_init=actor_kwargs.get("tau_init", 0.05),
            cfc_init=actor_kwargs.get("cfc_init", "legacy"),
            cfc_tau_log_std=actor_kwargs.get("cfc_tau_log_std", 0.5),
            cfc_gate_mod_init=actor_kwargs.get("cfc_gate_mod_init", 0.01))
    elif name == "connectome_cfc":
        if connectome is None:
            raise ValueError("connectome_cfc requires `connectome` argument")
        actor = ConnectomeCfcActor(
            connectome, obs_dim, action_dim,
            mode=actor_kwargs.get("mode", "cfc"),
            dt=actor_kwargs.get("dt", 1e-2),
            tau_init=actor_kwargs.get("tau_init", 0.05),
            use_lora=actor_kwargs.get("use_lora", True),
            lora_rank=actor_kwargs.get("lora_rank", 16),
            input_target=actor_kwargs.get("input_target", "internal"),
            cfc_init=actor_kwargs.get("cfc_init", "legacy"),
            cfc_tau_log_std=actor_kwargs.get("cfc_tau_log_std", 0.5),
            cfc_gate_mod_init=actor_kwargs.get("cfc_gate_mod_init", 0.01))
    elif name == "connectome_rate":
        if connectome is None:
            raise ValueError("connectome_rate requires `connectome` argument")
        actor = ConnectomeRateActor(
            connectome, obs_dim, action_dim,
            dt=actor_kwargs.get("dt", 1e-2),
            tau_init=actor_kwargs.get("tau_init", 0.02),
            use_lora=actor_kwargs.get("use_lora", False),
            lora_rank=actor_kwargs.get("lora_rank", 16),
            input_target=actor_kwargs.get("input_target", "sensory"),
            readout_hidden=actor_kwargs.get("readout_hidden"),
            internal_substeps=actor_kwargs.get("internal_substeps", 1),
            goal_adapter_hidden=actor_kwargs.get("goal_adapter_hidden"),
            goal_adapter_features=actor_kwargs.get(
                "goal_adapter_features", 4),
            goal_adapter_scale=actor_kwargs.get("goal_adapter_scale", 1.0),
            goal_adapter_motor_scale=actor_kwargs.get(
                "goal_adapter_motor_scale", 1.0),
            connectome_output_scale=actor_kwargs.get(
                "connectome_output_scale", 1.0),
            input_adapter_rank=actor_kwargs.get("input_adapter_rank"),
            input_mixing_seed=actor_kwargs.get("input_mixing_seed", 0),
            fixed_readout_by_root_id=actor_kwargs.get(
                "fixed_readout_by_root_id"),
            direct_skip_hidden=actor_kwargs.get("direct_skip_hidden"),
            direct_skip_scale=actor_kwargs.get("direct_skip_scale", 1.0),
            direct_skip_zero_init=actor_kwargs.get(
                "direct_skip_zero_init", True),
            initial_state_std=actor_kwargs.get("initial_state_std", 0.01),
            group_dynamics=actor_kwargs.get("group_dynamics"))
    elif name == "role_correct_connectome_rate":
        if connectome is None:
            raise ValueError(
                "role_correct_connectome_rate requires `connectome` argument"
            )
        actor = RoleCorrectConnectomeRateActor(
            connectome,
            obs_dim,
            action_dim,
            metadata_path=actor_kwargs["metadata_path"],
            dt=actor_kwargs.get("dt", 1e-2),
            tau_init=actor_kwargs.get("tau_init", 2e-2),
            selector_hidden=actor_kwargs.get("selector_hidden", 32),
            dn_amplitude_scale=actor_kwargs.get("dn_amplitude_scale", 0.1),
            pool_feature_scale=actor_kwargs.get("pool_feature_scale", 10.0),
            noise_std=actor_kwargs.get("noise_std", 0.0),
            structured_target_branch=actor_kwargs.get(
                "structured_target_branch", False
            ),
            context_selector_enabled=actor_kwargs.get(
                "context_selector_enabled", True
            ),
            trainable_sensory_gains=actor_kwargs.get(
                "trainable_sensory_gains", True
            ),
            trainable_target_gains=actor_kwargs.get(
                "trainable_target_gains", True
            ),
            trainable_readout_calibration=actor_kwargs.get(
                "trainable_readout_calibration", True
            ),
            minimal_calibration_parameters=actor_kwargs.get(
                "minimal_calibration_parameters", False
            ),
            residual_centering=actor_kwargs.get("residual_centering", "none"),
            motor_readout_mode=actor_kwargs.get(
                "motor_readout_mode", "leg_pool"
            ),
            normalized_turn_readout=actor_kwargs.get(
                "normalized_turn_readout", False
            ),
            turn_ema_tau=actor_kwargs.get("turn_ema_tau", 0.1),
            normalized_turn_epsilon=actor_kwargs.get(
                "normalized_turn_epsilon", 1e-4
            ),
            sensory_encoder_mode=actor_kwargs.get(
                "sensory_encoder_mode", "legacy_proprio"
            ),
            activity_mode=actor_kwargs.get(
                "activity_mode", "centered_tanh"
            ),
            tonic_rate=actor_kwargs.get("tonic_rate", 0.1),
        )
    else:
        raise ValueError(f"Unknown policy: {name!r}")

    goal_router_hidden = actor_kwargs.get("goal_router_hidden")
    if goal_router_hidden is not None:
        actor = GoalRouterActor(
            actor, obs_dim, goal_router_hidden,
            input_features=actor_kwargs.get(
                "goal_router_input_features", 4),
            smoothing=actor_kwargs.get("goal_router_smoothing", 1.0),
            codebook=actor_kwargs.get("goal_router_codebook"),
            codebook_latch_radius=actor_kwargs.get(
                "goal_router_codebook_latch_radius"),
            codebook_active_pairs=actor_kwargs.get(
                "goal_router_codebook_active_pairs"),
            codebook_pair_match_radius=actor_kwargs.get(
                "goal_router_codebook_pair_match_radius", 2.0))

    critic = ValueHead(obs_dim, value_hidden, activation=value_activation)
    return Policy(actor, critic, action_dim, log_std_init)

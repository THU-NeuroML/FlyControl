"""Ant torque actions, learned goal adapter, and unchanged latched goal router."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from fly_control.graph import ConnectomeData
from fly_control.rate_network import ConnectomeRateCore
from fly_control.actor_critic import Policy, ValueHead, mlp, _activation


class _BaseActor(nn.Module):
    """Common contract: outputs raw mean (real-valued); PPO squashes."""

    has_state: bool = False
    state_dim: int = 0

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, 0, device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MlpActor(_BaseActor):
    has_state = False

    def __init__(
        self, obs_dim: int, action_dim: int, hidden: list[int], activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, action_dim, act=_activation(activation))

    def forward(self, obs, hidden):
        return self.net(obs), hidden


class ConnectomeRateActor(_BaseActor):
    """Adult-connectome actor backed only by fixed rate dynamics."""

    has_state = True

    def __init__(
        self,
        data: ConnectomeData,
        obs_dim: int,
        action_dim: int,
        dt: float = 1e-2,
        tau_init: float = 0.02,
        use_lora: bool = False,
        lora_rank: int = 16,
        input_target: str = "sensory",
        readout_hidden: int | None = None,
        goal_adapter_hidden: int | list[int] | tuple[int, ...] | None = None,
        goal_adapter_features: int = 4,
        goal_adapter_scale: float = 1.0,
        goal_adapter_motor_scale: float = 1.0,
        connectome_output_scale: float = 1.0,
        internal_substeps: int = 1,
    ) -> None:
        super().__init__()
        if use_lora:
            raise ValueError(
                "connectome_rate keeps the adult recurrent graph frozen; "
                "LoRA is not supported"
            )
        del lora_rank
        self.core = ConnectomeRateCore(
            data,
            obs_dim,
            action_dim,
            dt=dt,
            tau=tau_init,
            input_target=input_target,
            readout_hidden=readout_hidden,
            internal_substeps=internal_substeps,
        )
        self.goal_adapter_features = int(goal_adapter_features)
        self.goal_adapter_scale = float(goal_adapter_scale)
        if self.goal_adapter_scale < 0.0:
            raise ValueError("goal_adapter_scale must be non-negative")
        self.goal_adapter_motor_scale = float(goal_adapter_motor_scale)
        self.connectome_output_scale = float(connectome_output_scale)
        if self.goal_adapter_motor_scale < 0.0:
            raise ValueError("goal_adapter_motor_scale must be non-negative")
        if self.connectome_output_scale < 0.0:
            raise ValueError("connectome_output_scale must be non-negative")
        self.goal_adapter: nn.Module | None = None
        if goal_adapter_hidden is not None:
            widths = (
                [int(goal_adapter_hidden)]
                if isinstance(goal_adapter_hidden, int)
                else [int(width) for width in goal_adapter_hidden]
            )
            if not widths or any(width <= 0 for width in widths):
                raise ValueError("goal_adapter_hidden must contain positive widths")
            if not 0 < self.goal_adapter_features <= obs_dim:
                raise ValueError("goal_adapter_features must be in (0, obs_dim]")
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
        mean = self.connectome_output_scale * mean
        if self.goal_adapter is not None:
            motor_rates = torch.tanh(hidden_next)[:, self.core.S + self.core.I :]
            motor_rates = self.goal_adapter_motor_scale * motor_rates
            adapter_input = torch.cat(
                [motor_rates, obs[:, -self.goal_adapter_features :]], dim=1
            )
            mean = mean + self.goal_adapter_scale * self.goal_adapter(adapter_input)
        return mean, hidden_next


class GoalRouterActor(_BaseActor):
    """Learn final-goal routing while preserving a locomotion actor.

    Inputs use the AntMaze world encoding ``[..., achieved_xy, goal_delta]``.
    The router predicts a residual in normalized goal-delta space and the
    wrapped actor receives the resulting learned local goal.  No maze map or
    external waypoint is consulted at inference time.
    """

    def __init__(
        self,
        base: _BaseActor,
        obs_dim: int,
        hidden: list[int] | tuple[int, ...],
        input_features: int | str = 4,
        smoothing: float = 1.0,
        codebook: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
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
        elif (
            isinstance(input_features, int)
            and not isinstance(input_features, bool)
            and 4 <= input_features <= obs_dim
        ):
            router_input_dim = input_features
        else:
            raise ValueError(
                "goal_router_input_features must be 'all' or an integer "
                "between 4 and obs_dim"
            )
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("goal_router_hidden must contain positive widths")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("goal_router_smoothing must be in (0, 1]")
        if codebook_latch_radius is not None and codebook_latch_radius <= 0.0:
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
                raise ValueError("goal_router_codebook must be a non-empty Nx2 array")
            self._goal_router_codebook_world = anchors
        if codebook_latch_radius is not None and codebook is None:
            raise ValueError("codebook latching requires goal_router_codebook")
        if codebook_active_pairs is None:
            self._goal_router_codebook_active_pairs = None
        else:
            pairs = torch.as_tensor(codebook_active_pairs, dtype=torch.float32)
            if pairs.ndim != 3 or pairs.shape[1:] != (2, 2) or len(pairs) < 1:
                raise ValueError(
                    "goal_router_codebook_active_pairs must be a non-empty "
                    "Nx2x2 array"
                )
            if codebook_latch_radius is None:
                raise ValueError(
                    "conditional codebook routing requires codebook latching"
                )
            if codebook_pair_match_radius <= 0.0:
                raise ValueError(
                    "goal_router_codebook_pair_match_radius must be positive"
                )
            self._goal_router_codebook_active_pairs = pairs
        self.goal_router_codebook_pair_match_radius = float(codebook_pair_match_radius)
        self._route_state_dim = (
            4
            if self._goal_router_codebook_active_pairs is not None
            else 3 if smoothing != 1.0 or codebook_latch_radius is not None else 0
        )
        self._goal_router_normalizer: tuple[torch.Tensor, ...] | None = None
        self.goal_router = mlp(router_input_dim, widths, 2, act=nn.ReLU)
        final = next(
            module
            for module in reversed(self.goal_router)
            if isinstance(module, nn.Linear)
        )
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
        self,
        obs: torch.Tensor,
        continuous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._goal_router_codebook_world is not None
        if self._goal_router_normalizer is None:
            raise RuntimeError("goal-router codebook requires normalizer calibration")
        achieved_mean, achieved_scale, delta_mean, delta_scale = (
            value.to(device=obs.device, dtype=obs.dtype)
            for value in self._goal_router_normalizer
        )
        achieved_world = obs[:, -4:-2] * achieved_scale + achieved_mean
        anchors = self._goal_router_codebook_world.to(
            device=obs.device, dtype=obs.dtype
        )
        anchor_deltas = (
            anchors.unsqueeze(0) - achieved_world.unsqueeze(1) - delta_mean
        ) / delta_scale
        final_world = obs[:, -2:] * delta_scale + delta_mean + achieved_world
        candidate_world = torch.cat(
            [
                anchors.unsqueeze(0).expand(len(obs), -1, -1),
                final_world[:, None, :],
            ],
            dim=1,
        )
        candidates = torch.cat(
            [
                anchor_deltas,
                obs[:, None, -2:],
            ],
            dim=1,
        )
        distances = torch.linalg.vector_norm(
            candidates - continuous[:, None, :], dim=-1
        )
        indices = distances.argmin(dim=1)[:, None, None]
        selected = candidates.gather(1, indices.expand(-1, 1, 2)).squeeze(1)
        selected_world = candidate_world.gather(1, indices.expand(-1, 1, 2)).squeeze(1)
        # Discrete forward routing with a straight-through continuous gradient
        # keeps TD3/HER refinement possible without deployment-time planning.
        return continuous + (selected - continuous).detach(), selected_world

    def route_goal(self, obs: torch.Tensor) -> torch.Tensor:
        continuous = obs[:, -2:] + self.goal_router(self._router_observation(obs))
        if self._goal_router_codebook_world is None:
            return continuous
        selected, _ = self._snap_goal_to_codebook(obs, continuous)
        return selected

    def _router_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.goal_router_input_features == "all":
            return obs
        return obs[:, -int(self.goal_router_input_features) :]

    def configure_goal_router_normalizer(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> None:
        if self._goal_router_codebook_world is None:
            return
        mean = torch.as_tensor(mean, dtype=torch.float32)
        variance = torch.as_tensor(variance, dtype=torch.float32)
        if mean.ndim != 1 or variance.shape != mean.shape or len(mean) < 4:
            raise ValueError("goal-router normalizer must contain four features")
        scale = torch.sqrt(variance + 1e-8)
        self._goal_router_normalizer = (
            mean[-4:-2],
            scale[-4:-2],
            mean[-2:],
            scale[-2:],
        )

    def forward(self, obs, hidden):
        base_hidden = hidden[:, : self.base.state_dim]
        continuous = obs[:, -2:] + self.goal_router(self._router_observation(obs))
        codebook_target = None
        if self._goal_router_codebook_world is not None:
            routed_goal, codebook_target = self._snap_goal_to_codebook(obs, continuous)
        else:
            routed_goal = continuous
        if self.goal_router_codebook_latch_radius is not None:
            assert self._goal_router_normalizer is not None
            assert codebook_target is not None
            route_state = hidden[:, self.base.state_dim :]
            previous = route_state[:, :2]
            if self._goal_router_codebook_active_pairs is None:
                active = torch.ones_like(route_state[:, 2:3])
                initialized = route_state[:, 2:3]
            else:
                previous_active = route_state[:, 2:3]
                initialized = route_state[:, 3:4]
            achieved_mean, achieved_scale, delta_mean, delta_scale = (
                value.to(device=obs.device, dtype=obs.dtype)
                for value in self._goal_router_normalizer
            )
            achieved_world = obs[:, -4:-2] * achieved_scale + achieved_mean
            reached = (
                torch.linalg.vector_norm(
                    achieved_world - previous,
                    dim=-1,
                    keepdim=True,
                )
                <= self.goal_router_codebook_latch_radius
            )
            switch = (initialized <= 0.5) | reached
            codebook_target = torch.where(switch, codebook_target, previous)
            selected_delta = (
                codebook_target - achieved_world - delta_mean
            ) / delta_scale
            routed_codebook = continuous + (selected_delta - continuous).detach()
            if self._goal_router_codebook_active_pairs is not None:
                pairs = self._goal_router_codebook_active_pairs.to(
                    device=obs.device, dtype=obs.dtype
                )
                final_world = obs[:, -2:] * delta_scale + delta_mean + achieved_world
                start_matches = (
                    torch.linalg.vector_norm(
                        achieved_world[:, None, :] - pairs[None, :, 0, :], dim=-1
                    )
                    <= self.goal_router_codebook_pair_match_radius
                )
                goal_matches = (
                    torch.linalg.vector_norm(
                        final_world[:, None, :] - pairs[None, :, 1, :], dim=-1
                    )
                    <= self.goal_router_codebook_pair_match_radius
                )
                proposed_active = (
                    (start_matches & goal_matches)
                    .any(dim=1, keepdim=True)
                    .to(obs.dtype)
                )
                active = torch.where(
                    initialized > 0.5, previous_active, proposed_active
                )
                routed_goal = torch.where(active > 0.5, routed_codebook, continuous)
            else:
                routed_goal = routed_codebook
        elif self._route_state_dim:
            previous = hidden[:, self.base.state_dim : self.base.state_dim + 2]
            initialized = hidden[:, self.base.state_dim + 2 : self.base.state_dim + 3]
            blended = (
                self.goal_router_smoothing * routed_goal
                + (1.0 - self.goal_router_smoothing) * previous
            )
            routed_goal = torch.where(initialized > 0.5, blended, routed_goal)
        routed = torch.cat([obs[:, :-2], routed_goal], dim=-1)
        mean, base_next = self.base(routed, base_hidden)
        if not self._route_state_dim:
            return mean, base_next
        state_value = (
            codebook_target
            if self.goal_router_codebook_latch_radius is not None
            else routed_goal
        )
        if self._goal_router_codebook_active_pairs is not None:
            route_next = torch.cat(
                [
                    state_value,
                    active,
                    torch.ones_like(initialized),
                ],
                dim=1,
            )
        else:
            route_next = torch.cat(
                [
                    state_value,
                    torch.ones_like(initialized),
                ],
                dim=1,
            )
        return mean, torch.cat([base_next, route_next], dim=1)


class TanhPolicy(Policy):

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (mean, h_next, value) = self.forward(obs, hidden)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            return (action, log_prob, value, h_next)
        dist = self.distribution(mean)
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw)
        log_prob = log_prob - torch.sum(torch.log1p(-action.pow(2) + 1e-06), dim=-1)
        return (action, log_prob, value, h_next)

    def evaluate(
        self, obs: torch.Tensor, hidden: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-prob, entropy, value for an executed (obs, action)."""
        (mean, h_next, value) = self.forward(obs, hidden)
        dist = self.distribution(mean)
        a = torch.clamp(actions, -1 + 1e-06, 1 - 1e-06)
        raw = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        log_prob = dist.log_prob(raw).sum(-1)
        log_prob = log_prob - torch.sum(torch.log1p(-a.pow(2) + 1e-06), dim=-1)
        entropy = dist.entropy().sum(-1)
        return (log_prob, entropy, value, h_next)


def build_policy(
    name: str,
    obs_dim: int,
    action_dim: int,
    *,
    actor_kwargs: Optional[dict] = None,
    value_hidden: Optional[list[int]] = None,
    value_activation: str = "tanh",
    log_std_init: float = -0.5,
    connectome: Optional[ConnectomeData] = None,
) -> Policy:
    actor_kwargs = dict(actor_kwargs or {})
    value_hidden = value_hidden or [128, 128]
    if name in {"gru", "lstm", "transformer", "reservoir", "reservoir_ro"}:
        from .extension_actors import GruActor, LstmActor, TransformerActor, ReservoirActor
        constructors = {"gru": GruActor, "lstm": LstmActor, "transformer": TransformerActor,
                        "reservoir": ReservoirActor, "reservoir_ro": ReservoirActor}
        kwargs = {k: v for k, v in actor_kwargs.items() if not k.startswith("goal_router")}
        if name == "reservoir_ro":
            kwargs["readout_only"] = True
        actor = constructors[name](obs_dim, action_dim, **kwargs)
    elif name == "mlp":
        hidden = actor_kwargs.get("hidden", [128, 128])
        actor: _BaseActor = MlpActor(
            obs_dim,
            action_dim,
            hidden,
            activation=actor_kwargs.get("activation", "tanh"),
        )
    else:
        if name == "connectome_rate":
            if connectome is None:
                raise ValueError("connectome_rate requires `connectome` argument")
            actor = ConnectomeRateActor(
                connectome,
                obs_dim,
                action_dim,
                dt=actor_kwargs.get("dt", 0.01),
                tau_init=actor_kwargs.get("tau_init", 0.02),
                use_lora=actor_kwargs.get("use_lora", False),
                lora_rank=actor_kwargs.get("lora_rank", 16),
                input_target=actor_kwargs.get("input_target", "sensory"),
                readout_hidden=actor_kwargs.get("readout_hidden"),
                internal_substeps=actor_kwargs.get("internal_substeps", 1),
                goal_adapter_hidden=actor_kwargs.get("goal_adapter_hidden"),
                goal_adapter_features=actor_kwargs.get("goal_adapter_features", 4),
                goal_adapter_scale=actor_kwargs.get("goal_adapter_scale", 1.0),
                goal_adapter_motor_scale=actor_kwargs.get(
                    "goal_adapter_motor_scale", 1.0
                ),
                connectome_output_scale=actor_kwargs.get(
                    "connectome_output_scale", 1.0
                ),
            )
        else:
            raise ValueError(f"Unknown policy: {name!r}")
    goal_router_hidden = actor_kwargs.get("goal_router_hidden")
    if goal_router_hidden is not None:
        actor = GoalRouterActor(
            actor,
            obs_dim,
            goal_router_hidden,
            input_features=actor_kwargs.get("goal_router_input_features", 4),
            smoothing=actor_kwargs.get("goal_router_smoothing", 1.0),
            codebook=actor_kwargs.get("goal_router_codebook"),
            codebook_latch_radius=actor_kwargs.get("goal_router_codebook_latch_radius"),
            codebook_active_pairs=actor_kwargs.get("goal_router_codebook_active_pairs"),
            codebook_pair_match_radius=actor_kwargs.get(
                "goal_router_codebook_pair_match_radius", 2.0
            ),
        )
    critic = ValueHead(obs_dim, value_hidden, activation=value_activation)
    return TanhPolicy(actor, critic, action_dim, log_std_init)

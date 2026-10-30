"""Biologically role-correct adapters for the frozen BANC rate model.

The recurrent graph is never a parameter.  Observations enter through fixed
afferent and descending-neuron masks; rates leave through fixed leg-motor
pool summaries.  Only small shared gains and the external context selector
are trainable.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .cpg import LEGS, OBS_DIM, VNC_SUBTYPE_FEATURE_SLICES, VNC_SUBTYPE_OBS_DIM
from .connectome import ConnectomeData


DN_TYPES = ("DNa01", "DNa02", "DNa03", "DNp09", "MDN")
SIDES = ("left", "right")
SEGMENTS = ("front", "middle", "hind")
DN_CHANNELS = tuple(
    f"{cell_type}_{side}" for cell_type in DN_TYPES for side in SIDES
)
SIGNED_MOTOR_AXES = {
    "coxa_ap": ("move_coxa_anterior", "move_coxa_posterior"),
    "coxa_trochanter": (
        "flex_coxa_trochanter_joint", "extend_coxa_trochanter_joint"
    ),
    "femur_tibia": ("flex_femur_tibia_joint", "extend_femur_tibia_joint"),
}


def _normalised_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [
        column.strip().lower().replace(" ", "_").replace("/", "_")
        for column in frame.columns
    ]
    required = {
        "root_id",
        "body_part",
        "function",
        "class",
        "soma_side",
        "primary_cell_type",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"BANC metadata lacks columns: {sorted(missing)}")
    return frame


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _graph_indices(
    metadata: pd.DataFrame,
    mask: pd.Series,
    id_to_index: dict[int, int],
) -> np.ndarray:
    root_ids = metadata.loc[mask, "root_id"].astype(np.int64)
    return np.asarray(
        sorted(
            {
                id_to_index[int(root_id)]
                for root_id in root_ids
                if int(root_id) in id_to_index
            }
        ),
        dtype=np.int64,
    )


def build_role_correct_masks(
    connectome: ConnectomeData,
    metadata_path: str | Path,
    obs_dim: int,
    *,
    include_dn_routing: bool = True,
    motor_readout_mode: str = "leg_pool",
    sensory_encoder_mode: str = "legacy_proprio",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build fixed L1-normalised afferent, DN, and leg-MN routing masks."""
    if sensory_encoder_mode not in {"legacy_proprio", "vnc_subtype_v1"}:
        raise ValueError(
            "sensory_encoder_mode must be 'legacy_proprio' or 'vnc_subtype_v1'"
        )
    minimum_obs_dim = (
        VNC_SUBTYPE_OBS_DIM
        if sensory_encoder_mode == "vnc_subtype_v1"
        else OBS_DIM
    )
    if obs_dim < minimum_obs_dim:
        raise ValueError(
            f"role-correct BANC actor needs at least {minimum_obs_dim} observations, "
            f"got {obs_dim}"
        )
    metadata = _normalised_metadata(metadata_path)
    ordered_ids = (
        connectome.sensory_ids
        + connectome.internal_ids
        + connectome.output_ids
    )
    id_to_index = {
        int(root_id): index for index, root_id in enumerate(ordered_ids)
    }
    cell_type = metadata.primary_cell_type.fillna("").astype(str)
    side = metadata.soma_side.fillna("").astype(str).str.lower()
    body = metadata.body_part.fillna("").astype(str).str.lower()
    function = metadata.function.fillna("").astype(str).str.lower()
    class_name = metadata["class"].fillna("").astype(str).str.lower()
    function_tokens = metadata.function.fillna("").astype(str).map(
        lambda value: {
            token.strip().lower() for token in value.split(",") if token.strip()
        }
    )

    sensory_mask = np.zeros(
        (obs_dim, connectome.sensory_dim), dtype=np.float32
    )
    sensory_counts: dict[str, int] = {}
    if sensory_encoder_mode == "legacy_proprio":
        position_pattern = "proprioception|position|joint_angle|direction"
        load_pattern = "mechanical_strain|tactile|vibro"
        for leg_index, leg in enumerate(LEGS):
            leg_side = "left" if leg[0] == "L" else "right"
            segment = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
            anatomical = (side == leg_side) & (body == f"{segment}_leg")
            modality_masks = {
                "position": anatomical & function.str.contains(
                    position_pattern, regex=True
                ),
                "load": anatomical & function.str.contains(
                    load_pattern, regex=True
                ),
            }
            for modality, metadata_mask in modality_masks.items():
                indices = _graph_indices(metadata, metadata_mask, id_to_index)
                indices = indices[indices < connectome.sensory_dim]
                key = f"{modality}_{leg}"
                sensory_counts[key] = int(len(indices))
                if not len(indices):
                    raise ValueError(f"empty role-correct afferent pool: {key}")
                if modality == "position":
                    for feature in range(3 * leg_index, 3 * leg_index + 3):
                        sensory_mask[feature, indices] = 1.0 / len(indices)
                else:
                    sensory_mask[18 + leg_index, indices] = 1.0 / len(indices)
        routed_sensory_features = tuple(range(18)) + tuple(range(18, 24))
    else:
        if "sub_class" not in metadata:
            raise ValueError("VNC subtype encoder requires BANC sub_class metadata")
        sub_class = metadata.sub_class.fillna("").astype(str).str.lower()
        for leg_index, leg in enumerate(LEGS):
            leg_side = "left" if leg[0] == "L" else "right"
            segment = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
            anatomical = (side == leg_side) & (body == f"{segment}_leg")
            modality_masks = {
                "claw_hair_plate_position": anatomical & (
                    sub_class.str.contains("_claw_chordotonal_", regex=False)
                    | class_name.eq("hair_plate_neuron")
                ),
                "hook_direction_velocity": anatomical
                & sub_class.str.contains("_hook_chordotonal_", regex=False),
                "club_bidirectional_velocity": anatomical
                & sub_class.str.contains("_club_chordotonal_", regex=False),
                "campaniform_load": anatomical
                & class_name.eq("campaniform_sensillum_neuron"),
                "bristle_tactile": anatomical
                & class_name.str.contains("bristle", regex=False)
                & function_tokens.map(lambda tokens: "tactile" in tokens),
            }
            for modality, metadata_mask in modality_masks.items():
                indices = _graph_indices(metadata, metadata_mask, id_to_index)
                indices = indices[indices < connectome.sensory_dim]
                key = f"{modality}_{leg}"
                sensory_counts[key] = int(len(indices))
                if not len(indices):
                    raise ValueError(f"empty role-correct afferent pool: {key}")
                feature = VNC_SUBTYPE_FEATURE_SLICES[modality].start + leg_index
                sensory_mask[feature, indices] = 1.0 / len(indices)
        routed_sensory_features = tuple(range(30))

    dn_mask = np.zeros((len(DN_CHANNELS), connectome.N), dtype=np.float32)
    dn_counts: dict[str, int] = {}
    for channel, name in enumerate(DN_CHANNELS):
        if not include_dn_routing:
            dn_counts[name] = 0
            continue
        cell_name, channel_side = name.rsplit("_", 1)
        indices = _graph_indices(
            metadata,
            (cell_type == cell_name) & (side == channel_side),
            id_to_index,
        )
        dn_counts[name] = int(len(indices))
        if not len(indices):
            raise ValueError(f"empty role-correct DN pool: {name}")
        if np.any(indices < connectome.sensory_dim) or np.any(
            indices >= connectome.sensory_dim + connectome.internal_dim
        ):
            raise ValueError(f"DN pool is not entirely internal: {name}")
        dn_mask[channel, indices] = 1.0 / len(indices)

    if motor_readout_mode not in {"leg_pool", "signed_joint18"}:
        raise ValueError(
            "motor_readout_mode must be 'leg_pool' or 'signed_joint18'"
        )
    n_motor_channels = (
        len(LEGS)
        if motor_readout_mode == "leg_pool"
        else len(LEGS) * len(SIGNED_MOTOR_AXES)
    )
    motor_mask = np.zeros(
        (connectome.output_dim, n_motor_channels), dtype=np.float32
    )
    motor_counts: dict[str, int] = {}
    motor_output_channels: list[str] = []
    output_start = connectome.sensory_dim + connectome.internal_dim
    for leg_index, leg in enumerate(LEGS):
        leg_side = "left" if leg[0] == "L" else "right"
        segment = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
        anatomical_motor = (
            (class_name == "leg_motor_neuron")
            & (side == leg_side)
            & (body == f"{segment}_leg")
        )
        if motor_readout_mode == "leg_pool":
            indices = _graph_indices(
                metadata, anatomical_motor, id_to_index
            )
            local_indices = indices - output_start
            if not len(local_indices) or np.any(local_indices < 0) or np.any(
                local_indices >= connectome.output_dim
            ):
                raise ValueError(f"invalid role-correct motor pool: {leg}")
            motor_counts[leg] = int(len(local_indices))
            motor_mask[local_indices, leg_index] = 1.0 / len(local_indices)
            motor_output_channels.append(leg)
            continue

        for axis_index, (axis, (positive_name, negative_name)) in enumerate(
            SIGNED_MOTOR_AXES.items()
        ):
            channel = leg_index * len(SIGNED_MOTOR_AXES) + axis_index
            positive = _graph_indices(
                metadata,
                anatomical_motor
                & function_tokens.map(lambda tokens: positive_name in tokens),
                id_to_index,
            ) - output_start
            negative = _graph_indices(
                metadata,
                anatomical_motor
                & function_tokens.map(lambda tokens: negative_name in tokens),
                id_to_index,
            ) - output_start
            if (
                not len(positive)
                or not len(negative)
                or np.any(positive < 0)
                or np.any(negative < 0)
                or np.any(positive >= connectome.output_dim)
                or np.any(negative >= connectome.output_dim)
            ):
                raise ValueError(
                    f"invalid signed motor antagonists: {leg}/{axis}"
                )
            motor_mask[positive, channel] = 0.5 / len(positive)
            motor_mask[negative, channel] = -0.5 / len(negative)
            motor_counts[f"{leg}_{axis}_positive"] = int(len(positive))
            motor_counts[f"{leg}_{axis}_negative"] = int(len(negative))
            motor_output_channels.append(f"{leg}_{axis}")

    sensory_l1 = np.abs(sensory_mask).sum(axis=1)
    if not np.allclose(sensory_l1[list(routed_sensory_features)], 1.0):
        raise RuntimeError("afferent feature masks are not unit-L1")
    if include_dn_routing and not np.allclose(
        np.abs(dn_mask).sum(axis=1), 1.0
    ):
        raise RuntimeError("DN masks are not unit-L1")
    if not np.allclose(np.abs(motor_mask).sum(axis=0), 1.0):
        raise RuntimeError("motor masks are not unit-L1")

    inventory: dict[str, Any] = {
        "metadata_path": str(Path(metadata_path).resolve()),
        "dn_channels": list(DN_CHANNELS),
        "dn_routing_included": bool(include_dn_routing),
        "dn_counts": dn_counts,
        "sensory_counts": sensory_counts,
        "motor_counts": motor_counts,
        "motor_readout_mode": str(motor_readout_mode),
        "motor_output_channels": motor_output_channels,
        "sensory_encoder_mode": str(sensory_encoder_mode),
        "routed_observation_indices": list(routed_sensory_features),
        "selector_context_indices": list(range(18, obs_dim)),
        "sensory_mask_sha256": _array_sha256(sensory_mask),
        "dn_mask_sha256": _array_sha256(dn_mask),
        "motor_mask_sha256": _array_sha256(motor_mask),
    }
    return sensory_mask, dn_mask, motor_mask, inventory


class FrozenSparseRateCore(nn.Module):
    """Sparse rate dynamics whose recurrent graph exists only as buffers."""

    def __init__(
        self,
        connectome: ConnectomeData,
        *,
        dt: float = 1e-2,
        tau: float = 2e-2,
        noise_std: float = 0.0,
        activity_mode: str = "centered_tanh",
        tonic_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if not connectome.is_sparse:
            raise ValueError("role-correct BANC core requires sparse edges")
        if dt <= 0 or tau <= 0:
            raise ValueError("dt and tau must be positive")
        assert connectome.edge_pre is not None
        assert connectome.edge_post is not None
        assert connectome.edge_weight is not None
        self.N = connectome.N
        self.S = connectome.sensory_dim
        self.I = connectome.internal_dim
        self.O = connectome.output_dim
        self.noise_std = float(noise_std)
        if activity_mode not in {"centered_tanh", "nonnegative_tonic"}:
            raise ValueError(
                "activity_mode must be 'centered_tanh' or 'nonnegative_tonic'"
            )
        if not np.isfinite(tonic_rate) or tonic_rate <= 0.0:
            raise ValueError("tonic_rate must be finite and positive")
        self.activity_mode = str(activity_mode)
        self._recurrent_gain = 1.0
        self._deterministic_reduction = False
        self._deterministic_edge_order: torch.Tensor | None = None
        self._deterministic_post_lengths: torch.Tensor | None = None
        self.register_buffer(
            "edge_pre", torch.as_tensor(connectome.edge_pre, dtype=torch.long)
        )
        self.register_buffer(
            "edge_post", torch.as_tensor(connectome.edge_post, dtype=torch.long)
        )
        self.register_buffer(
            "edge_weight",
            torch.as_tensor(connectome.edge_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "integration_gate",
            torch.tensor(float(np.clip(dt / tau, 0.0, 1.0)), dtype=torch.float32),
        )
        self.register_buffer(
            "tonic_rate", torch.tensor(float(tonic_rate), dtype=torch.float32)
        )
        self.register_buffer(
            "tonic_pre_activation",
            torch.tensor(float(np.log(np.expm1(tonic_rate))), dtype=torch.float32),
        )

    def init_state(
        self, batch: int, device: torch.device | None = None
    ) -> torch.Tensor:
        target_device = device or self.edge_weight.device
        if self.activity_mode == "nonnegative_tonic":
            return self.tonic_rate.to(target_device).expand(batch, self.N).clone()
        return 0.01 * torch.randn(batch, self.N, device=target_device)

    def recurrent_drive(self, rates: torch.Tensor) -> torch.Tensor:
        messages = rates.index_select(1, self.edge_pre) * self.edge_weight
        if self._deterministic_reduction:
            if (
                self._deterministic_edge_order is None
                or self._deterministic_post_lengths is None
            ):
                raise RuntimeError("deterministic reduction was not initialized")
            ordered = messages.index_select(1, self._deterministic_edge_order)
            # CUDA index_add uses atomic additions whose rounding order changes
            # across processes.  A stable postsynaptic ordering followed by a
            # fixed segment reduction keeps the same sparse equation while
            # making closed-loop evaluation bitwise replayable.
            drive = torch.stack([
                torch.segment_reduce(
                    row,
                    reduce="sum",
                    lengths=self._deterministic_post_lengths,
                )
                for row in ordered
            ])
        else:
            drive = torch.zeros_like(rates)
            drive.index_add_(1, self.edge_post, messages)
        if self._recurrent_gain != 1.0:
            drive = drive * self._recurrent_gain
        return drive

    def enable_deterministic_inference(self) -> None:
        """Use a stable segment sum for bitwise-replayable sparse inference."""
        post = self.edge_post.detach().cpu().numpy().astype(np.int64, copy=False)
        order = np.argsort(post, kind="stable")
        lengths = np.bincount(post, minlength=self.N)
        self._deterministic_edge_order = torch.as_tensor(
            order, dtype=torch.long, device=self.edge_post.device
        )
        self._deterministic_post_lengths = torch.as_tensor(
            lengths, dtype=torch.long, device=self.edge_post.device
        )
        self._deterministic_reduction = True

    @contextmanager
    def intervention(
        self, *, recurrent_gain: float = 1.0
    ) -> Iterator["FrozenSparseRateCore"]:
        """Temporarily scale recurrent drive without changing graph buffers."""
        if not np.isfinite(recurrent_gain):
            raise ValueError("recurrent_gain must be finite")
        previous = self._recurrent_gain
        self._recurrent_gain = float(recurrent_gain)
        try:
            yield self
        finally:
            self._recurrent_gain = previous

    @contextmanager
    def graph_intervention(
        self, *, edge_post: torch.Tensor | np.ndarray
    ) -> Iterator["FrozenSparseRateCore"]:
        """Temporarily replace postsynaptic identities without editing a checkpoint."""
        override = torch.as_tensor(
            edge_post, dtype=torch.long, device=self.edge_post.device
        )
        if override.shape != self.edge_post.shape:
            raise ValueError("edge_post override must preserve the edge count")
        if override.numel() and (
            int(override.min()) < 0 or int(override.max()) >= self.N
        ):
            raise ValueError("edge_post override contains an invalid node index")
        previous = self.edge_post
        self.edge_post = override
        try:
            yield self
        finally:
            self.edge_post = previous

    def step(
        self, state: torch.Tensor, external_drive: torch.Tensor
    ) -> torch.Tensor:
        if state.shape != external_drive.shape or state.shape[-1] != self.N:
            raise ValueError("state and external_drive must both have shape (B, N)")
        if self.activity_mode == "nonnegative_tonic":
            if bool((state < 0.0).any()):
                raise FloatingPointError("nonnegative rate state became negative")
            target = torch.nn.functional.softplus(
                self.recurrent_drive(state)
                + external_drive
                + self.tonic_pre_activation
            )
        else:
            rates = torch.tanh(state)
            target = torch.tanh(self.recurrent_drive(rates) + external_drive)
        next_state = (
            (1.0 - self.integration_gate) * state
            + self.integration_gate * target
        )
        if self.training and self.noise_std > 0:
            next_state = next_state + self.noise_std * torch.randn_like(next_state)
        return next_state

    def output_activity(self, state: torch.Tensor) -> torch.Tensor:
        """Return a centered readout while retaining nonnegative internal rates."""
        if self.activity_mode == "nonnegative_tonic":
            return state - self.tonic_rate
        return torch.tanh(state)


class RoleCorrectInputAdapter(nn.Module):
    """Fixed anatomical masks plus a small external context-to-DN selector."""

    def __init__(
        self,
        sensory_mask: np.ndarray,
        dn_mask: np.ndarray,
        *,
        selector_hidden: int = 32,
        dn_amplitude_scale: float = 0.1,
        structured_target_branch: bool = False,
        context_selector_enabled: bool = True,
        trainable_sensory_gains: bool = True,
        trainable_target_gains: bool = True,
        sensory_encoder_mode: str = "legacy_proprio",
    ) -> None:
        super().__init__()
        obs_dim, _ = sensory_mask.shape
        context_indices = np.arange(18, obs_dim, dtype=np.int64)
        if context_selector_enabled and selector_hidden <= 0:
            raise ValueError("selector_hidden must be positive")
        self.register_buffer(
            "sensory_mask", torch.as_tensor(sensory_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "dn_mask", torch.as_tensor(dn_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "context_indices", torch.as_tensor(context_indices, dtype=torch.long)
        )
        self.sensory_encoder_mode = str(sensory_encoder_mode)
        if self.sensory_encoder_mode == "legacy_proprio":
            # x/y/z end-effector gains and a force gain, shared across legs.
            sensory_gain_init = torch.log(torch.tensor([0.1, 0.1, 0.1, 0.03]))
        elif self.sensory_encoder_mode == "vnc_subtype_v1":
            # Five channels were already normalized by fixed physical scales.
            sensory_gain_init = torch.zeros(5)
        else:
            raise ValueError(f"unknown sensory encoder: {sensory_encoder_mode!r}")
        if trainable_sensory_gains:
            self.log_sensory_gains = nn.Parameter(sensory_gain_init)
        else:
            self.register_buffer("log_sensory_gains", sensory_gain_init)
        self.context_selector_enabled = bool(context_selector_enabled)
        if self.context_selector_enabled:
            self.selector_norm: nn.Module | None = nn.LayerNorm(
                len(context_indices)
            )
            self.selector: nn.Module | None = nn.Sequential(
                nn.Linear(len(context_indices), selector_hidden),
                nn.Tanh(),
                nn.Linear(selector_hidden, dn_mask.shape[0]),
                nn.Tanh(),
            )
        else:
            self.selector_norm = None
            self.selector = None
        self.dn_amplitude_scale = float(dn_amplitude_scale)
        self.structured_target_branch = bool(structured_target_branch)
        self._mirror_pair_starts = (0, 2, 8)
        self._target_branch_multipliers = (1.0, 1.0, 1.0)
        if self.structured_target_branch:
            if obs_dim <= OBS_DIM:
                raise ValueError(
                    "structured target branch requires proprio_task observation"
                )
            if trainable_target_gains:
                self.target_log_gains = nn.Parameter(torch.zeros(3))
            else:
                self.register_buffer("target_log_gains", torch.zeros(3))
        else:
            self.register_parameter("target_log_gains", None)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gains = torch.exp(self.log_sensory_gains)
        scaled = torch.zeros_like(observations)
        if self.sensory_encoder_mode == "legacy_proprio":
            for axis in range(3):
                scaled[:, axis:18:3] = observations[:, axis:18:3] * gains[axis]
            scaled[:, 18:24] = observations[:, 18:24] * gains[3]
        else:
            for gain, feature_slice in zip(
                gains, VNC_SUBTYPE_FEATURE_SLICES.values()
            ):
                scaled[:, feature_slice] = observations[:, feature_slice] * gain
        sensory_drive = scaled @ self.sensory_mask
        context = observations.index_select(1, self.context_indices)
        if self.selector is None or self.selector_norm is None:
            selector_amplitudes = observations.new_zeros(
                observations.shape[0], self.dn_mask.shape[0]
            )
        else:
            selector_amplitudes = self.selector(self.selector_norm(context))
        if self.structured_target_branch:
            selector_amplitudes = selector_amplitudes.clone()
            for pair_start in self._mirror_pair_starts:
                common = selector_amplitudes[:, pair_start : pair_start + 2].mean(
                    dim=1, keepdim=True
                )
                selector_amplitudes[:, pair_start : pair_start + 2] = common
            heading_error_sin = observations[:, OBS_DIM]
            gains = torch.exp(self.target_log_gains)
            for gain_index, pair_start in enumerate(self._mirror_pair_starts):
                branch = (
                    heading_error_sin
                    * gains[gain_index]
                    * self._target_branch_multipliers[gain_index]
                )
                selector_amplitudes[:, pair_start] -= branch
                selector_amplitudes[:, pair_start + 1] += branch
        dn_amplitudes = self.dn_amplitude_scale * selector_amplitudes
        dn_drive = dn_amplitudes @ self.dn_mask
        external = dn_drive
        external[:, : sensory_drive.shape[1]] += sensory_drive
        return external, dn_amplitudes

    @contextmanager
    def target_branch_intervention(
        self, *, multipliers: tuple[float, float, float]
    ) -> Iterator["RoleCorrectInputAdapter"]:
        """Temporarily ablate or reverse fixed DNa01/DNa02/MDN target branches."""
        if not self.structured_target_branch:
            raise ValueError("target branch intervention requires the v2 adapter")
        values = tuple(float(value) for value in multipliers)
        if len(values) != len(self._mirror_pair_starts):
            raise ValueError("target branch intervention needs three multipliers")
        if not np.isfinite(values).all():
            raise ValueError("target branch multipliers must be finite")
        previous = self._target_branch_multipliers
        self._target_branch_multipliers = values
        try:
            yield self
        finally:
            self._target_branch_multipliers = previous


class RoleCorrectMotorReadout(nn.Module):
    """Fixed motor-pool summaries with shared low-dimensional calibration."""

    def __init__(
        self,
        motor_mask: np.ndarray,
        action_dim: int,
        *,
        pool_feature_scale: float = 10.0,
        normalized_turn_readout: bool = False,
        normalized_turn_epsilon: float = 1e-4,
        trainable_calibration: bool = True,
        minimal_calibration_parameters: bool = False,
        residual_centering: str = "none",
    ) -> None:
        super().__init__()
        if action_dim not in {2, 3, 6, 12, 14, 18}:
            raise ValueError(
                "role-correct readout supports command2, command3, "
                "correction6, local12, hybrid14, or signed motor18"
            )
        self.action_dim = int(action_dim)
        self.pool_feature_scale = float(pool_feature_scale)
        self.trainable_calibration = bool(trainable_calibration)
        self.minimal_calibration_parameters = bool(
            minimal_calibration_parameters
        )
        if residual_centering not in {"none", "across_legs"}:
            raise ValueError("residual_centering must be 'none' or 'across_legs'")
        if residual_centering != "none" and action_dim != 6:
            raise ValueError("residual centering is currently defined for correction6")
        self.residual_centering = str(residual_centering)
        self.normalized_turn_readout = bool(normalized_turn_readout)
        if normalized_turn_epsilon <= 0:
            raise ValueError("normalized_turn_epsilon must be positive")
        self.normalized_turn_epsilon = float(normalized_turn_epsilon)
        self.register_buffer(
            "motor_mask", torch.as_tensor(motor_mask, dtype=torch.float32)
        )
        calibrations = {
            "command_gain": torch.ones(3),
            "command_bias": torch.zeros(3),
            "local_gain": torch.ones(2),
            "local_bias": torch.zeros(2),
        }
        for name, value in calibrations.items():
            family_is_active = (
                (name.startswith("command_") and action_dim in {2, 3, 14})
                or (name.startswith("local_") and action_dim in {12, 14})
            )
            if (
                trainable_calibration
                and (
                    not self.minimal_calibration_parameters
                    or family_is_active
                )
            ):
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        # correction6 consumes only local_gain[0] and local_bias[0].  Giving it
        # dedicated scalars avoids registering two unused trainable elements and
        # makes the biological-vs-null comparison depend on exactly the same two
        # non-connectomic degrees of freedom.
        if (
            trainable_calibration
            and self.minimal_calibration_parameters
            and action_dim == 6
        ):
            self.correction_gain = nn.Parameter(torch.ones(()))
            self.correction_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("correction_gain", None)
            self.register_parameter("correction_bias", None)
        if trainable_calibration and action_dim == 18:
            self.semantic_motor_gain = nn.Parameter(torch.ones(()))
        else:
            self.register_buffer("semantic_motor_gain", torch.ones(()))
        if self.normalized_turn_readout:
            self.normalized_turn_log_gain = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("normalized_turn_log_gain", None)

    def normalized_turn(self, pools: torch.Tensor) -> torch.Tensor:
        """Fixed right-minus-left population contrast in approximately [-1, 1]."""
        left = pools[:, :3].mean(dim=1)
        right = pools[:, 3:].mean(dim=1)
        activity = pools.abs().mean(dim=1)
        return 0.5 * (right - left) / (activity + self.normalized_turn_epsilon)

    def _command(self, pools: torch.Tensor, width: int) -> torch.Tensor:
        left = pools[:, :3].mean(dim=1)
        right = pools[:, 3:].mean(dim=1)
        global_mean = pools.mean(dim=1)
        features = torch.stack(
            [global_mean, left - right, global_mean], dim=1
        )
        return (
            features[:, :width] * self.command_gain[:width]
            + self.command_bias[:width]
        )

    def _local(self, pools: torch.Tensor) -> torch.Tensor:
        retract = pools * self.local_gain[0] + self.local_bias[0]
        placement = pools * self.local_gain[1] + self.local_bias[1]
        return torch.cat([retract, placement], dim=1)

    def forward(
        self, output_rates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pools = self.pool_feature_scale * (output_rates @ self.motor_mask)
        if self.action_dim in {2, 3}:
            return self._command(pools, self.action_dim), pools
        if self.action_dim == 6:
            correction_features = pools
            if self.residual_centering == "across_legs":
                correction_features = pools - pools.mean(dim=1, keepdim=True)
            if self.correction_gain is not None:
                return (
                    correction_features * self.correction_gain
                    + self.correction_bias,
                    pools,
                )
            return (
                correction_features * self.local_gain[0] + self.local_bias[0],
                pools,
            )
        if self.action_dim == 18:
            return pools * self.semantic_motor_gain, pools
        local = self._local(pools)
        if self.action_dim == 12:
            return local, pools
        return torch.cat([self._command(pools, 2), local], dim=1), pools

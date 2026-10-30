"""CPG-driven base gait + per-leg correction mixer.

This module owns the only copy of the low-level locomotion logic.
Both env.py (training) and evaluate.py (rollout) import from here so that
training-time observation/action assembly cannot drift from evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d

from nmf_sim import Camera, Fly, SingleFlySimulation
from nmf_sim.examples.locomotion import CPGNetwork, PreprogrammedSteps
from nmf_sim.preprogrammed import get_cpg_biases

from .command import CommandToCPGAdapter, CPGModulation
from .terrain import TerrainSpec, make_arena_from_spec


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

CONTACT_SENSOR_PLACEMENTS = [
    f"{leg}{seg}"
    for leg in LEGS
    for seg in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
]

CORRECTION_VECTORS = {
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
RIGHT_LEG_INVERSION = (1, -1, -1, 1, -1, 1, 1)

OBS_DIM = 46
ACTION_DIM = 6
LOCAL_ACTION_DIM = 12
SEMANTIC_MOTOR_AXES = ("coxa_ap", "coxa_trochanter", "femur_tibia")
SEMANTIC_MOTOR_ACTION_DIM = len(LEGS) * len(SEMANTIC_MOTOR_AXES)
VNC_SUBTYPE_OBS_DIM = 52
VNC_SUBTYPE_FEATURE_SLICES = {
    "claw_hair_plate_position": slice(0, 6),
    "hook_direction_velocity": slice(6, 12),
    "club_bidirectional_velocity": slice(12, 18),
    "campaniform_load": slice(18, 24),
    "bristle_tactile": slice(24, 30),
}

# A second, independent per-leg primitive for structured local control.
# The legacy correction vector above primarily retracts/lifts a leg.  This
# vector moves the coxa in the anterior/posterior direction, allowing a local
# controller to change the next foothold without changing the global CPG.
PLACEMENT_VECTORS = {
    "F": np.array([0.03, 0, 0, 0, 0, 0, 0]),
    "M": np.array([0.03, 0, 0, 0, 0, 0, 0]),
    "H": np.array([0.03, 0, 0, 0, 0, 0, 0]),
}

# Defaults copied from ``nmf_sim.examples.locomotion.turning_fly``.  The port
# below keeps the official phase-gated stumbling/retraction algorithm while
# emitting the same low-level action dictionary as the other controllers in
# this benchmark.  This lets every system share one body, terrain, perturbation
# implementation, and traversal classifier.
OFFICIAL_HYBRID_CORRECTION_RATES = {
    "retraction": (800.0, 700.0),
    "stumbling": (2200.0, 1800.0),
}
OFFICIAL_HYBRID_MAX_INCREMENT = 80.0
OFFICIAL_HYBRID_PERSISTENCE_DURATION_STEPS = 20.0
OFFICIAL_HYBRID_PERSISTENCE_INIT_CORRECTION = 20.0


def make_arena(
    name: str | TerrainSpec,
    seed: int = 0,
    **terrain_params,
):
    """Return a nmf_sim arena instance.

    Supported names: flat, gapped, blocks, mixed, random_mixed.
    """
    if isinstance(name, TerrainSpec):
        if terrain_params:
            raise TypeError("terrain_params cannot be combined with TerrainSpec")
        spec = name
    else:
        spec = TerrainSpec(name=name, seed=seed, **terrain_params)
    return make_arena_from_spec(spec)


def make_fly(
    spawn_pos=None,
    draw_adhesion: bool = False,
    *,
    body_mass_scale: float = 1.0,
    adhesion_force_scale: float = 1.0,
) -> Fly:
    body_mass_scale = float(body_mass_scale)
    adhesion_force_scale = float(adhesion_force_scale)
    if body_mass_scale <= 0.0:
        raise ValueError("body_mass_scale must be positive")
    if adhesion_force_scale < 0.0:
        raise ValueError("adhesion_force_scale must be non-negative")
    kwargs = dict(
        enable_adhesion=True,
        adhesion_force=40.0 * adhesion_force_scale,
        draw_adhesion=draw_adhesion,
        init_pose="stretch",
        contact_sensor_placements=CONTACT_SENSOR_PLACEMENTS,
    )
    if spawn_pos is not None:
        kwargs["spawn_pos"] = spawn_pos
    fly = Fly(**kwargs)
    if body_mass_scale != 1.0:
        # The source model uses explicit geom masses. Change only inertial
        # properties, after Fly has loaded MJCF but before simulation compile;
        # geometry and collision dimensions are untouched.
        for geom in fly.model.find_all("geom"):
            if geom.mass is not None:
                geom.mass = float(geom.mass) * body_mass_scale
            elif geom.density is not None:
                geom.density = float(geom.density) * body_mass_scale
        for body in fly.model.find_all("body"):
            inertial = body.inertial
            if inertial is not None and inertial.mass is not None:
                inertial.mass = float(inertial.mass) * body_mass_scale
    return fly


def make_simulation(
    arena, fly: Fly, timestep: float, cameras=None
) -> SingleFlySimulation:
    return SingleFlySimulation(
        fly=fly,
        cameras=cameras or [],
        timestep=timestep,
        arena=arena,
    )


def make_diagnostic_camera(fly: Fly) -> Camera:
    """Create the fixed, model-blind camera used for outcome auditing."""
    return Camera(
        attachment_point=fly.model.worldbody,
        camera_name="ood_birdeye",
        camera_parameters={
            "pos": (0, 0, 12),
            "euler": (0, 0, 0),
            "fovy": 60.0,
        },
        targeted_fly_names=fly.name,
        play_speed=0.2,
    )


def make_cpg(timestep: float, seed: int = 0) -> CPGNetwork:
    phase_biases = get_cpg_biases("tripod")
    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        coupling_weights=(phase_biases > 0) * 10,
        phase_biases=phase_biases,
        convergence_coefs=np.ones(6) * 20,
        seed=seed,
    )


def phase_biases_from_offsets(offsets_cycles: np.ndarray) -> np.ndarray:
    """Build cycle-consistent phase biases from six absolute offsets."""
    offsets = np.asarray(offsets_cycles, dtype=np.float64)
    if offsets.shape != (6,):
        raise ValueError("offsets_cycles must have shape (6,)")
    delta = offsets[np.newaxis, :] - offsets[:, np.newaxis]
    return np.mod(2 * np.pi * delta, 2 * np.pi)


def gait_offsets_cycles(ipsilateral_lag_cycles: float) -> np.ndarray:
    """Return LF,LM,LH,RF,RM,RH offsets with contralateral lag 0.5."""
    lag = float(ipsilateral_lag_cycles)
    right = np.mod(np.array([0.0, lag, 2.0 * lag]), 1.0)
    left = np.mod(right + 0.5, 1.0)
    return np.concatenate([left, right])


def warp_step_phase(
    phase: float,
    swing_period: np.ndarray,
    duty_factor: float,
) -> float:
    """Warp uniform oscillator phase to a requested stance duty factor."""
    swing_start, swing_end = np.asarray(swing_period, dtype=np.float64)
    if abs(float(swing_start)) > 1e-9:
        raise ValueError("continuous gait currently requires swing_start == 0")
    duty = float(duty_factor)
    if not 0.0 < duty < 1.0:
        raise ValueError("duty_factor must be in (0, 1)")
    cycle_phase = float(phase % (2 * np.pi))
    time_fraction = cycle_phase / (2 * np.pi)
    swing_fraction = 1.0 - duty
    if time_fraction < swing_fraction:
        return float(swing_end * time_fraction / swing_fraction)
    stance_progress = (time_fraction - swing_fraction) / duty
    return float(swing_end + stance_progress * (2 * np.pi - swing_end))


def _phasic_gain(steps: PreprogrammedSteps) -> Dict[str, interp1d]:
    """Match HybridTurningController phase-modulated correction window.

    Mutates ``steps.swing_period`` to extend the swing window by pi/4.
    """
    table: Dict[str, interp1d] = {}
    for leg in LEGS:
        sw_s, sw_e = steps.swing_period[leg]
        pts = [
            sw_s,
            np.mean([sw_s, sw_e]),
            sw_e + np.pi / 4,
            np.mean([sw_e, 2 * np.pi]),
            2 * np.pi,
        ]
        steps.swing_period[leg] = (sw_s, sw_e + np.pi / 4)
        table[leg] = interp1d(pts, [0, 0.8, 0, -0.1, 0], fill_value="extrapolate")
    return table


def leg_sensor_indices(fly: Fly) -> Dict[str, np.ndarray]:
    out: Dict[str, list] = {leg: [] for leg in LEGS}
    for i, name in enumerate(fly.contact_sensor_placements):
        out[name.split("/")[1][:2]].append(i)
    return {k: np.asarray(v) for k, v in out.items()}


@dataclass
class CPGController:
    """Stateful wrapper combining CPG, base step, and learned correction.

    Holds all per-step bookkeeping that used to be duplicated across env
    and evaluation scripts. Owns a CPG, PreprogrammedSteps, the phasic
    gain table, and a leg-sensor index map for a given Fly instance.
    """

    cpg: CPGNetwork
    steps: PreprogrammedSteps
    phasic: Dict[str, interp1d]
    leg_sensors: Dict[str, np.ndarray]
    max_correction: float = 10.0
    base_intrinsic_freqs: np.ndarray | None = None
    base_intrinsic_amps: np.ndarray | None = None
    gait_coordinate: float | None = None
    duty_factor: float | None = None
    ipsilateral_lag_cycles: float | None = None
    timestep: float = 1e-4

    @classmethod
    def build(
        cls, fly: Fly, timestep: float, seed: int = 0, max_correction: float = 10.0
    ) -> "CPGController":
        steps = PreprogrammedSteps()
        phasic = _phasic_gain(steps)
        cpg = make_cpg(timestep, seed)
        return cls(
            cpg=cpg,
            steps=steps,
            phasic=phasic,
            leg_sensors=leg_sensor_indices(fly),
            max_correction=max_correction,
            base_intrinsic_freqs=cpg.intrinsic_freqs.copy(),
            base_intrinsic_amps=cpg.intrinsic_amps.copy(),
            timestep=float(timestep),
        )

    def reset(self, seed: int = 0) -> None:
        self.cpg.random_state = np.random.RandomState(seed)
        self.cpg.reset()
        self.gait_coordinate = None
        self.duty_factor = None
        self.ipsilateral_lag_cycles = None
        self.retraction_correction = np.zeros(len(LEGS), dtype=np.float64)
        self.stumbling_correction = np.zeros(len(LEGS), dtype=np.float64)
        self.retraction_persistence_counter = np.zeros(
            len(LEGS), dtype=np.float64
        )

    def step_cpg(self) -> None:
        self.cpg.step()

    def apply_command_modulation(
        self,
        command: np.ndarray,
        adapter: CommandToCPGAdapter,
    ) -> CPGModulation:
        modulation = adapter.map(command)
        base_freqs = self.base_intrinsic_freqs
        base_amps = self.base_intrinsic_amps
        assert base_freqs is not None and base_amps is not None

        frequency_gain = modulation.frequency_gain
        if modulation.gait_coordinate is not None:
            alpha = modulation.gait_smoothing_alpha
            if self.gait_coordinate is None:
                self.gait_coordinate = modulation.gait_coordinate
                self.duty_factor = modulation.duty_factor
                self.ipsilateral_lag_cycles = modulation.ipsilateral_lag_cycles
            else:
                self.gait_coordinate += alpha * (
                    modulation.gait_coordinate - self.gait_coordinate
                )
                assert self.duty_factor is not None
                assert self.ipsilateral_lag_cycles is not None
                assert modulation.duty_factor is not None
                assert modulation.ipsilateral_lag_cycles is not None
                self.duty_factor += alpha * (
                    modulation.duty_factor - self.duty_factor
                )
                self.ipsilateral_lag_cycles += alpha * (
                    modulation.ipsilateral_lag_cycles
                    - self.ipsilateral_lag_cycles
                )
                previous_gain = float(
                    self.cpg.intrinsic_freqs.mean() / base_freqs.mean()
                )
                frequency_gain = previous_gain + alpha * (
                    modulation.frequency_gain - previous_gain
                )
            offsets = gait_offsets_cycles(self.ipsilateral_lag_cycles)
            self.cpg.phase_biases = phase_biases_from_offsets(offsets)

        side_gain = np.array(
            [modulation.left_amplitude_gain] * 3
            + [modulation.right_amplitude_gain] * 3,
            dtype=np.float64,
        )
        self.cpg.intrinsic_freqs = base_freqs * frequency_gain
        self.cpg.intrinsic_amps = base_amps * side_gain * modulation.step_gain
        return modulation

    def trajectory_phase(self, leg: str, phase: float) -> float:
        if self.duty_factor is None:
            return float(phase)
        return warp_step_phase(
            phase,
            self.steps.swing_period[leg],
            self.duty_factor,
        )

    def proprio_obs(self, raw_obs) -> np.ndarray:
        """Build the canonical 46D proprioception vector from a raw obs dict.

        Layout (locked, do not change without bumping a version):
          end_effectors_rel  18
          per_leg_force       6
          fly_orientation     3
          body_z              1
          cpg_sin             6
          cpg_cos             6
          cpg_magnitude       6
        """
        fly_pos = raw_obs["fly"][0, :3]
        ee_rel = raw_obs["end_effectors"] - fly_pos
        contact = raw_obs["contact_forces"]
        per_leg = np.empty(6, dtype=np.float32)
        for i, leg in enumerate(LEGS):
            per_leg[i] = np.linalg.norm(contact[self.leg_sensors[leg]], axis=1).max()
        return np.concatenate(
            [
                ee_rel.ravel(),
                per_leg,
                raw_obs["fly_orientation"],
                [fly_pos[2]],
                np.sin(self.cpg.curr_phases),
                np.cos(self.cpg.curr_phases),
                self.cpg.curr_magnitudes,
            ]
        ).astype(np.float32)

    def vnc_subtype_obs(self, raw_obs) -> np.ndarray:
        """Build the evidence-constrained 52D VNC sensory interface.

        The five six-leg channels are, in order: femur--tibia position,
        signed femur--tibia velocity, absolute velocity, actuator load, and
        segment-contact magnitude.  They are normalized by fixed physical
        scales before the anatomical mask routes them to claw/hair-plate,
        hook, club, campaniform, and tactile-bristle populations.  Remaining
        fields expose body pose and the shared executor CPG to the critic and
        diagnostics; they are not routed into afferents by the VNC-B mask.
        """
        joints = np.asarray(raw_obs["joints"], dtype=np.float32)
        if joints.shape != (3, 42):
            raise ValueError(f"expected FlyGym joints shape (3, 42), got {joints.shape}")
        tibia = np.arange(5, 42, 7)
        position = joints[0, tibia] / np.pi
        velocity = joints[1, tibia] / 50.0
        bidirectional_velocity = np.abs(joints[1, tibia]) / 50.0
        load = np.abs(joints[2, tibia]) / 5e-9
        contact = np.asarray(raw_obs["contact_forces"], dtype=np.float32)
        per_sensor = np.linalg.norm(contact, axis=1).reshape(len(LEGS), -1)
        tactile = per_sensor.max(axis=1) / 5.0
        fly_pos = np.asarray(raw_obs["fly"][0, :3], dtype=np.float32)
        values = np.concatenate([
            position,
            velocity,
            bidirectional_velocity,
            load,
            tactile,
            np.asarray(raw_obs["fly_orientation"], dtype=np.float32),
            np.asarray([fly_pos[2]], dtype=np.float32),
            np.sin(self.cpg.curr_phases),
            np.cos(self.cpg.curr_phases),
            self.cpg.curr_magnitudes,
        ]).astype(np.float32)
        if values.shape != (VNC_SUBTYPE_OBS_DIM,):
            raise RuntimeError(f"VNC subtype observation has shape {values.shape}")
        return values

    def low_level_action(self, correction6: np.ndarray) -> Dict[str, np.ndarray]:
        """Mix the 6D correction with the CPG base joint angles.

        ``correction6`` is in [-1, 1]; scaled by ``max_correction`` and
        the per-phase gain.
        """
        correction6 = np.clip(correction6, -1.0, 1.0)
        joints = []
        adhesion = []
        for i, leg in enumerate(LEGS):
            trajectory_phase = self.trajectory_phase(leg, self.cpg.curr_phases[i])
            base = self.steps.get_joint_angles(
                leg, trajectory_phase, self.cpg.curr_magnitudes[i])
            corr = correction6[i] * self.max_correction
            if leg[0] == "R":
                corr *= RIGHT_LEG_INVERSION[i]
            corr *= self.phasic[leg](trajectory_phase % (2 * np.pi))
            base += corr * CORRECTION_VECTORS[leg[1]]
            joints.append(base)
            adhesion.append(
                self.steps.get_adhesion_onoff(leg, trajectory_phase))
        return {
            "joints": np.concatenate(joints).astype(np.float32),
            "adhesion": np.asarray(adhesion, dtype=np.int32),
        }

    def low_level_local_action(self, local12: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply two phase-gated residual primitives per leg.

        The action layout is ``[lift/retract(6), placement(6)]`` in ``LEGS``
        order.  This is deliberately separate from ``low_level_action`` so the
        legacy ``correction6`` numerical path remains unchanged.
        """
        local12 = np.asarray(local12, dtype=np.float32)
        if local12.shape != (LOCAL_ACTION_DIM,):
            raise ValueError(f"local action must have shape ({LOCAL_ACTION_DIM},)")
        retract = np.clip(local12[:ACTION_DIM], -1.0, 1.0)
        placement = np.clip(local12[ACTION_DIM:], -1.0, 1.0)
        joints = []
        adhesion = []
        for i, leg in enumerate(LEGS):
            base = self.steps.get_joint_angles(
                leg, self.cpg.curr_phases[i], self.cpg.curr_magnitudes[i]
            )
            mirror = np.asarray(
                RIGHT_LEG_INVERSION if leg[0] == "R" else np.ones(7)
            )
            phase_gain = float(self.phasic[leg](self.cpg.curr_phases[i] % (2 * np.pi)))
            residual = (
                retract[i] * CORRECTION_VECTORS[leg[1]]
                + placement[i] * PLACEMENT_VECTORS[leg[1]]
            )
            base += self.max_correction * phase_gain * residual * mirror
            joints.append(base)
            adhesion.append(
                self.steps.get_adhesion_onoff(leg, self.cpg.curr_phases[i])
            )
        return {
            "joints": np.concatenate(joints).astype(np.float32),
            "adhesion": np.asarray(adhesion, dtype=np.int32),
        }

    def low_level_semantic_motor_action(
        self, motor18: np.ndarray, *, phase_gated: bool = False
    ) -> Dict[str, np.ndarray]:
        """Apply fixed, anatomically named antagonistic motor residuals.

        Per leg the three coordinates are coxa anterior/posterior motion,
        coxa--trochanter flexion/extension, and femur--tibia
        flexion/extension. They map to the corresponding FlyGym Coxa, Femur,
        and Tibia joint coordinates.  The 0.03-radian unit scale matches the
        order of the existing correction vectors; ``max_correction`` remains
        the shared environment-level bound.
        """
        motor18 = np.asarray(motor18, dtype=np.float32)
        if motor18.shape != (SEMANTIC_MOTOR_ACTION_DIM,):
            raise ValueError(
                "semantic motor action must have shape "
                f"({SEMANTIC_MOTOR_ACTION_DIM},)"
            )
        per_leg = np.clip(motor18.reshape(len(LEGS), -1), -1.0, 1.0)
        joints = []
        adhesion = []
        for leg_index, leg in enumerate(LEGS):
            trajectory_phase = self.trajectory_phase(
                leg, self.cpg.curr_phases[leg_index]
            )
            base = self.steps.get_joint_angles(
                leg, trajectory_phase, self.cpg.curr_magnitudes[leg_index]
            )
            residual = np.zeros(7, dtype=np.float32)
            residual[[0, 3, 5]] = 0.03 * per_leg[leg_index]
            if phase_gated:
                residual *= float(
                    self.phasic[leg](trajectory_phase % (2 * np.pi))
                )
            base += self.max_correction * residual
            joints.append(base)
            adhesion.append(
                self.steps.get_adhesion_onoff(leg, trajectory_phase)
            )
        return {
            "joints": np.concatenate(joints).astype(np.float32),
            "adhesion": np.asarray(adhesion, dtype=np.int32),
        }

    def prepare_official_hybrid_reflex(
        self, raw_obs: dict, descending_command: np.ndarray
    ) -> None:
        """Update the official FlyGym hybrid reflex state before a CPG step.

        The implementation follows ``HybridTurningFly.pre_step``: the two
        descending coordinates set side-specific CPG amplitude/direction;
        body/foot geometry triggers retraction; tibia/tarsus contact projected
        against body heading triggers stumbling.  The CPG step itself remains
        in :class:`TerrainLocomotionEnv` so registered CPG perturbations can be
        applied identically to every controller.
        """
        command = np.asarray(descending_command, dtype=np.float64)
        if command.shape != (2,):
            raise ValueError("official hybrid command must have shape (2,)")
        base_freqs = self.base_intrinsic_freqs
        base_amps = self.base_intrinsic_amps
        assert base_freqs is not None and base_amps is not None
        self.cpg.intrinsic_amps = base_amps * np.repeat(
            np.abs(command[:, None]), 3, axis=1
        ).ravel()
        frequencies = base_freqs.copy()
        frequencies[:3] *= 1.0 if command[0] > 0.0 else -1.0
        frequencies[3:] *= 1.0 if command[1] > 0.0 else -1.0
        self.cpg.intrinsic_freqs = frequencies

        relative_foot_height = (
            float(raw_obs["fly"][0][2])
            - np.asarray(raw_obs["end_effectors"], dtype=np.float64)[:, 2]
        )
        sorted_indices = np.argsort(relative_foot_height)
        sorted_heights = relative_foot_height[sorted_indices]
        retraction_leg: int | None = None
        if sorted_heights[-1] > sorted_heights[-3] + 0.05:
            retraction_leg = int(sorted_indices[-1])
            if (
                self.retraction_correction[retraction_leg]
                > OFFICIAL_HYBRID_PERSISTENCE_INIT_CORRECTION
            ):
                self.retraction_persistence_counter[retraction_leg] = 1.0

        active = self.retraction_persistence_counter > 0.0
        self.retraction_persistence_counter[active] += 1.0
        expired = (
            self.retraction_persistence_counter
            > (OFFICIAL_HYBRID_PERSISTENCE_DURATION_STEPS
               if getattr(self, "official_persistence_duration_s", None) is None
               else self.official_persistence_duration_s / self.timestep)
        )
        self.retraction_persistence_counter[expired] = 0.0
        persistent = self.retraction_persistence_counter > 0.0

        contact = np.asarray(raw_obs["contact_forces"], dtype=np.float64)
        orientation = np.asarray(raw_obs["fly_orientation"], dtype=np.float64)
        for index, leg in enumerate(LEGS):
            retract = index == retraction_leg or bool(persistent[index])
            self.retraction_correction[index] = self._official_reflex_update(
                retract,
                self.retraction_correction[index],
                OFFICIAL_HYBRID_CORRECTION_RATES["retraction"],
            )
            # CONTACT_SENSOR_PLACEMENTS is ordered Tibia, Tarsus1, Tarsus2,
            # ... for every leg, exactly matching the official stumble set.
            stumble_indices = self.leg_sensors[leg][:3]
            force_projection = contact[stumble_indices] @ orientation
            stumbling = bool((force_projection < -1.0).any())
            self.stumbling_correction[index] = self._official_reflex_update(
                stumbling,
                self.stumbling_correction[index],
                OFFICIAL_HYBRID_CORRECTION_RATES["stumbling"],
            )

    def _official_reflex_update(
        self,
        condition: bool,
        current: float,
        rates: tuple[float, float],
    ) -> float:
        if condition:
            return float(current + rates[0] * self.timestep)
        return float(max(0.0, current - rates[1] * self.timestep))

    def low_level_official_hybrid_action(self) -> Dict[str, np.ndarray]:
        """Emit the official hybrid reflex action after the CPG was stepped."""
        joints = []
        adhesion = []
        for index, leg in enumerate(LEGS):
            retraction = float(self.retraction_correction[index])
            if retraction > 0.0:
                correction = retraction
                self.stumbling_correction[index] = 0.0
            else:
                correction = float(self.stumbling_correction[index])
            correction = float(np.clip(
                correction, 0.0, OFFICIAL_HYBRID_MAX_INCREMENT
            ))
            # Preserve the scalar right-side inversion used by the upstream
            # implementation rather than substituting our residual mixer.
            if leg[0] == "R":
                correction *= RIGHT_LEG_INVERSION[index]
            phase = float(self.cpg.curr_phases[index])
            correction *= float(self.phasic[leg](phase % (2 * np.pi)))
            base = self.steps.get_joint_angles(
                leg, phase, self.cpg.curr_magnitudes[index]
            )
            base += correction * CORRECTION_VECTORS[leg[1]]
            joints.append(base)
            adhesion.append(self.steps.get_adhesion_onoff(leg, phase))
        return {
            "joints": np.concatenate(joints).astype(np.float32),
            "adhesion": np.asarray(adhesion, dtype=np.int32),
        }

    def apply_single_leg_joint_bias(
        self, low: Dict[str, np.ndarray], leg: str, magnitude: float
    ) -> Dict[str, np.ndarray]:
        """Apply an opposing retraction-vector bias to one leg's targets."""
        if leg not in LEGS:
            raise ValueError(f"unknown leg: {leg}")
        idx = LEGS.index(leg)
        joints = np.asarray(low["joints"], dtype=np.float32).copy()
        phase_gain = float(
            self.phasic[leg](self.cpg.curr_phases[idx] % (2 * np.pi))
        )
        mirror = np.asarray(
            RIGHT_LEG_INVERSION if leg[0] == "R" else np.ones(7)
        )
        bias = (
            float(magnitude)
            * self.max_correction
            * phase_gain
            * CORRECTION_VECTORS[leg[1]]
            * mirror
        )
        start = idx * 7
        joints[start:start + 7] -= bias.astype(np.float32)
        return {**low, "joints": joints}

    def low_level_command_action(
        self,
        command: np.ndarray,
        adapter: CommandToCPGAdapter,
    ) -> Dict[str, np.ndarray]:
        """Apply command modulation and emit the CPG-only low-level action."""
        self.apply_command_modulation(command, adapter)
        return self.low_level_action(np.zeros(6, dtype=np.float32))

"""NeuromechFly CPG plus six coordinated leg-correction residuals.

Each leg exposes its bundled correction template, normalized to joint-angle L2 bound0.15*sqrt(3)rad.
Right-side joint coordinates are mirrored individually. No residual phase
mask limits contact/posture correction. The body, base stepping and shared
adhesion schedule remain unchanged; the fresh finite task supplies reward.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import interp1d
from dm_control.rl.control import PhysicsError

from nmf_sim import Fly, SingleFlySimulation
from nmf_sim.arena import (
    FlatTerrain,
    GappedTerrain,
    BlocksTerrain,
    MixedTerrain,
)
from nmf_sim.preprogrammed import all_leg_dofs, get_cpg_biases
from nmf_sim.examples.locomotion import CPGNetwork, PreprogrammedSteps


CORRECTION_VECTORS = {
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
RIGHT_LEG_INVERSION = [1, -1, -1, 1, -1, 1, 1]
LEG_SYNERGY_L2_BOUND = 0.15 * np.sqrt(3.0)
NORMALIZED_CORRECTION_VECTORS = {
    kind: LEG_SYNERGY_L2_BOUND * vector / np.linalg.norm(vector)
    for kind, vector in CORRECTION_VECTORS.items()
}
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]

CONTACT_SENSOR_PLACEMENTS = [
    f"{leg}{segment}"
    for leg in LEGS
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

TERRAIN_CHOICES = ["flat", "gapped", "blocks", "mixed"]


def _make_arena(terrain_name: str, seed: int = 0):
    if terrain_name == "flat":
        return FlatTerrain()
    elif terrain_name == "gapped":
        return GappedTerrain()
    elif terrain_name == "blocks":
        return BlocksTerrain(rand_seed=seed)
    elif terrain_name == "mixed":
        return MixedTerrain(rand_seed=seed)
    raise ValueError(f"Unknown terrain: {terrain_name}")


class CPGRLEnv(gym.Env):
    """
    Gymnasium environment: CPG walking + learned leg corrections.

    One RL step = ``decision_interval`` physics steps (default 100 = 10 ms).

    Observation (46D):
        end_effectors_rel 18 + contact_forces 6 + orientation 3 +
        body_z 1 + cpg_sin 6 + cpg_cos 6 + cpg_mag 6

    Action (6D):
        Six signed, normalized multijoint leg corrections.

    Reward:
        + forward displacement (mm) over the decision interval
        - flip_penalty on flip (terminates)
        - small energy penalty
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        terrain: str = "mixed",
        timestep: float = 1e-4,
        max_time: float = 1.5,
        decision_interval: int = 100,
        max_correction: float = 10.0,
        correction_energy_penalty: float = 1e-3,
        flip_penalty: float = 10.0,
        render_mode: str = None,
        terrain_seed: int = 0,
        randomize_terrain: bool = True,
    ):
        super().__init__()

        self.terrain_name = terrain
        self.timestep = timestep
        self.max_time = max_time
        self.decision_interval = decision_interval
        self.max_correction = max_correction
        self.correction_energy_penalty = correction_energy_penalty
        self.flip_penalty = flip_penalty
        self.render_mode = render_mode
        self.terrain_seed = terrain_seed
        self.randomize_terrain = randomize_terrain

        self._obs_dim = 46
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._sim = None
        self._cpg = None
        self._preprogrammed_steps = None
        self._phasic_multiplier = None
        self._leg_sensors = None
        self._physics_steps = 0
        self._prev_fly_pos = None
        self._initial_heading = None

    def _build_sim(self, terrain_name: str, seed: int):
        arena = _make_arena(terrain_name, seed)

        fly = Fly(
            enable_adhesion=True,
            draw_adhesion=False,
            init_pose="stretch",
            contact_sensor_placements=CONTACT_SENSOR_PLACEMENTS,
        )

        self._sim = SingleFlySimulation(
            fly=fly,
            cameras=[],
            timestep=self.timestep,
            arena=arena,
        )

        self._preprogrammed_steps = PreprogrammedSteps()

        intrinsic_freqs = np.ones(6) * 12
        intrinsic_amps = np.ones(6) * 1
        phase_biases = get_cpg_biases("tripod")
        coupling_weights = (phase_biases > 0) * 10
        convergence_coefs = np.ones(6) * 20

        self._cpg = CPGNetwork(
            timestep=self.timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            seed=seed,
        )

        # Retain the identical shared CPG/adhesion schedule; residuals below
        # do not use the legacy one-direction phase gain.
        self._phasic_multiplier = self._init_phasic_gain()
        self._leg_sensors = self._find_leg_sensors()

    def _init_phasic_gain(self):
        phasic = {}
        swing_ext = np.pi / 4
        for leg in LEGS:
            swing_start, swing_end = self._preprogrammed_steps.swing_period[leg]
            step_points = [
                swing_start,
                np.mean([swing_start, swing_end]),
                swing_end + swing_ext,
                np.mean([swing_end, 2 * np.pi]),
                2 * np.pi,
            ]
            self._preprogrammed_steps.swing_period[leg] = (
                swing_start,
                swing_end + swing_ext,
            )
            increment_vals = [0, 0.8, 0, -0.1, 0]
            phasic[leg] = interp1d(
                step_points, increment_vals, kind="linear", fill_value="extrapolate"
            )
        return phasic

    def _find_leg_sensors(self):
        sensors = {leg: [] for leg in LEGS}
        for i, name in enumerate(self._sim.fly.contact_sensor_placements):
            leg = name.split("/")[1][:2]
            sensors[leg].append(i)
        return {k: np.array(v) for k, v in sensors.items()}

    def _extract_obs(self, raw_obs) -> np.ndarray:
        fly_pos = raw_obs["fly"][0, :3]

        ee_rel = raw_obs["end_effectors"] - fly_pos

        cf = raw_obs["contact_forces"]
        per_leg_force = np.zeros(6)
        for i, leg in enumerate(LEGS):
            idx = self._leg_sensors[leg]
            per_leg_force[i] = np.linalg.norm(cf[idx], axis=1).max()

        obs = np.concatenate(
            [
                ee_rel.ravel(),
                per_leg_force,
                raw_obs["fly_orientation"],
                [fly_pos[2]],
                np.sin(self._cpg.curr_phases),
                np.cos(self._cpg.curr_phases),
                self._cpg.curr_magnitudes,
            ]
        ).astype(np.float32)

        return obs

    @staticmethod
    def _has_nan(arr):
        return not np.isfinite(arr).all()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_terrain:
            terrain = self.np_random.choice(TERRAIN_CHOICES)
        else:
            terrain = self.terrain_name
        env_seed = seed if seed is not None else self.terrain_seed

        self._build_sim(terrain, env_seed)
        raw_obs, _ = self._sim.reset(seed=env_seed)
        self._cpg.random_state = np.random.RandomState(env_seed)
        self._cpg.reset()

        self._physics_steps = 0
        self._prev_fly_pos = raw_obs["fly"][0, :3].copy()
        heading = raw_obs["fly_orientation"][:2].copy()
        norm = np.linalg.norm(heading)
        self._initial_heading = heading / norm if norm > 1e-6 else np.array([1.0, 0.0])

        return self._extract_obs(raw_obs), {}

    def step(self, action: np.ndarray):
        """Execute ``decision_interval`` physics steps with the same action."""
        action = np.clip(action, -1.0, 1.0)
        # Read-only summaries of the complete physical control interval.
        # They do not change targets, CPG, adhesion, or simulator stepping.
        self._interval_force_min = np.full(len(CONTACT_SENSOR_PLACEMENTS), np.inf)
        self._interval_force_peak_norm = np.zeros(len(CONTACT_SENSOR_PLACEMENTS))
        self._interval_force_samples = 0

        interval_reward = 0.0
        terminated = False
        truncated = False
        raw_obs = None
        info = {}

        for _ in range(self.decision_interval):
            self._cpg.step()

            joints_angles = []
            adhesion_onoff = []
            for i, leg in enumerate(LEGS):
                base = self._preprogrammed_steps.get_joint_angles(
                    leg,
                    self._cpg.curr_phases[i],
                    self._cpg.curr_magnitudes[i],
                )
                # One coordinated leg correction across all template joints.
                # Mirrored controls have the same local meaning on both sides.
                delta = action[i] * NORMALIZED_CORRECTION_VECTORS[leg[1]]
                if leg[0] == "R":
                    delta = delta * np.asarray(RIGHT_LEG_INVERSION)
                base += delta
                joints_angles.append(base)
                adhesion_onoff.append(
                    self._preprogrammed_steps.get_adhesion_onoff(
                        leg, self._cpg.curr_phases[i]
                    )
                )

            low_level = {
                "joints": np.concatenate(joints_angles).astype(np.float32),
                "adhesion": np.array(adhesion_onoff, dtype=np.int32),
            }

            try:
                raw_obs, _, _, _, info = self._sim.step(low_level)
            except PhysicsError:
                obs = np.zeros(self._obs_dim, dtype=np.float32)
                return (
                    obs,
                    interval_reward - self.flip_penalty,
                    True,
                    False,
                    {"physics_error": True},
                )

            self._physics_steps += 1

            # NaN guard
            fly_arr = raw_obs["fly"]
            if self._has_nan(fly_arr):
                obs = np.zeros(self._obs_dim, dtype=np.float32)
                return (
                    obs,
                    interval_reward - self.flip_penalty,
                    True,
                    False,
                    {"nan_detected": True},
                )

            forces = raw_obs["contact_forces"]
            orientation = raw_obs["fly_orientation"]
            if self._has_nan(forces) or self._has_nan(orientation):
                return (
                    np.zeros(self._obs_dim, dtype=np.float32),
                    interval_reward - self.flip_penalty,
                    True,
                    False,
                    {"nan_detected": True},
                )
            np.minimum(
                self._interval_force_min,
                forces @ orientation,
                out=self._interval_force_min,
            )
            np.maximum(
                self._interval_force_peak_norm,
                np.linalg.norm(forces, axis=1),
                out=self._interval_force_peak_norm,
            )
            self._interval_force_samples += 1

            fly_pos = fly_arr[0, :3]
            disp = fly_pos[:2] - self._prev_fly_pos[:2]
            interval_reward += np.dot(disp, self._initial_heading)
            self._prev_fly_pos = fly_pos.copy()

            if info.get("flip", False):
                terminated = True
                break
            if (self._physics_steps * self.timestep) >= self.max_time:
                truncated = True
                break

        # Scale displacement reward to mm (flygym uses mm internally)
        reward = interval_reward * 100.0

        if terminated:
            reward -= self.flip_penalty

        energy_cost = np.sum(action**2) * self.correction_energy_penalty
        reward -= energy_cost

        if raw_obs is not None:
            obs = self._extract_obs(raw_obs)
            if self._has_nan(obs):
                obs = np.zeros(self._obs_dim, dtype=np.float32)
                return (
                    obs,
                    reward - self.flip_penalty,
                    True,
                    False,
                    {"nan_detected": True},
                )
        else:
            obs = np.zeros(self._obs_dim, dtype=np.float32)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._sim is not None:
            return self._sim.render()
        return None

"""Finite-horizon single-obstacle locomotion in the frozen real fly body.

No teacher, demonstration, or body parameter change. Both architectures use
the same 6-D coordinated leg actuator and172-D sensor frame; the final wrapper suppliesfour causal frames (688D). Geometries contain
one gap or raised rectangular obstacle followed by a flat landing platform.
"""

from .context import ROOT, CONFIG, write_json
import copy, json
import numpy as np
import gymnasium as gym
from nmf_sim.arena import BaseArena
from . import physical_interface as physical
from .onset import map_action

CALIBRATION_CELLS = [("gapped", v) for v in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]] + [
    ("blocks", v) for v in [0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
]


class SingleObstacleArena(BaseArena):
    friction = (1.0, 0.005, 0.0001)

    def __init__(self, terrain, value, near=3.0, block_width=1.3):
        super().__init__()
        self.terrain, self.value, self.near = terrain, value, near
        self.far = near + (value if terrain == "gapped" else block_width)

        def box(name, left, right, top, bottom=-2.0):
            self.root_element.worldbody.add(
                "geom",
                name=name,
                type="box",
                pos=((left + right) / 2, 0, (top + bottom) / 2),
                size=((right - left) / 2, 20, (top - bottom) / 2),
                friction=self.friction,
                rgba=(0.35, 0.35, 0.35, 1.0),
            )

        if terrain == "gapped":
            box("start_platform", -10, near, 0.0)
            box("landing_platform", self.far, 25.0, 0.0)
            self.root_element.worldbody.add(
                "geom",
                type="plane",
                name="gap_bottom",
                pos=(0, 0, -2.0),
                size=(25, 20, 1),
                friction=self.friction,
                rgba=(0.2, 0.2, 0.2, 1.0),
            )
        elif terrain == "blocks":
            box("ground", -10, 25, 0.0)
            if value > 0.0:
                box("single_obstacle", near, self.far, value, bottom=0.0)
            elif value < 0.0:
                raise ValueError("Block height must be non-negative")
        else:
            raise ValueError(terrain)

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return self.value if self.terrain == "blocks" else 0.0


class ObstacleTask(physical.CPGRLEnv):
    def __init__(
        self,
        cells=None,
        fixed_cell=None,
        horizon=1.2,
        phase_mode="tripod",
        rank=0,
        n_envs=1,
        log_path=None,
        objective="crossing",
        cache_physics=True,
        **kw,
    ):
        super().__init__(
            terrain="gapped",
            randomize_terrain=False,
            timestep=5e-5,
            decision_interval=200,
            max_time=horizon,
            max_correction=10.0,
            **kw,
        )
        self.cells = list(cells or CALIBRATION_CELLS)
        self.fixed_cell = fixed_cell
        self.phase_mode = phase_mode
        self.rank, self.n_envs = rank, n_envs
        self.log_path = log_path
        self.objective = objective
        self.cache_physics = bool(cache_physics)
        self.dt = self.timestep * self.decision_interval
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (172,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        self.stream = None
        self.base_seed = None
        self.episode_index = 0
        self.used = set()
        self.restored = False
        self.near = 3.0

    def _build_sim(self, terrain_name, seed):
        if self._sim is not None:
            self._sim.close()
        old = physical._make_arena
        physical._make_arena = lambda name, sd: SingleObstacleArena(
            self.terrain_name, self.value, self.near
        )
        try:
            super()._build_sim(terrain_name, seed)
        finally:
            physical._make_arena = old
        if self.cache_physics:
            from .physics_cache import install

            self.forward_cache_stats = install(self._sim.physics)

    def reset(self, *, seed=None, options=None):
        self._interval_force_samples = 0
        if self.fixed_cell is not None:
            if seed is None:
                raise ValueError("Explicit evaluation phase seed required")
            episode_seed = int(seed)
            terrain, value = self.fixed_cell
        else:
            if seed is not None and not self.restored:
                self.base_seed = int(seed)
                self.stream = np.random.default_rng(
                    np.random.SeedSequence([self.base_seed, self.rank, 20261002])
                )
                self.episode_index = 0
                self.used = set()
            self.restored = False
            if self.stream is None:
                raise ValueError("Training seed required on initial reset")
            while True:
                code = int(
                    self.stream.integers(
                        10_000_000 // self.n_envs, (2**31 - 1) // self.n_envs
                    )
                )
                episode_seed = code * self.n_envs + self.rank
                if episode_seed not in self.used:
                    break
            self.used.add(episode_seed)
            terrain, value = self.cells[self.rank % len(self.cells)]
        self.training_nominal_value = float(value)
        if self.fixed_cell is None and self.training_height_jitter:
            assert terrain == "blocks"
            rule = self.training_height_jitter
            jitter_rng = np.random.default_rng(
                np.random.SeedSequence([episode_seed, int(rule["salt"])])
            )
            value = float(value) + float(
                jitter_rng.uniform(-rule["half_width_mm"], rule["half_width_mm"])
            )
            assert value > 0.0
        self.terrain_name = terrain
        self.value = float(value)
        self.far = self.near + (self.value if terrain == "gapped" else 1.3)
        # Goal and shaping use the obstacle far edge for every foot.
        super().reset(seed=episode_seed)
        initial_phase = float(self._cpg.curr_phases[0])
        if self.fixed_cell is None and self.training_phase_cycle:
            # Separate local RNG: preserve episode-seed, action, and PPO streams.
            bins = int(self.training_phase_cycle["bins"])
            origin = int(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            self.base_seed - self.rank,
                            self.rank % len(self.cells),
                            int(self.training_phase_cycle["origin_salt"]),
                        ]
                    )
                ).integers(0, bins)
            )
            phase_bin = (
                origin
                + (bins // 2) * (self.rank // len(self.cells))
                + self.episode_index
            ) % bins
            within = (initial_phase % (2 * np.pi)) / (2 * np.pi)
            initial_phase = 2 * np.pi * (phase_bin + within) / bins
        if self.phase_mode == "tripod":
            self._cpg.curr_phases = initial_phase + np.pi * np.array([0, 1, 0, 1, 0, 1])
        elif self.phase_mode != "random":
            raise ValueError(self.phase_mode)
        self.start = self._prev_fly_pos.copy()
        self.start_min_foot_x = float(
            self._sim.fly.get_observation(self._sim)["end_effectors"][:, 0].min()
        )
        self.previous_action = np.zeros(6)
        self.previous_progress = 0.0
        initial_heading = self._sim.fly.get_observation(self._sim)["fly_orientation"]
        self.previous_heading_potential = -self.heading_scale * abs(
            float(np.arctan2(initial_heading[1], initial_heading[0]))
        )
        self.max_lateral = 0.0
        self.clear_duration = 0.0
        self.episode_return = 0.0
        self.elapsed = 0.0
        self.success = False
        self.episode_seed = episode_seed
        self.last_info = {}
        self.signed_indices = [
            next(
                i
                for i, name in enumerate(self._sim.fly.contact_sensor_placements)
                if name.split("/")[-1] == leg + segment
            )
            for leg in physical.LEGS
            for segment in ("Tibia", "Tarsus1", "Tarsus2")
        ]
        self.episode_index += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            episode_index=self.episode_index,
                            episode_seed=episode_seed,
                            terrain=terrain,
                            value_mm=self.value,
                            nominal_value_mm=self.training_nominal_value,
                            initial_cpg_phase_rad=float(
                                self._cpg.curr_phases[0] % (2 * np.pi)
                            ),
                            initial_phase_bin=int(
                                (self._cpg.curr_phases[0] % (2 * np.pi))
                                / (2 * np.pi)
                                * 8
                            ),
                        )
                    )
                    + "\n"
                )
        return self.observation(), dict(episode_seed=episode_seed)

    def observation(self):
        raw = self._sim.fly.get_observation(self._sim)
        if not all(np.isfinite(v).all() for v in raw.values()):
            return np.zeros(172, dtype=np.float32)
        pos = raw["fly"][0]
        base = self._extract_obs(raw)
        force = raw["contact_forces"][self.signed_indices] @ raw["fly_orientation"]
        if getattr(self, "_interval_force_samples", 0):
            force = self._interval_force_min[self.signed_indices]
            base[18:24] = np.array(
                [
                    self._interval_force_peak_norm[self._leg_sensors[leg]].max()
                    for leg in physical.LEGS
                ]
            )
        result = np.r_[
            base,
            raw["fly"][1] / 10.0,
            raw["fly"][3] / 50.0,
            raw["cardinal_vectors"][2],
            force / 10.0,
            self.previous_action,
            raw["joints"][0] / np.pi,
            raw["joints"][1] / 100.0,
            (float(raw["end_effectors"][:, 0].min()) - self.start_min_foot_x)
            / (self.far - self.start_min_foot_x),
            (pos[1] - self.start[1]) / 0.75,
            self.far - pos[0],
            max(0.0, 1 - self.elapsed / self.max_time),
            float(self.terrain_name == "gapped"),
            float(self.terrain_name == "blocks"),
            self.value,
            self.max_lateral / 1.5,
            self.clear_duration / 0.1,
        ].astype(np.float32)
        assert result.shape == (172,) and np.isfinite(result).all()
        return result

    def step(self, gaussian_action):
        raw_action = np.asarray(gaussian_action, dtype=np.float32)
        if raw_action.shape != (6,) or not np.isfinite(raw_action).all():
            raise ValueError("Invalid action")
        action = np.tanh(raw_action)
        before = self._physics_steps
        _, _, physical_term, physical_timeout, info = super().step(action)
        dt = (self._physics_steps - before) * self.timestep
        self.elapsed = self._physics_steps * self.timestep
        raw = self._sim.fly.get_observation(self._sim)
        finite = all(np.isfinite(v).all() for v in raw.values())
        numerical = bool(
            info.get("physics_error") or info.get("nan_detected") or not finite
        )
        pos = raw["fly"][0] if finite else self._prev_fly_pos
        up = float(raw["cardinal_vectors"][2, 2]) if finite else -1.0
        lateral = float(pos[1] - self.start[1])
        self.max_lateral = max(self.max_lateral, abs(lateral))
        clear = bool(
            finite and np.all(raw["end_effectors"][:, 0] >= self.far) and up >= 0.8
        )
        self.clear_duration = (self.clear_duration + dt) if clear else 0.0
        self.success = bool(
            not physical_term and not numerical and self.clear_duration >= 0.1 - 1e-8
        )
        deadline = bool(physical_timeout or self.elapsed >= self.max_time - 1e-8)
        terminated = bool(physical_term or numerical or self.success or deadline)
        progress = (
            float(
                np.clip(
                    (float(raw["end_effectors"][:, 0].min()) - self.start_min_foot_x)
                    / (self.far - self.start_min_foot_x),
                    -1.0,
                    1.0,
                )
            )
            if finite
            else self.previous_progress
        )
        # Potential is aligned to the last foot and zero at every terminal
        # state. Its discounted contribution telescopes to zero from reset,
        # so failed partial progress cannot substitute for task completion.
        gamma = float(np.exp(-0.01 / 5.0))
        potential = 4.0 * progress if not terminated else 0.0
        terms = dict(
            progress=gamma * potential - 4.0 * self.previous_progress,
            success=10.0 * self.success,
            failure=-10.0 * bool(terminated and not self.success),
            timeout=0.0,
            posture=-0.5 * dt * float(np.clip((1 - up) / 0.5, 0, 4)),
            path=-0.8 * dt * min((lateral / 0.75) ** 2, 4),
            correction=-0.05 * dt * float(np.mean(action**2)),
        )
        heading = raw["fly_orientation"] if finite else np.array([1.0, 0.0, 0.0])
        heading_potential = (
            -self.heading_scale * abs(float(np.arctan2(heading[1], heading[0])))
            if not terminated
            else 0.0
        )
        terms["heading_shape"] = (
            gamma * heading_potential - self.previous_heading_potential
        )
        self.previous_heading_potential = heading_potential
        reward = float(sum(terms.values()))
        self.previous_progress = progress
        self.previous_action = action.copy()
        self.episode_return += reward
        info.update(
            terrain=self.terrain_name,
            terrain_spec_id=f"{self.terrain_name}_{self.value:g}",
            episode_seed=self.episode_seed,
            training_band_id=f"{self.terrain_name}_{self.training_nominal_value:g}",
            training_nominal_value_mm=self.training_nominal_value,
            terrain_value_mm=self.value,
            success=self.success,
            success_ever=self.success,
            goal_terminated=self.success,
            physical_termination=bool(physical_term or numerical),
            functional_failure=bool(terminated and not self.success),
            elapsed_s=self.elapsed,
            task_deadline_reached=deadline,
            forward_velocity_mm_s=float(raw["fly"][1, 0]) if finite else 0.0,
            body_z_mm=float(pos[2]),
            reward_terms=terms,
            action_rms=float(np.sqrt(np.mean(action**2))),
            x_mm=float(pos[0]),
            max_lateral_mm=self.max_lateral,
            clear_duration_s=self.clear_duration,
            numerical_error=numerical,
            episode_return=self.episode_return,
            force_observation_mode="interval_minimum_and_peak",
            force_window_physics_samples=int(
                getattr(self, "_interval_force_samples", 0)
            ),
            shaping_potential=potential,
            shaping_gamma=gamma,
            foot_progress_fraction=progress,
        )
        self.last_info = info
        # This is a finite task with remaining time observed: deadline has zero
        # future task return, so it is a terminal state, not a bootstrap timeout.
        return self.observation(), reward, terminated, False, info

    def get_state(self):
        return dict(
            base_seed=self.base_seed,
            stream=copy.deepcopy(self.stream.bit_generator.state)
            if self.stream
            else None,
            episode_index=self.episode_index,
            used=sorted(self.used),
        )

    def set_state(self, state):
        self.base_seed = state["base_seed"]
        self.stream = np.random.default_rng()
        self.stream.bit_generator.state = state["stream"]
        self.episode_index = state["episode_index"]
        self.used = set(state["used"])
        self.restored = True

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None


def make_env(rank, n_envs, cells, run=None, **kwargs):
    return ObstacleTask(
        cells=cells,
        rank=rank,
        n_envs=n_envs,
        log_path=str(ROOT / "runs" / run / f"resets_{rank}.jsonl") if run else None,
        **kwargs,
    )


# Preserve the audited 10 ms task, including success and terminal events.
MicroObstacleTask = ObstacleTask


class ObstacleTask(MicroObstacleTask):
    def __init__(self, *args, capture_micro_trace=False, **kwargs):
        conf = json.loads((CONFIG / "protocol.json").read_text())
        self.heading_scale = float(
            conf.get("reward_intervention", {}).get("heading_scale", 0.0)
        )
        self.onset_rule = conf.get("onset_gate")
        self.training_phase_cycle = conf.get("training_phase_cycle")
        self.training_height_jitter = conf.get("training_height_jitter")
        self.hold_repeat = int(conf["action_hold"]["repeat"])
        self.micro_gamma = float(conf["action_hold"]["micro_gamma"])
        self.fixed_observation_scale = bool(conf["fixed_observation_scale"])
        self.capture_micro_trace = capture_micro_trace
        self.last_micro_trace = []
        super().__init__(*args, **kwargs)

    def observation(self):
        obs = super().observation()
        if self.fixed_observation_scale:
            # Fixed unit scales, selected from the SOP rather than fitted
            # to learned trajectories. Other channels already carry scales.
            obs[:18] /= 2.0
            obs[18:24] /= 10.0
            obs = np.clip(obs, -10.0, 10.0)
        return obs

    def step(self, action):
        x = float(self._sim.fly.get_observation(self._sim)["fly"][0, 0])
        action, applied_gate = map_action(action, x, self.onset_rule)
        reward = 0.0
        before = self._physics_steps
        self.last_micro_trace = []
        self.last_micro_infos = []
        for index in range(self.hold_repeat):
            obs, micro_reward, term, trunc, info = super().step(action)
            reward += self.micro_gamma**index * micro_reward
            self.last_micro_infos.append(dict(info, micro_reward=micro_reward))
            if self.capture_micro_trace:
                raw = self._sim.fly.get_observation(self._sim)
                fields = dict(
                    time_s=float(self._sim.curr_time),
                    qpos=self._sim.physics.data.qpos,
                    qvel=self._sim.physics.data.qvel,
                    position=raw["fly"][0],
                    feet=raw["end_effectors"],
                    up=raw["cardinal_vectors"][2],
                    action=np.tanh(action),
                    observation=obs,
                )
                self.last_micro_trace.append(
                    {k: np.asarray(v).copy() for k, v in fields.items()}
                )
            if term or trunc:
                break
        info = dict(
            info,
            action_hold_micro_steps=index + 1,
            action_hold_physics_steps=int(self._physics_steps - before),
            policy_control_dt_s=0.01 * self.hold_repeat,
            macro_reward=float(reward),
            correction_onset_gate=applied_gate,
        )
        self.last_info = info
        return obs, float(reward), term, trunc, info


from .history_observation import history_task

ObstacleTask = history_task(ObstacleTask)

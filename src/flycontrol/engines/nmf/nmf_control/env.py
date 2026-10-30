"""Terrain locomotion gym environment.

Action: 6D per-leg correction in [-1, 1], applied for ``decision_interval``
physics steps before the next decision.

Observation: 46D proprioception (see ``cpg.CPGController.proprio_obs``).
Note: the observation does NOT include terrain identity. The recurrent
policy must infer terrain from contact / orientation history.

Reward (per env step):

    r_step = forward_velocity_mm_per_s     # main signal, ~10 healthy
    r_term = -flip_penalty if flipped      # only on terminal step

Design rationale (post-mortem on the previous reward):

    Previous design had three near-zero terms (flip_penalty=10 ≈ 0.5%
    of episode reward, energy_penalty=1e-3 producing ~0.04%, plus the
    main displacement term). The flip term was so small that the policy
    learned reckless gaits that flipped frequently; the energy term was
    pure numerical noise. We now keep ONE main signal (forward velocity,
    projected on initial heading so sideways drift gives no credit)
    and ONE terminal penalty large enough to dominate (1000 by default,
    configurable). No alive bonus: that would create a "stand still"
    optimum where the policy does nothing rather than risk flipping.
    No energy penalty: it was noise, drop it.

Terrain randomization:

    On each reset, sample (terrain_type, layout_seed) uniformly from
    the cartesian product of ``terrains`` and ``terrain_seeds``. Both
    must be lists. The mujoco model for each (type, seed) is built
    lazily on first encounter and cached for the lifetime of the env;
    reset() on a cached sim is fast (~ms vs ~seconds for build).

    The layout pool is what separates "the policy memorized 4 fixed
    obstacle layouts" (single seed) from "the policy generalizes
    across terrain instances" (pool of seeds). Without the pool the
    generalization claim is bogus -- with terrain_seeds=[0] the entire
    training distribution is exactly 4 environment instances.

    For evaluation, use HELD-OUT seeds (e.g. terrain_seeds=[100, 101,
    ...]) to test true generalization.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from collections import deque

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    if os.environ.get("PYOPENGL_PLATFORM", "").lower() == "egl":
        os.environ.pop("PYOPENGL_PLATFORM")

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from dm_control.rl.control import PhysicsError
from gymnasium import spaces

from .command import CommandToCPGAdapter
from .cpg import (
    ACTION_DIM,
    LOCAL_ACTION_DIM,
    SEMANTIC_MOTOR_ACTION_DIM,
    VNC_SUBTYPE_OBS_DIM,
    LEGS,
    OBS_DIM,
    CPGController,
    make_arena,
    make_diagnostic_camera,
    make_fly,
    make_simulation,
)
from .tasks import (
    TASK_OBS_DIM,
    SinePathTarget,
    _wrap_pi,
    heading_target_observation,
    heading_target_reward,
    heading_speed_target_observation,
    heading_speed_target_reward,
    path_following_reward,
    speed_control_reward,
    stumble_recovery_reward,
    speed_target_observation,
    steering_target_observation,
    steering_reward,
)
from .terrain import TerrainSpec
from .ood_metrics import (
    TraversalSample,
    TraversalState,
    classify_traversal_outcome,
    traversal_criteria,
    update_traversal_state,
)


DEFAULT_TERRAINS = ("gapped", "blocks", "mixed")
DEFAULT_TERRAIN_SEEDS = (0, 1, 2, 3)


class TerrainLocomotionEnv(gym.Env):
    """CPG + RL correction on procedural terrains.

    Args:
      terrains: list of terrain type names. Each reset samples one.
        Valid: "flat", "gapped", "blocks", "mixed".
      terrain_seeds: list of layout seeds. Each reset samples one. The
        mujoco model is keyed by (terrain, seed) and cached per-env.
        Memory cost ≈ 50MB × len(terrains) × len(terrain_seeds) per env.
      timestep: mujoco physics dt in seconds.
      decision_interval: number of mujoco steps between policy decisions.
      max_time: episode time limit in sim seconds (truncates if reached).
      max_correction: action ∈ [-1,1] is scaled by this before mixing
        with the CPG base joint angles.
      flip_penalty: reward subtracted on terminal flip. Default 1000 is
        ~50% of a full healthy episode reward, large enough to dominate.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        terrains: Sequence[str] = DEFAULT_TERRAINS,
        terrain_seeds: Sequence[int] = DEFAULT_TERRAIN_SEEDS,
        timestep: float = 1e-4,
        decision_interval: int = 100,
        max_time: float = 1.5,
        max_correction: float = 10.0,
        flip_penalty: float = 1000.0,
        flip_duration_s: float = 0.0,
        reward_mode: str = "legacy_velocity",
        traversal_success_bonus: float = 0.0,
        terminate_on_traversal_success: bool = False,
        terminate_on_traversal_failure: bool = False,
        action_mode: str = "correction6",
        command_adapter: CommandToCPGAdapter | None = None,
        command_adapter_config: dict | None = None,
        task: str = "forward",
        observation_mode: str = "proprio",
        steering_targets: Sequence[float] = (-1.0, 1.0),
        steering_reward_mode: str = "sign",
        steering_rate_max_rad_s: float = 2.5,
        reorientation_targets_rad: Sequence[float] = (
            -np.pi / 2,
            -np.pi / 3,
            -np.pi / 6,
            np.pi / 6,
            np.pi / 3,
            np.pi / 2,
        ),
        speed_targets_mm_s: Sequence[float] = (5.0, 10.0, 15.0),
        speed_target_sampling: str = "choice",
        speed_target_range_mm_s: Sequence[float] | None = None,
        speed_target_scale_mm_s: float = 20.0,
        speed_filter_alpha: float = 0.25,
        speed_heading_rate_scale_rad_s: float = 5.0,
        speed_error_weight: float = 2.0,
        speed_heading_rate_weight: float = 0.25,
        speed_huber_delta_mm_s: float = 1.0,
        heading_speed_error_weight: float = 20.0,
        perturbation: dict | None = None,
        perturbation_specs: Sequence[Mapping[str, Any]] | None = None,
        terrain_specs: Sequence[TerrainSpec | Mapping[str, Any]] | None = None,
        terrain_params: Mapping[str, Any] | None = None,
        sampling_mode: str = "random",
        observation_noise_std: float | Sequence[float] = 0.0,
        observation_delay: int = 0,
        action_delay: int = 0,
        physics_seed: int | None = None,
        physics_integrator: str | None = None,
        official_hybrid_config: Mapping[str, Any] | None = None,
        body_mass_scale: float = 1.0,
        adhesion_force_scale: float = 1.0,
        adhesion_dropout_probability: float = 0.0,
        record_video: bool = False,
    ) -> None:
        super().__init__()
        if not terrains:
            raise ValueError("terrains must be non-empty")
        if not terrain_seeds:
            raise ValueError("terrain_seeds must be non-empty")
        if terrain_specs is not None and not terrain_specs:
            raise ValueError("terrain_specs must be non-empty")
        if action_mode not in {
            "correction6", "command2", "command3", "local12", "hybrid14",
            "motor18", "motor18_phase", "official_hybrid2",
        }:
            raise ValueError(
                "action_mode must be one of: correction6, command2, command3, "
                "local12, hybrid14, motor18, motor18_phase, official_hybrid2"
            )
        if task not in {
            "forward",
            "steering_left",
            "steering_right",
            "steering_target",
            "path_sine",
            "reorientation",
            "heading_hold",
            "heading_speed_hold",
            "speed_control",
            "stumble_recovery",
        }:
            raise ValueError(
                "task must be one of: forward, steering_left, steering_right, "
                "steering_target, path_sine, reorientation, heading_hold, "
                "heading_speed_hold, speed_control, stumble_recovery"
            )
        if observation_mode not in {"proprio", "proprio_task", "vnc_subtype"}:
            raise ValueError(
                "observation_mode must be one of: proprio, proprio_task, "
                "vnc_subtype"
            )
        if reward_mode not in {"legacy_velocity", "traversal_aligned", "traversal_success_potential_v1"}:
            raise ValueError(
                "unsupported locomotion reward_mode"
            )
        if traversal_success_bonus < 0 or not np.isfinite(traversal_success_bonus):
            raise ValueError("traversal_success_bonus must be finite and non-negative")

        self.terrains = tuple(str(x) for x in terrains)
        self.terrain_seeds = tuple(int(s) for s in terrain_seeds)
        if sampling_mode not in {"random", "balanced_cycle"}:
            raise ValueError("sampling_mode must be 'random' or 'balanced_cycle'")
        self.sampling_mode = str(sampling_mode)
        self._legacy_terrain_sampling = terrain_specs is None and terrain_params is None
        if terrain_specs is not None:
            self.terrain_specs = tuple(
                x if isinstance(x, TerrainSpec) else TerrainSpec.from_mapping(x)
                for x in terrain_specs
            )
        else:
            params = dict(terrain_params or {})
            specs = []
            for terrain in self.terrains:
                per_terrain = params
                if params and not any(
                    key in params
                    for key in (
                        "gap_width",
                        "layout_mode",
                        "block_height",
                        "friction",
                        "observation_noise_std",
                        "observation_delay",
                        "action_delay",
                        "body_mass_scale",
                        "adhesion_force_scale",
                        "adhesion_dropout_probability",
                    )
                ):
                    per_terrain = dict(params.get(terrain, {}))
                for layout_seed in self.terrain_seeds:
                    specs.append(
                        TerrainSpec(name=terrain, seed=layout_seed, **per_terrain)
                    )
            self.terrain_specs = tuple(specs)
        self.timestep = timestep
        self.decision_interval = decision_interval
        self.max_time = max_time
        self.max_correction = max_correction
        self.flip_penalty = float(flip_penalty)
        self.reward_mode = str(reward_mode)
        self.traversal_success_bonus = float(traversal_success_bonus)
        self.terminate_on_traversal_success = bool(terminate_on_traversal_success)
        self.terminate_on_traversal_failure = bool(terminate_on_traversal_failure)
        if self.reward_mode == "traversal_success_potential_v1":
            if (task != "forward" or not self.terminate_on_traversal_success
                    or not self.terminate_on_traversal_failure or self.traversal_success_bonus != 0.
                    or not np.isfinite(max_time) or max_time <= 0.
                    or (terrain_specs is not None and
                        any(spec.layout_mode != "course_v1" for spec in self.terrain_specs))):
                raise ValueError("finite traversal objective requires forward course_v1, positive deadline, both traversal termination flags, and no extra success bonus")
        if not np.isfinite(flip_duration_s) or flip_duration_s < 0:
            raise ValueError("flip_duration_s must be finite and nonnegative")
        self.flip_duration_s = float(flip_duration_s)
        self.action_mode = action_mode
        if command_adapter is not None and command_adapter_config is not None:
            raise ValueError(
                "pass either command_adapter or command_adapter_config, not both"
            )
        self.command_adapter = command_adapter or CommandToCPGAdapter(
            **(command_adapter_config or {})
        )
        self.task = task
        self.observation_mode = observation_mode
        self.path_target = SinePathTarget()
        self.steering_targets = tuple(float(x) for x in steering_targets)
        if not self.steering_targets:
            raise ValueError("steering_targets must be non-empty")
        if any(float(np.sign(x)) == 0.0 for x in self.steering_targets):
            raise ValueError("steering_targets must not contain zero")
        if steering_reward_mode not in {"sign", "rate"}:
            raise ValueError("steering_reward_mode must be 'sign' or 'rate'")
        self.steering_reward_mode = steering_reward_mode
        self.steering_rate_max_rad_s = float(steering_rate_max_rad_s)
        self.reorientation_targets_rad = tuple(
            float(x) for x in reorientation_targets_rad
        )
        if not self.reorientation_targets_rad:
            raise ValueError("reorientation_targets_rad must be non-empty")
        if any(abs(float(x)) < 1e-9 for x in self.reorientation_targets_rad):
            raise ValueError("reorientation_targets_rad must not contain zero")
        self.speed_targets_mm_s = tuple(float(x) for x in speed_targets_mm_s)
        if not self.speed_targets_mm_s:
            raise ValueError("speed_targets_mm_s must be non-empty")
        if any(float(x) <= 0.0 for x in self.speed_targets_mm_s):
            raise ValueError("speed_targets_mm_s must be positive")
        if speed_target_sampling not in {"choice", "uniform"}:
            raise ValueError("speed_target_sampling must be 'choice' or 'uniform'")
        self.speed_target_sampling = speed_target_sampling
        target_range = (
            tuple(float(x) for x in speed_target_range_mm_s)
            if speed_target_range_mm_s is not None
            else (min(self.speed_targets_mm_s), max(self.speed_targets_mm_s))
        )
        if len(target_range) != 2 or not 0.0 < target_range[0] < target_range[1]:
            raise ValueError("speed_target_range_mm_s must be [positive_low, high]")
        self.speed_target_range_mm_s = target_range
        self.speed_target_scale_mm_s = float(speed_target_scale_mm_s)
        if self.speed_target_scale_mm_s <= 0.0:
            raise ValueError("speed_target_scale_mm_s must be positive")
        self.speed_filter_alpha = float(speed_filter_alpha)
        if not 0.0 < self.speed_filter_alpha <= 1.0:
            raise ValueError("speed_filter_alpha must be in (0, 1]")
        self.speed_heading_rate_scale_rad_s = float(speed_heading_rate_scale_rad_s)
        if self.speed_heading_rate_scale_rad_s <= 0.0:
            raise ValueError("speed_heading_rate_scale_rad_s must be positive")
        self.speed_error_weight = float(speed_error_weight)
        if self.speed_error_weight <= 0.0:
            raise ValueError("speed_error_weight must be positive")
        self.speed_heading_rate_weight = float(speed_heading_rate_weight)
        if self.speed_heading_rate_weight < 0.0:
            raise ValueError("speed_heading_rate_weight must be non-negative")
        self.speed_huber_delta_mm_s = float(speed_huber_delta_mm_s)
        if self.speed_huber_delta_mm_s <= 0.0:
            raise ValueError("speed_huber_delta_mm_s must be positive")
        self.heading_speed_error_weight = float(heading_speed_error_weight)
        if self.heading_speed_error_weight < 0.0:
            raise ValueError("heading_speed_error_weight must be non-negative")
        if perturbation_specs is not None and perturbation is not None:
            raise ValueError("pass either perturbation or perturbation_specs, not both")
        if perturbation_specs is not None:
            if not perturbation_specs:
                raise ValueError("perturbation_specs must be non-empty")
            self.perturbation_specs = tuple(
                dict(spec) for spec in perturbation_specs
            )
        else:
            self.perturbation_specs = (
                dict(perturbation or {"type": "none"}),
            )
        self.perturbation = dict(self.perturbation_specs[0])
        self._balanced_condition_order = np.empty(0, dtype=np.int64)
        self._balanced_condition_cursor = 0
        self.observation_delay = int(observation_delay)
        self.action_delay = int(action_delay)
        if self.observation_delay < 0 or self.action_delay < 0:
            raise ValueError("observation_delay and action_delay must be non-negative")
        self.physics_seed = None if physics_seed is None else int(physics_seed)
        if physics_integrator not in {None, "Euler", "RK4", "implicit", "implicitfast"}:
            raise ValueError("unsupported physics_integrator")
        self.physics_integrator = physics_integrator
        hybrid_cfg = dict(official_hybrid_config or {})
        unknown_hybrid = set(hybrid_cfg) - {"command_center", "command_scale", "persistence_duration_s"}
        if unknown_hybrid or (hybrid_cfg and action_mode != "official_hybrid2"):
            raise ValueError("official_hybrid_config requires official_hybrid2 and known keys")
        self.official_hybrid_command_center = float(hybrid_cfg.get("command_center", 0.))
        self.official_hybrid_command_scale = float(hybrid_cfg.get("command_scale", 1.))
        self.official_hybrid_persistence_duration_s = hybrid_cfg.get("persistence_duration_s")
        if (not np.isfinite(self.official_hybrid_command_center)
                or not np.isfinite(self.official_hybrid_command_scale)
                or self.official_hybrid_command_scale <= 0):
            raise ValueError("hybrid command center/scale must be finite with positive scale")
        if self.official_hybrid_persistence_duration_s is not None:
            duration = float(self.official_hybrid_persistence_duration_s)
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("hybrid persistence duration must be finite and positive")
            self.official_hybrid_persistence_duration_s = duration
        self.body_mass_scale = float(body_mass_scale)
        self.adhesion_force_scale = float(adhesion_force_scale)
        self.adhesion_dropout_probability = float(adhesion_dropout_probability)
        self.record_video = bool(record_video)
        if self.body_mass_scale <= 0.0:
            raise ValueError("body_mass_scale must be positive")
        if self.adhesion_force_scale < 0.0:
            raise ValueError("adhesion_force_scale must be non-negative")
        if not 0.0 <= self.adhesion_dropout_probability <= 1.0:
            raise ValueError("adhesion_dropout_probability must be in [0, 1]")
        self._target_turn_command = 1.0
        self._target_angle_rad = 0.0
        self._target_heading_rad = 0.0
        self._target_speed_mm_s = self.speed_targets_mm_s[0]
        self._filtered_forward_velocity = 0.0
        self._last_heading_rate = 0.0
        self._perturbation_side = 1.0
        self._perturbation_leg = LEGS[0]
        self._cpg_phase_impulse_applied = False
        self._fly = None

        obs_dim = {
            "proprio": OBS_DIM,
            "proprio_task": OBS_DIM + TASK_OBS_DIM,
            "vnc_subtype": VNC_SUBTYPE_OBS_DIM,
        }[observation_mode]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        noise_std = np.asarray(observation_noise_std, dtype=np.float32)
        if noise_std.ndim == 0:
            noise_std = np.full(obs_dim, float(noise_std), dtype=np.float32)
        if noise_std.shape != (obs_dim,) or np.any(noise_std < 0):
            raise ValueError(
                f"observation_noise_std must be scalar or shape ({obs_dim},)"
            )
        self.observation_noise_std = noise_std
        self._default_observation_noise_std = noise_std.copy()
        self._default_observation_delay = self.observation_delay
        self._default_action_delay = self.action_delay
        self._default_body_mass_scale = self.body_mass_scale
        self._default_adhesion_force_scale = self.adhesion_force_scale
        self._default_adhesion_dropout_probability = self.adhesion_dropout_probability
        action_dim = {
            "correction6": ACTION_DIM,
            "command2": 2,
            "command3": 3,
            "local12": LOCAL_ACTION_DIM,
            "hybrid14": 2 + LOCAL_ACTION_DIM,
            "motor18": SEMANTIC_MOTOR_ACTION_DIM,
            "motor18_phase": SEMANTIC_MOTOR_ACTION_DIM,
            "official_hybrid2": 2,
        }[self.action_mode]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        self._sim = None
        self._controller: CPGController | None = None
        self._physics_steps = 0
        self._inverted_steps = 0
        self._inverted_steps_total = 0
        self._max_inverted_steps = 0
        self._completed_inverted_run_lengths: list[int] = []
        self._prev_xy = np.zeros(2, dtype=np.float64)
        self._heading = np.array([1.0, 0.0])
        self._initial_heading_angle = 0.0
        self._prev_heading_angle = 0.0
        self._current_terrain: str | None = None
        self._current_layout_seed: int | None = None
        self._current_terrain_spec: TerrainSpec | None = None
        self._last_raw_obs = None
        self._max_progress_x = float("-inf")
        self._traversal_state = TraversalState()
        self._traversal_criteria = None
        self._randomization_rng = np.random.default_rng()
        self._obs_buffer: deque[np.ndarray] = deque()
        self._action_buffer: deque[np.ndarray] = deque()
        # Cache includes the complete normalized spec, not just name + seed.
        self._sim_cache: dict[tuple[Any, ...], tuple] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_build(self, spec: TerrainSpec) -> tuple:
        body_mass_scale = (
            self._default_body_mass_scale
            if spec.body_mass_scale is None
            else spec.body_mass_scale
        )
        adhesion_force_scale = (
            self._default_adhesion_force_scale
            if spec.adhesion_force_scale is None
            else spec.adhesion_force_scale
        )
        key = (
            *spec.cache_key,
            body_mass_scale,
            adhesion_force_scale,
            self.record_video,
        )
        cached = self._sim_cache.get(key)
        if cached is not None:
            return cached

        t0 = time.time()
        arena = make_arena(spec)
        fly = make_fly(
            draw_adhesion=self.record_video,
            body_mass_scale=body_mass_scale,
            adhesion_force_scale=adhesion_force_scale,
        )
        cameras = [make_diagnostic_camera(fly)] if self.record_video else []
        sim = make_simulation(arena, fly, self.timestep, cameras=cameras)
        if self.physics_integrator is not None:
            import mujoco
            modes = {"Euler": mujoco.mjtIntegrator.mjINT_EULER,
                     "RK4": mujoco.mjtIntegrator.mjINT_RK4,
                     "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                     "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
            sim.physics.model.opt.integrator = modes[self.physics_integrator]
        ctrl = CPGController.build(fly, self.timestep, spec.seed, self.max_correction)
        ctrl.official_persistence_duration_s = self.official_hybrid_persistence_duration_s
        self._sim_cache[key] = (sim, fly, ctrl)
        # One-line print on cache miss; useful during warmup, silent
        # afterwards. No flag to disable -- if it's noisy you have too
        # many (terrain, seed) combos and should reduce the pool.
        print(
            f"  [env cache] built {spec.terrain_spec_id} "
            f"in {time.time() - t0:.1f}s "
            f"(cache size={len(self._sim_cache)})",
            file=sys.stderr,
            flush=True,
        )
        return self._sim_cache[key]

    def _activate_spec(self, spec: TerrainSpec) -> None:
        """Resolve per-spec perturbations over constructor-level defaults."""
        noise = (
            self._default_observation_noise_std
            if spec.observation_noise_std is None
            else np.asarray(spec.observation_noise_std, dtype=np.float32)
        )
        if noise.ndim == 0:
            noise = np.full(
                self.observation_space.shape, float(noise), dtype=np.float32
            )
        if noise.shape != self.observation_space.shape:
            raise ValueError(
                "TerrainSpec observation_noise_std must be scalar or match "
                f"{self.observation_space.shape}"
            )
        self.observation_noise_std = noise
        self.observation_delay = (
            self._default_observation_delay
            if spec.observation_delay is None
            else spec.observation_delay
        )
        self.action_delay = (
            self._default_action_delay
            if spec.action_delay is None
            else spec.action_delay
        )
        self.body_mass_scale = (
            self._default_body_mass_scale
            if spec.body_mass_scale is None
            else spec.body_mass_scale
        )
        self.adhesion_force_scale = (
            self._default_adhesion_force_scale
            if spec.adhesion_force_scale is None
            else spec.adhesion_force_scale
        )
        self.adhesion_dropout_probability = (
            self._default_adhesion_dropout_probability
            if spec.adhesion_dropout_probability is None
            else spec.adhesion_dropout_probability
        )

    def _sample_terrain(self, fixed_seed: int | None) -> TerrainSpec:
        """Pick (terrain, layout_seed) for the next episode.

        If ``fixed_seed`` is not None (initial reset with explicit seed),
        the np_random has just been seeded with it, so the sampled
        terrain+layout is deterministic given the seed. Otherwise we
        draw from the configured pool with the env's evolving RNG.
        """
        if self._legacy_terrain_sampling:
            terrain = str(self.np_random.choice(self.terrains))
            layout_seed = int(self.np_random.choice(self.terrain_seeds))
            return TerrainSpec(name=terrain, seed=layout_seed)
        index = int(self.np_random.integers(len(self.terrain_specs)))
        return self.terrain_specs[index]

    def _next_balanced_condition(self) -> tuple[TerrainSpec, dict[str, Any]]:
        """Draw one terrain×perturbation cell from a shuffled full cycle."""
        n_perturbations = len(self.perturbation_specs)
        n_conditions = len(self.terrain_specs) * n_perturbations
        if (
            self._balanced_condition_order.size != n_conditions
            or self._balanced_condition_cursor >= n_conditions
        ):
            self._balanced_condition_order = self.np_random.permutation(
                n_conditions
            ).astype(np.int64, copy=False)
            self._balanced_condition_cursor = 0
        joint_index = int(
            self._balanced_condition_order[self._balanced_condition_cursor]
        )
        self._balanced_condition_cursor += 1
        terrain_index, perturbation_index = divmod(
            joint_index, n_perturbations
        )
        return (
            self.terrain_specs[terrain_index],
            dict(self.perturbation_specs[perturbation_index]),
        )

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def _tag(self, info: dict, *, flip: bool) -> dict:
        """Stamp every step's info with terrain identity + flip flag.

        Why every step (not just on done): the vec-env wrappers auto-reset
        the env when an episode ends, so by the time PPO inspects `info`
        on a done index, the env's *current* terrain has already changed.
        Tagging on each step means PPO can read `info[i]["terrain"]` at
        the exact step where `done[i] = True` and get the terrain of the
        episode that just ended. The cost is one dict-copy per step.
        """
        out = dict(info) if info else {}
        out["terrain"] = self._current_terrain
        out["layout_seed"] = self._current_layout_seed
        if self._current_terrain_spec is not None:
            effective_spec = self._current_terrain_spec.to_dict()
            effective_spec.update(
                {
                    "observation_noise_std": self.observation_noise_std.tolist(),
                    "observation_delay": self.observation_delay,
                    "action_delay": self.action_delay,
                    "body_mass_scale": self.body_mass_scale,
                    "adhesion_force_scale": self.adhesion_force_scale,
                    "adhesion_dropout_probability": (self.adhesion_dropout_probability),
                }
            )
            out["terrain_spec"] = effective_spec
            out["terrain_spec_id"] = self._current_terrain_spec.terrain_spec_id
        out["flip"] = bool(flip)
        out["task"] = self.task
        # The shared leg draw is used by single-leg perturbations. A side push
        # always targets the thorax; reporting that draw as its location is false.
        # Keep the reset-time draw unchanged to preserve seeded trajectories.
        is_side_push = self.perturbation.get("type") == "side_push"
        out["perturbation_leg"] = "" if is_side_push else self._perturbation_leg
        out["perturbation_body"] = "thorax" if is_side_push else ""
        out["perturbation_side"] = float(self._perturbation_side) if is_side_push else ""
        out["perturbation_force_frame"] = "world" if is_side_push else ""
        if self._last_raw_obs is not None:
            fly_pos = np.asarray(self._last_raw_obs["fly"][0, :3], dtype=float)
            out["progress_x_mm"] = float(fly_pos[0])
            out["progress_y_mm"] = float(fly_pos[1])
            out["body_z_mm"] = float(fly_pos[2])
            out["max_progress_x_mm"] = float(self._max_progress_x)
        out["elapsed_s"] = self._physics_steps * self.timestep
        # Early numerical returns bypass the normal step-info assembly.
        # Keep the already observed inversion statistics at every boundary.
        out["inverted_duration_s"] = self._inverted_steps_total * self.timestep
        out["max_contiguous_inverted_duration_s"] = self._max_inverted_steps * self.timestep
        return out

    def _task_obs(self, raw_obs) -> tuple[np.ndarray, dict]:
        if self.task == "path_sine":
            return self.path_target.observe(raw_obs)
        if self.task == "steering_target":
            return steering_target_observation(self._target_turn_command)
        if self.task in {"reorientation", "heading_hold"}:
            current_heading = self._raw_heading_angle(raw_obs)
            return heading_target_observation(
                current_heading,
                self._target_heading_rad,
                self._target_angle_rad,
            )
        if self.task == "heading_speed_hold":
            current_heading = self._raw_heading_angle(raw_obs)
            return heading_speed_target_observation(
                current_heading,
                self._target_heading_rad,
                self._target_speed_mm_s,
                self._filtered_forward_velocity,
                self.speed_target_scale_mm_s,
            )
        if self.task == "speed_control":
            return speed_target_observation(
                self._target_speed_mm_s,
                self.speed_target_scale_mm_s,
                current_speed_mm_s=self._filtered_forward_velocity,
                heading_rate_rad_s=self._last_heading_rate,
                heading_rate_scale_rad_s=self.speed_heading_rate_scale_rad_s,
            )
        return np.zeros(TASK_OBS_DIM, dtype=np.float32), {}

    @staticmethod
    def _raw_heading_angle(raw_obs) -> float:
        heading = np.asarray(raw_obs["fly_orientation"][:2], dtype=np.float64)
        return float(np.arctan2(heading[1], heading[0]))

    def _perturbation_active(self, elapsed_s: float) -> bool:
        cfg = self.perturbation
        if cfg.get("type", "none") == "none":
            return False
        start = float(cfg.get("start_time_s", 0.5))
        duration = float(cfg.get("duration_s", 0.2))
        return start <= elapsed_s < start + duration

    def _perturbation_info(self, elapsed_s: float) -> dict:
        active = self._perturbation_active(elapsed_s)
        cfg = self.perturbation
        return {
            "perturbation_type": str(cfg.get("type", "none")),
            "perturbation_active": active,
            "perturbation_side": self._perturbation_side if active else 0.0,
        }

    def _apply_action_perturbation(
        self,
        low: dict[str, np.ndarray],
        elapsed_s: float,
    ) -> tuple[dict[str, np.ndarray], dict]:
        info = self._perturbation_info(elapsed_s)
        if not info["perturbation_active"]:
            return low, info
        if info["perturbation_type"] == "single_leg_joint_bias":
            assert self._controller is not None
            return self._controller.apply_single_leg_joint_bias(
                low,
                self._perturbation_leg,
                float(self.perturbation.get("magnitude", 0.8)),
            ), info
        if info["perturbation_type"] not in {
            "adhesion_drop", "single_leg_adhesion_drop"
        }:
            return low, info

        out = dict(low)
        adhesion = np.asarray(low["adhesion"], dtype=np.int32).copy()
        cfg = self.perturbation
        legs = tuple(str(leg) for leg in cfg.get("legs", ()))
        if info["perturbation_type"] == "single_leg_adhesion_drop":
            legs = (self._perturbation_leg,)
        if not legs:
            legs = (
                ("LF", "LM", "LH")
                if self._perturbation_side < 0
                else ("RF", "RM", "RH")
            )
        leg_to_idx = {leg: i for i, leg in enumerate(LEGS)}
        for leg in legs:
            adhesion[leg_to_idx[leg]] = 0
        out["adhesion"] = adhesion
        return out, info

    def _cpg_perturbation_indices(self) -> np.ndarray:
        scope = str(self.perturbation.get("scope", "single_leg"))
        if scope == "all":
            return np.arange(len(LEGS), dtype=np.int64)
        if scope != "single_leg":
            raise ValueError("CPG perturbation scope must be 'single_leg' or 'all'")
        return np.asarray([LEGS.index(self._perturbation_leg)], dtype=np.int64)

    def _step_cpg_with_perturbation(self, elapsed_s: float) -> dict:
        """Advance the oscillator while applying a reversible neural insult.

        Frequency and coupling perturbations are restored immediately after
        the integration step, so they cannot compound across physics steps or
        overwrite command modulation.  A phase impulse is applied exactly once
        per episode and then recovery is entirely closed-loop.
        """
        assert self._controller is not None
        ctrl = self._controller
        cpg = ctrl.cpg
        info = self._perturbation_info(elapsed_s)
        info["cpg_perturbation_applied"] = False
        if not info["perturbation_active"]:
            ctrl.step_cpg()
            return info

        kind = info["perturbation_type"]
        if kind == "cpg_phase_impulse":
            indices = self._cpg_perturbation_indices()
            if not self._cpg_phase_impulse_applied:
                magnitude = float(
                    self.perturbation.get(
                        "magnitude_rad", self.perturbation.get("magnitude", np.pi)
                    )
                )
                cpg.curr_phases[indices] = np.mod(
                    cpg.curr_phases[indices] + magnitude, 2 * np.pi
                )
                self._cpg_phase_impulse_applied = True
                info["cpg_perturbation_applied"] = True
            ctrl.step_cpg()
            return info

        if kind == "cpg_frequency_scale":
            indices = self._cpg_perturbation_indices()
            scale = float(self.perturbation.get("scale", 0.0))
            if scale < 0.0:
                raise ValueError("cpg_frequency_scale scale must be non-negative")
            original = cpg.intrinsic_freqs.copy()
            cpg.intrinsic_freqs[indices] *= scale
            info["cpg_perturbation_applied"] = True
            try:
                ctrl.step_cpg()
            finally:
                cpg.intrinsic_freqs[:] = original
            return info

        if kind == "cpg_coupling_scale":
            indices = self._cpg_perturbation_indices()
            scale = float(self.perturbation.get("scale", 0.0))
            if scale < 0.0:
                raise ValueError("cpg_coupling_scale scale must be non-negative")
            original = cpg.coupling_weights.copy()
            cpg.coupling_weights[indices, :] *= scale
            cpg.coupling_weights[:, indices] *= scale
            info["cpg_perturbation_applied"] = True
            try:
                ctrl.step_cpg()
            finally:
                cpg.coupling_weights[:] = original
            return info

        ctrl.step_cpg()
        return info

    def _apply_adhesion_dropout(
        self, low: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.adhesion_dropout_probability == 0.0:
            return low
        out = dict(low)
        adhesion = np.asarray(low["adhesion"], dtype=np.int32).copy()
        dropped = self._randomization_rng.random(adhesion.shape) < (
            self.adhesion_dropout_probability
        )
        adhesion[dropped] = 0
        out["adhesion"] = adhesion
        return out

    def _clear_external_forces(self) -> None:
        if self._sim is not None:
            self._sim.physics.data.xfrc_applied[:] = 0.0

    def _apply_external_perturbation(self, elapsed_s: float) -> None:
        self._clear_external_forces()
        if not self._perturbation_active(elapsed_s):
            return
        if self.perturbation.get("type") != "side_push":
            return
        if self._sim is None or self._fly is None:
            return

        force = float(self.perturbation.get("force", 2500.0))
        direction = str(self.perturbation.get("direction", "lateral"))
        if direction != "lateral":
            raise ValueError("side_push direction must be 'lateral'")
        body_force = np.array(
            [0.0, self._perturbation_side * force, 0.0, 0.0, 0.0, 0.0]
        )
        self._sim.physics.bind(self._fly.thorax).xfrc_applied = body_force

    def _obs(self, raw_obs) -> np.ndarray:
        ctrl = self._controller
        assert ctrl is not None
        if self.observation_mode == "vnc_subtype":
            return ctrl.vnc_subtype_obs(raw_obs)
        proprio = ctrl.proprio_obs(raw_obs)
        if self.observation_mode == "proprio":
            return proprio
        task_obs, _ = self._task_obs(raw_obs)
        return np.concatenate([proprio, task_obs]).astype(np.float32)

    def _noisy_obs(self, raw_obs) -> np.ndarray:
        """Apply sensor noise after the canonical normalized observation."""
        obs = self._obs(raw_obs)
        if np.any(self.observation_noise_std):
            noise = self._randomization_rng.normal(
                0.0, self.observation_noise_std, size=obs.shape
            )
            obs = (obs + noise).astype(np.float32)
        return obs

    def _delayed_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.observation_delay == 0:
            return obs
        self._obs_buffer.append(obs.copy())
        return self._obs_buffer.popleft()

    def _delayed_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_delay == 0:
            return action
        self._action_buffer.append(action.copy())
        return self._action_buffer.popleft()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        if seed is not None and self.sampling_mode == "balanced_cycle":
            # An explicit reseed starts a reproducible fresh cycle. Normal
            # vector-env auto-resets pass seed=None and continue the cycle.
            self._balanced_condition_order = np.empty(0, dtype=np.int64)
            self._balanced_condition_cursor = 0

        if "terrain_spec" in options:
            supplied = options["terrain_spec"]
            spec = (
                supplied
                if isinstance(supplied, TerrainSpec)
                else TerrainSpec.from_mapping(supplied)
            )
            sampled_perturbation = None
        elif (
            self.sampling_mode == "balanced_cycle"
            and "perturbation" not in options
        ):
            spec, sampled_perturbation = self._next_balanced_condition()
        else:
            spec = self._sample_terrain(seed)
            sampled_perturbation = None
        if self.reward_mode == "traversal_success_potential_v1" and spec.layout_mode != "course_v1":
            raise ValueError("finite traversal objective requires course_v1 reset geometry")
        self._activate_spec(spec)
        sim, _fly, ctrl = self._get_or_build(spec)
        self._sim = sim
        self._fly = _fly
        self._controller = ctrl
        self._current_terrain = spec.name
        self._current_layout_seed = spec.seed
        self._current_terrain_spec = spec
        if self.task == "steering_target":
            if "target_turn_sign" in options:
                self._target_turn_command = float(
                    np.clip(options["target_turn_sign"], -1.0, 1.0)
                )
                if float(np.sign(self._target_turn_command)) == 0.0:
                    raise ValueError("target_turn_sign must be non-zero")
            else:
                self._target_turn_command = float(
                    self.np_random.choice(self.steering_targets)
                )
        elif self.task == "reorientation":
            if "target_angle_rad" in options:
                self._target_angle_rad = float(options["target_angle_rad"])
                if abs(self._target_angle_rad) < 1e-9:
                    raise ValueError("target_angle_rad must be non-zero")
            else:
                self._target_angle_rad = float(
                    self.np_random.choice(self.reorientation_targets_rad)
                )
        elif self.task in {"heading_hold", "heading_speed_hold"}:
            self._target_angle_rad = 0.0
        if self.task in {"speed_control", "heading_speed_hold"}:
            if "target_speed_mm_s" in options:
                self._target_speed_mm_s = float(options["target_speed_mm_s"])
                if self._target_speed_mm_s <= 0.0:
                    raise ValueError("target_speed_mm_s must be positive")
            else:
                if self.speed_target_sampling == "uniform":
                    self._target_speed_mm_s = float(self.np_random.uniform(
                        *self.speed_target_range_mm_s
                    ))
                else:
                    self._target_speed_mm_s = float(
                        self.np_random.choice(self.speed_targets_mm_s)
                    )
        self._perturbation_side = float(self.np_random.choice([-1.0, 1.0]))
        if "perturbation" in options:
            self.perturbation = dict(options["perturbation"])
        elif sampled_perturbation is not None:
            self.perturbation = sampled_perturbation
        else:
            self.perturbation = dict(
                self.perturbation_specs[
                    int(self.np_random.integers(len(self.perturbation_specs)))
                ]
            )
        configured_legs = tuple(str(x) for x in self.perturbation.get("legs", LEGS))
        if not configured_legs or any(x not in LEGS for x in configured_legs):
            raise ValueError("perturbation legs must be non-empty members of LEGS")
        self._perturbation_leg = str(self.np_random.choice(configured_legs))
        self._cpg_phase_impulse_applied = False

        reset_physics_seed = options.get("physics_seed", self.physics_seed)
        if reset_physics_seed is None:
            reset_physics_seed = spec.seed
        reset_physics_seed = int(reset_physics_seed)
        rng_seed = np.random.SeedSequence([reset_physics_seed, 0x4F4F44])
        self._randomization_rng = np.random.default_rng(rng_seed)
        raw, _ = self._sim.reset(seed=reset_physics_seed)
        self._controller.reset(reset_physics_seed)
        if self.action_mode in {"command2", "command3", "hybrid14"}:
            neutral = np.zeros(self.action_space.shape, dtype=np.float32)
            command = neutral[:2] if self.action_mode == "hybrid14" else neutral
            self._controller.apply_command_modulation(command, self.command_adapter)
        self._last_raw_obs = raw

        self._physics_steps = 0
        self._objective_potential = 0.
        self._inverted_steps = 0
        self._inverted_steps_total = 0
        self._max_inverted_steps = 0
        self._completed_inverted_run_lengths = []
        self._filtered_forward_velocity = 0.0
        self._last_heading_rate = 0.0
        self._max_progress_x = float(raw["fly"][0, 0])
        self._traversal_criteria = traversal_criteria(spec)
        initial_sample = TraversalSample(
            x=float(raw["fly"][0, 0]),
            y=float(raw["fly"][0, 1]),
            body_z=float(raw["fly"][0, 2]),
            elapsed_s=0.0,
        )
        self._traversal_state = update_traversal_state(
            TraversalState(), initial_sample, self._traversal_criteria, dt=0.0
        )
        self._prev_xy = raw["fly"][0, :2].copy()
        heading = raw["fly_orientation"][:2].copy()
        norm = np.linalg.norm(heading)
        self._heading = heading / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self._prev_heading_angle = float(np.arctan2(self._heading[1], self._heading[0]))
        self._initial_heading_angle = self._prev_heading_angle
        if self.task in {"reorientation", "heading_hold", "heading_speed_hold"}:
            self._target_heading_rad = _wrap_pi(
                self._initial_heading_angle + self._target_angle_rad
            )

        initial_obs = self._noisy_obs(raw)
        self._obs_buffer.clear()
        self._obs_buffer.extend(
            initial_obs.copy() for _ in range(self.observation_delay)
        )
        self._action_buffer.clear()
        zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._action_buffer.extend(zero_action.copy() for _ in range(self.action_delay))
        self._numerical_failure_trace = None
        if spec.layout_mode == "course_v1":
            from .numerical_failure_trace import NumericalFailureTrace
            self._numerical_failure_trace = NumericalFailureTrace(
                {
                    "terrain_spec": spec.to_dict(), "physics_seed": reset_physics_seed,
                    "timestep_s": self.timestep, "decision_interval": self.decision_interval,
                    "action_mode": self.action_mode, "task": self.task,
                    "perturbation": dict(self.perturbation),
                    "action_delay": self.action_delay,
                    "exact_action_replay_scope_supported": (
                        self.task == "forward" and self.perturbation.get("type", "none") == "none"
                    ),
                },
                self._sim.physics.get_state(), raw["fly"],
                self.max_time, self.timestep * self.decision_interval,
            )
        _task_obs, task_info = self._task_obs(raw)
        return initial_obs, self._tag(task_info, flip=False)

    def step(self, action: np.ndarray):
        trace = getattr(self, "_numerical_failure_trace", None)
        if trace is not None and np.asarray(action).shape == self.action_space.shape:
            trace.before_step(action, self._physics_steps, self._last_raw_obs["fly"])
        obs, reward, terminated, truncated, info = self._step_with_velocity_reward(action)
        if trace is not None and (info.get("physics_error") or info.get("nan_detected")):
            info = {**info, "numerical_failure_replay": trace.payload(
                self._last_raw_obs["fly"], self._physics_steps)}
        if self.reward_mode != "traversal_success_potential_v1":
            return obs, reward, terminated, truncated, info
        from .terrain_objective import traversal_objective_transition

        terminal = bool(terminated or truncated)
        progress = 0.
        if not terminal:
            start_x = float(self._traversal_state.start_x)
            target_x = float(self._traversal_criteria.checkpoints_x[-1])
            progress = (float(info["progress_x_mm"]) - start_x) / max(1e-12, target_x - start_x)
        reward, components = traversal_objective_transition(
            self._objective_potential, progress, float(info["elapsed_s"]), self.max_time,
            success=bool(info.get("success", False)), terminal=terminal,
        )
        self._objective_potential = components["reward_potential_after"]
        info = {**info, **components, "task_deadline_reached": bool(truncated),
                "task_deadline_is_terminal": True}
        # The deadline defines task failure, not an artificial rollout cutoff.
        return obs, reward, terminal, False, info

    def _step_with_velocity_reward(self, action: np.ndarray):
        ctrl = self._controller
        assert ctrl is not None and self._sim is not None, "call reset first"

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action shape must be {self.action_space.shape}, got {action.shape}"
            )
        action = self._delayed_action(action)
        terminated = False
        truncated = False
        flipped = False
        physics_steps_before = self._physics_steps
        interval_disp = 0.0  # mm of forward motion in this env step
        heading_delta = 0.0
        raw = None
        info: dict = {}
        cpg_perturbation_applied_this_decision = False

        for _ in range(self.decision_interval):
            elapsed_s = self._physics_steps * self.timestep
            if self.action_mode == "correction6":
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_action(action)
            elif self.action_mode == "local12":
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_local_action(action)
            elif self.action_mode == "motor18":
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_semantic_motor_action(action)
            elif self.action_mode == "motor18_phase":
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_semantic_motor_action(
                    action, phase_gated=True
                )
            elif self.action_mode == "official_hybrid2":
                assert self._last_raw_obs is not None
                descending_drive = self.official_hybrid_command_center + self.official_hybrid_command_scale * action
                ctrl.prepare_official_hybrid_reflex(self._last_raw_obs, descending_drive)
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_official_hybrid_action()
            elif self.action_mode == "hybrid14":
                ctrl.apply_command_modulation(action[:2], self.command_adapter)
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_local_action(action[2:])
            else:
                ctrl.apply_command_modulation(action, self.command_adapter)
                perturb_info = self._step_cpg_with_perturbation(elapsed_s)
                low = ctrl.low_level_action(np.zeros(ACTION_DIM, dtype=np.float32))
            low, action_perturb_info = self._apply_action_perturbation(
                low, elapsed_s
            )
            perturb_info = {**action_perturb_info, **perturb_info}
            cpg_perturbation_applied_this_decision |= bool(
                perturb_info.get("cpg_perturbation_applied", False)
            )
            low = self._apply_adhesion_dropout(low)
            self._apply_external_perturbation(elapsed_s)
            try:
                raw, _, _, _, info = self._sim.step(low)
            except PhysicsError as exc:
                # Treat physics blowup like a flip: large negative
                # reward so the policy strongly avoids the regime that
                # produced it.
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return (
                    obs,
                    -self.flip_penalty,
                    True,
                    False,
                    self._tag({
                        "physics_error": True,
                        "physics_error_type": type(exc).__name__,
                        "physics_error_message": str(exc),
                        # Count only completed steps: the failed MuJoCo step
                        # does not provide a valid physical observation.
                        "physics_error_after_physics_steps": self._physics_steps,
                        "success": False,
                    }, flip=False),
                )

            self._physics_steps += 1
            self._last_raw_obs = raw
            self._max_progress_x = max(self._max_progress_x, float(raw["fly"][0, 0]))

            xy = raw["fly"][0, :2]
            disp = xy - self._prev_xy
            interval_disp += float(np.dot(disp, self._heading))
            self._prev_xy = xy.copy()
            heading = raw["fly_orientation"][:2]
            heading_angle = float(np.arctan2(heading[1], heading[0]))
            heading_delta += float(
                (heading_angle - self._prev_heading_angle + np.pi) % (2 * np.pi) - np.pi
            )
            self._prev_heading_angle = heading_angle

            if not np.isfinite(raw["fly"]).all():
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return (
                    obs,
                    -self.flip_penalty,
                    True,
                    False,
                    self._tag({"nan_detected": True}, flip=False),
                )
            inverted = bool(info.get("flip", False))
            if inverted:
                self._inverted_steps += 1
            else:
                if self._inverted_steps:
                    self._completed_inverted_run_lengths.append(
                        self._inverted_steps
                    )
                self._inverted_steps = 0
            self._inverted_steps_total += int(inverted)
            self._max_inverted_steps = max(self._max_inverted_steps, self._inverted_steps)
            info["instantaneous_inversion"] = inverted
            info["inverted_duration_s"] = self._inverted_steps_total * self.timestep
            info["max_contiguous_inverted_duration_s"] = self._max_inverted_steps * self.timestep
            if inverted and self._inverted_steps * self.timestep + 1e-12 >= self.flip_duration_s:
                terminated = True
                flipped = True
                break
            if (self._physics_steps * self.timestep) >= self.max_time:
                truncated = True
                break

        # Reward: forward velocity in mm/s. Per env step covers
        # decision_interval * timestep = 0.01 sim seconds; multiplying
        # the per-step displacement by 1/dt gives velocity. With the
        # default dt this is exactly interval_disp * 100.
        env_step_dt = self.decision_interval * self.timestep
        forward_velocity = float(interval_disp / env_step_dt)
        heading_rate = float(heading_delta / env_step_dt)
        self._filtered_forward_velocity = (
            (1.0 - self.speed_filter_alpha) * self._filtered_forward_velocity
            + self.speed_filter_alpha * forward_velocity
        )
        self._last_heading_rate = heading_rate
        reward = forward_velocity
        info = {
            **info,
            **perturb_info,
            "cpg_perturbation_applied": cpg_perturbation_applied_this_decision,
            "forward_velocity_mm_s": forward_velocity,
            "filtered_forward_velocity_mm_s": self._filtered_forward_velocity,
            "heading_delta_rad": heading_delta,
            "heading_rate_rad_s": heading_rate,
            "gait_coordinate": (
                "" if ctrl.gait_coordinate is None else ctrl.gait_coordinate
            ),
            "duty_factor_command": (
                "" if ctrl.duty_factor is None else ctrl.duty_factor
            ),
            "ipsilateral_lag_cycles_command": (
                "" if ctrl.ipsilateral_lag_cycles is None
                else ctrl.ipsilateral_lag_cycles
            ),
            "cpg_frequency_mean_hz": float(ctrl.cpg.intrinsic_freqs.mean()),
            "cpg_amplitude_mean": float(ctrl.cpg.intrinsic_amps.mean()),
        }
        traversal_success = False
        if raw is not None and self._traversal_criteria is not None:
            traversal_sample = TraversalSample(
                x=float(raw["fly"][0, 0]),
                y=float(raw["fly"][0, 1]),
                body_z=float(raw["fly"][0, 2]),
                elapsed_s=self._physics_steps * self.timestep,
                flipped=flipped,
                upright=not bool(info.get("instantaneous_inversion", False)),
            )
            self._traversal_state = update_traversal_state(
                self._traversal_state,
                traversal_sample,
                self._traversal_criteria,
                dt=(self._physics_steps - physics_steps_before) * self.timestep,
            )
            traversal_outcome = classify_traversal_outcome(
                self._traversal_state,
                traversal_sample,
                self._traversal_criteria,
                episode_done=bool(terminated or truncated),
            )
            traversal_success = bool(traversal_outcome.success)
            start_x = self._traversal_state.start_x
            target_x = self._traversal_criteria.checkpoints_x[-1]
            required = max(1e-12, target_x - float(start_x))
            info.update({
                "success": traversal_success,
                "success_ever": traversal_success,
                "training_traversal_success": traversal_success,
                "training_progress_fraction": float(np.clip(
                    (self._traversal_state.max_x - float(start_x)) / required,
                    0.0,
                    1.0,
                )),
                "training_checkpoints_passed": int(
                    self._traversal_state.next_checkpoint
                ),
                "training_traversal_failure": traversal_outcome.failure.value,
            })
            if (self.terminate_on_traversal_failure and
                    traversal_outcome.failure.value in {"fall_into_gap", "bypass"}):
                terminated = True
                reward -= self.flip_penalty
            if traversal_success and self.reward_mode in {"traversal_aligned", "traversal_success_potential_v1"}:
                reward += self.traversal_success_bonus
                if self.terminate_on_traversal_success:
                    terminated = True
        if self.task in {"steering_left", "steering_right", "steering_target"}:
            reward = steering_reward(
                self.task,
                forward_velocity,
                heading_rate,
                target_turn_sign=self._target_turn_command,
                mode=self.steering_reward_mode,
                steering_rate_max_rad_s=self.steering_rate_max_rad_s,
            )
            target_rate = self._target_turn_command * self.steering_rate_max_rad_s
            info["target_heading_rate_rad_s"] = target_rate
            info["steering_rate_error_abs"] = abs(heading_rate - target_rate)
        if flipped:
            reward -= self.flip_penalty

        if raw is not None:
            obs = self._delayed_obs(self._noisy_obs(raw))
            _task_obs, task_info = self._task_obs(raw)
            info = {**info, **task_info}
            if self.task == "path_sine":
                reward = path_following_reward(
                    forward_velocity,
                    task_info["path_error_mm"],
                    task_info["path_heading_error_rad"],
                )
                if flipped:
                    reward -= self.flip_penalty
            elif self.task in {"reorientation", "heading_hold"}:
                reward = heading_target_reward(
                    forward_velocity,
                    task_info["heading_error_rad"],
                )
                if flipped:
                    reward -= self.flip_penalty
            elif self.task == "heading_speed_hold":
                speed_error = abs(
                    self._filtered_forward_velocity - self._target_speed_mm_s
                )
                info["speed_error_abs_mm_s"] = speed_error
                reward = heading_speed_target_reward(
                    self._filtered_forward_velocity,
                    heading_rate,
                    task_info["heading_error_rad"],
                    self._target_speed_mm_s,
                    speed_error_weight=self.speed_error_weight,
                    heading_rate_weight=self.speed_heading_rate_weight,
                    heading_error_weight=self.heading_speed_error_weight,
                    huber_delta_mm_s=self.speed_huber_delta_mm_s,
                )
                if flipped:
                    reward -= self.flip_penalty
            elif self.task == "speed_control":
                speed_error = abs(
                    self._filtered_forward_velocity - self._target_speed_mm_s
                )
                info["speed_error_abs_mm_s"] = speed_error
                reward = speed_control_reward(
                    self._filtered_forward_velocity,
                    heading_rate,
                    self._target_speed_mm_s,
                    speed_error_weight=self.speed_error_weight,
                    heading_rate_weight=self.speed_heading_rate_weight,
                    huber_delta_mm_s=self.speed_huber_delta_mm_s,
                )
                if flipped:
                    reward -= self.flip_penalty
            elif self.task == "stumble_recovery":
                local_action = (
                    action[2:] if self.action_mode == "hybrid14" else action
                )
                reward = stumble_recovery_reward(
                    forward_velocity,
                    heading_rate,
                    bool(perturb_info["perturbation_active"]),
                    local_action_l2=float(np.linalg.norm(local_action)),
                )
                info["local_action_l2"] = float(np.linalg.norm(local_action))
                if flipped:
                    reward -= self.flip_penalty
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, self._tag(info, flip=flipped)

    @property
    def inversion_run_lengths_physics_steps(self) -> tuple[int, ...]:
        """Completed runs plus a currently open inversion run.

        This compact physics-resolution trace is sufficient to independently
        reproduce total and maximum contiguous inversion durations without
        storing one boolean for every MuJoCo step.
        """
        runs = list(self._completed_inverted_run_lengths)
        if self._inverted_steps:
            runs.append(self._inverted_steps)
        return tuple(runs)

    def render(self):
        if self._sim is not None:
            return self._sim.render()
        return None

    def save_video(self, path: str) -> None:
        if self._sim is None or not self._sim.cameras:
            raise RuntimeError("record_video=True is required to save diagnostics")
        self._sim.cameras[0].save_video(path)

    def close(self) -> None:
        for sim, _fly, _ctrl in self._sim_cache.values():
            close = getattr(sim, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        self._sim_cache.clear()
        self._sim = None
        self._fly = None
        self._controller = None
        self._last_raw_obs = None

    @property
    def last_raw_obs(self):
        """Most recent raw FlyGym observation, for evaluation diagnostics."""
        return self._last_raw_obs

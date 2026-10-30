"""Gymnasium Ant-v5 with episode-boundary dynamics transfers.

The Nature loss-of-plasticity experiment used Gym 0.23 ``SlipperyAnt-v3``
and rebuilt the MuJoCo XML whenever ground friction changed.  Installing
that stack would require Python 3.8, ``mujoco-py`` and NumPy 1.x, all of
which conflict with this repository.

This module keeps the scientific intervention while using the supported
Gymnasium/MuJoCo API: only the floor geom's sliding-friction coefficient is
changed, and a pending change is committed at the next episode reset.  No
Ant dataset is required; the Ant XML ships with Gymnasium.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import gymnasium as gym
import mujoco
import numpy as np


DEFAULT_SCHEDULE_PATH = Path(__file__).parent / "data" / "official_frictions.pkl"
OFFICIAL_SCHEDULE_SHA256 = (
    "2b275c261849289e060d40a90306aac5c7f3d5359fa68b8420a6dbe399cfc314"
)


@dataclass(frozen=True)
class StabilizationSpec:
    """Pre-registered V2-SR reward and termination parameters."""

    enabled: bool = False
    forward_cap_mps: float = 2.2
    inversion_threshold_degrees: float = 90.0
    inversion_persistence_steps: int = 10
    action_scale: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "StabilizationSpec":
        spec = cls(**dict(value or {}))
        if not math.isfinite(spec.forward_cap_mps) or spec.forward_cap_mps <= 0:
            raise ValueError("forward_cap_mps must be finite and positive")
        if not 0 < spec.inversion_threshold_degrees < 180:
            raise ValueError("inversion_threshold_degrees must be between 0 and 180")
        if spec.inversion_persistence_steps <= 0:
            raise ValueError("inversion_persistence_steps must be positive")
        if not math.isfinite(spec.action_scale) or not 0 < spec.action_scale <= 1:
            raise ValueError("action_scale must be in (0, 1]")
        return spec


def stabilized_forward_reward(
    velocity: float,
    upright_factor: float,
    cap_mps: float,
) -> float:
    """Cap/modulate positive progress without attenuating backward cost."""
    velocity = float(velocity)
    positive = max(velocity, 0.0)
    negative = min(velocity, 0.0)
    return negative + float(upright_factor) * min(positive, float(cap_mps))


def _torso_posture(model, data, torso_body_id: int) -> dict[str, float | bool]:
    torso_height = float(data.qpos[2])
    torso_up_z = float(
        np.clip(np.asarray(data.xmat[torso_body_id]).reshape(3, 3)[2, 2], -1.0, 1.0)
    )
    tilt_radians = math.acos(torso_up_z)
    finite_state = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    return {
        "torso_height": torso_height,
        "torso_up_z": torso_up_z,
        "torso_tilt_degrees": math.degrees(tilt_radians),
        "upright_factor": float(np.clip((1.0 + torso_up_z) / 2.0, 0.0, 1.0)),
        "finite_state": finite_state,
    }


def _wall_contact_diagnostics(
    model,
    data,
    wall_geom_ids: frozenset[int],
) -> dict[str, float | int]:
    """Measure Ant-to-maze contacts without changing MuJoCo state."""
    count = 0
    normal_force_sum = 0.0
    normal_force_max = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        wall_contact = (
            int(contact.geom1) in wall_geom_ids or int(contact.geom2) in wall_geom_ids
        )
        if not wall_contact:
            continue
        count += 1
        mujoco.mj_contactForce(model, data, index, force)
        normal_force = abs(float(force[0]))
        normal_force_sum += normal_force
        normal_force_max = max(normal_force_max, normal_force)
    return {
        "wall_contact_count": count,
        "wall_contact_normal_force": normal_force_sum,
        "wall_contact_max_normal_force": normal_force_max,
    }


def _append_goal_features(
    observation: np.ndarray,
    achieved_goal: np.ndarray,
    goal_delta: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation),
            np.asarray(achieved_goal),
            np.asarray(goal_delta),
        ]
    ).astype(np.float64, copy=False)


def _goal_augmented_space(space: gym.spaces.Box) -> gym.spaces.Box:
    dtype = np.dtype(space.dtype)
    return gym.spaces.Box(
        low=np.concatenate([space.low, np.full(4, -np.inf, dtype=dtype)]),
        high=np.concatenate([space.high, np.full(4, np.inf, dtype=dtype)]),
        dtype=dtype,
    )


def load_friction_schedule(
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Load one of the reference paper's 100 fixed friction streams.

    The downloaded file is a ``(100, 50)`` float64 array.  Values are
    log-uniform samples in ``[0.02, 2.0]``.  Loading a fixed stream instead
    of sampling inside the environment makes all policy variants see the
    same dynamics sequence.
    """
    schedule_path = Path(path)
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Friction schedule not found: {schedule_path}. "
            "Use the fixed friction_values in configs/figure5/experiments/."
        )
    if schedule_path.resolve() == DEFAULT_SCHEDULE_PATH.resolve():
        digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
        if digest != OFFICIAL_SCHEDULE_SHA256:
            raise RuntimeError(
                "Pinned friction schedule checksum mismatch: "
                f"expected {OFFICIAL_SCHEDULE_SHA256}, got {digest}"
            )
    with schedule_path.open("rb") as file:
        schedules = np.asarray(pickle.load(file), dtype=np.float64)
    if schedules.ndim != 2 or schedules.shape[0] < 1:
        raise ValueError(
            f"Expected a 2D friction schedule, got shape {schedules.shape}"
        )
    if not 0 <= int(seed) < schedules.shape[0]:
        raise ValueError(f"schedule seed {seed} outside [0, {schedules.shape[0] - 1}]")
    values = schedules[int(seed)].copy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Friction schedule must contain finite positive values")
    return values


class ContinualAntEnv(gym.Env):
    """Ant-v5 whose ground friction changes at episode boundaries.

    Parameters
    ----------
    friction_values:
        Explicit dynamics stream.  If omitted, ``schedule_path`` and
        ``schedule_seed`` select one fixed stream from the paper.
    steps_per_context:
        Number of transitions collected by *this environment* before a
        switch becomes pending.  With vectorized training, the training
        entry point divides a global change budget by the number of workers.
    cycle:
        Cycle after the final value.  The full reference run consumes only
        the first ten values, so the default is false and the last context is
        held if a longer run is requested.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        *,
        env_id: str = "Ant-v5",
        friction_values: Sequence[float] | None = None,
        schedule_path: str | Path = DEFAULT_SCHEDULE_PATH,
        schedule_seed: int = 0,
        steps_per_context: int = 2_000_000,
        cycle: bool = False,
        render_mode: str | None = None,
        stabilization: Mapping[str, object] | None = None,
        append_zero_goal_features: bool = False,
        **ant_kwargs,
    ) -> None:
        super().__init__()
        if env_id != "Ant-v5":
            raise ValueError(
                "This compatibility port is intentionally pinned to Ant-v5; "
                f"got {env_id!r}"
            )
        if int(steps_per_context) <= 0:
            raise ValueError("steps_per_context must be positive")

        values = (
            np.asarray(friction_values, dtype=np.float64)
            if friction_values is not None
            else load_friction_schedule(schedule_path, seed=schedule_seed)
        )
        if values.ndim != 1 or values.size == 0:
            raise ValueError("friction_values must be a non-empty 1D sequence")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("friction_values must be finite and positive")

        self.friction_values = values.copy()
        self.steps_per_context = int(steps_per_context)
        self.cycle = bool(cycle)
        self.context_index = 0
        self.context_steps = 0
        self.total_steps = 0
        self._switch_pending = False
        self.stabilization = StabilizationSpec.from_mapping(stabilization)
        self.append_zero_goal_features = bool(append_zero_goal_features)
        self._inverted_steps = 0

        # Ant-v5 uses the maintained native ``mujoco`` bindings.  Do not
        # install the old ``mujoco-py`` package beside this environment.
        self.base_env = gym.make(env_id, render_mode=render_mode, **ant_kwargs)
        self.observation_space = (
            _goal_augmented_space(self.base_env.observation_space)
            if self.append_zero_goal_features
            else self.base_env.observation_space
        )
        self.action_space = self.base_env.action_space
        self.render_mode = render_mode

        model = self.base_env.unwrapped.model
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            self.base_env.close()
            raise RuntimeError("Ant-v5 model has no geom named 'floor'")
        self._floor_geom_id = int(floor_id)
        torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if torso_body_id < 0:
            self.base_env.close()
            raise RuntimeError("Ant-v5 model has no body named 'torso'")
        self._torso_body_id = int(torso_body_id)
        self._apply_friction()

    def _observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.append_zero_goal_features:
            return observation
        return _append_goal_features(
            observation, np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        )

    @property
    def friction(self) -> float:
        return float(self.friction_values[self.context_index])

    def _context_label(self) -> str:
        return f"friction_{self.context_index:02d}_{self.friction:.6g}"

    def _apply_friction(self) -> None:
        unwrapped = self.base_env.unwrapped
        # Match the reference XML edit exactly: sliding friction changes,
        # torsional and rolling friction remain fixed at 0.5.
        unwrapped.model.geom_friction[self._floor_geom_id] = (self.friction, 0.5, 0.5)
        mujoco.mj_forward(unwrapped.model, unwrapped.data)

    def _advance_context(self) -> bool:
        if self.cycle:
            next_index = (self.context_index + 1) % len(self.friction_values)
        else:
            next_index = min(self.context_index + 1, len(self.friction_values) - 1)
        changed = next_index != self.context_index
        self.context_index = int(next_index)
        self.context_steps = 0
        self._switch_pending = False
        self._apply_friction()
        return changed

    def _augment_info(
        self, info: dict | None, *, context_changed: bool = False
    ) -> dict:
        result = dict(info or {})
        result.update(
            {
                "terrain": self._context_label(),
                "context_id": int(self.context_index),
                "friction": self.friction,
                "context_steps": int(self.context_steps),
                "global_steps": int(self.total_steps),
                "context_changed": bool(context_changed),
            }
        )
        return result

    def get_state(self) -> dict:
        """Return the protocol state needed to resume after preemption.

        MuJoCo positions and the in-progress episode are intentionally not
        checkpointed: a resumed process starts a fresh episode. The
        friction-stream position and transition counters must survive,
        however, or a resume would silently replay the schedule from the
        first context.
        """
        return {
            "format_version": 1,
            "context_index": int(self.context_index),
            "context_steps": int(self.context_steps),
            "total_steps": int(self.total_steps),
            "switch_pending": bool(self._switch_pending),
            "steps_per_context": int(self.steps_per_context),
            "friction": self.friction,
        }

    def set_state(self, state: dict) -> None:
        """Restore a state produced by :meth:`get_state`.

        The next ``reset`` commits a pending switch exactly as it would have
        at the next episode boundary in an uninterrupted run.
        """
        if int(state.get("format_version", 0)) != 1:
            raise ValueError("Unsupported ContinualAntEnv state format")
        context_index = int(state["context_index"])
        context_steps = int(state["context_steps"])
        total_steps = int(state["total_steps"])
        if not 0 <= context_index < len(self.friction_values):
            raise ValueError(f"Invalid context_index {context_index}")
        if context_steps < 0 or total_steps < 0:
            raise ValueError("Environment step counters must be non-negative")
        if int(state["steps_per_context"]) != self.steps_per_context:
            raise ValueError(
                "Checkpoint steps_per_context does not match the current run"
            )
        expected_friction = float(self.friction_values[context_index])
        if not np.isclose(float(state["friction"]), expected_friction):
            raise ValueError(
                "Checkpoint friction schedule does not match the current run"
            )

        self.context_index = context_index
        self.context_steps = context_steps
        self.total_steps = total_steps
        self._switch_pending = bool(state["switch_pending"])
        self._apply_friction()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        changed = self._advance_context() if self._switch_pending else False
        self._inverted_steps = 0
        observation, info = self.base_env.reset(seed=seed, options=options)
        # Reset may reinitialize data but not model parameters.  Reapplying is
        # cheap and guards against future Gymnasium implementation changes.
        self._apply_friction()
        return self._observation(observation), self._augment_info(
            info, context_changed=changed
        )

    def step(self, action):
        policy_action = np.asarray(action, dtype=np.float32)
        executed_action = policy_action * self.stabilization.action_scale
        observation, native_reward, native_terminated, truncated, info = (
            self.base_env.step(executed_action)
        )
        info = dict(info or {})
        posture = _torso_posture(
            self.base_env.unwrapped.model,
            self.base_env.unwrapped.data,
            self._torso_body_id,
        )
        inverted = bool(
            posture["torso_tilt_degrees"]
            > self.stabilization.inversion_threshold_degrees
        )
        self._inverted_steps = self._inverted_steps + 1 if inverted else 0
        inverted_termination = bool(
            self.stabilization.enabled
            and self._inverted_steps >= self.stabilization.inversion_persistence_steps
        )
        terminated = bool(native_terminated or inverted_termination)

        native_forward = float(info.get("reward_forward", 0.0))
        shaped_forward = native_forward
        reward = float(native_reward)
        if self.stabilization.enabled:
            shaped_forward = stabilized_forward_reward(
                float(info.get("x_velocity", native_forward)),
                float(posture["upright_factor"]),
                self.stabilization.forward_cap_mps,
            )
            reward = reward - native_forward + shaped_forward

        healthy_low, healthy_high = self.base_env.unwrapped._healthy_z_range
        high_z_termination = bool(
            native_terminated and posture["torso_height"] > healthy_high
        )
        low_z_termination = bool(
            native_terminated and posture["torso_height"] < healthy_low
        )
        nonfinite_termination = bool(native_terminated and not posture["finite_state"])
        functional_failure = bool(
            high_z_termination
            or low_z_termination
            or nonfinite_termination
            or inverted_termination
        )

        self.total_steps += 1
        self.context_steps += 1
        if self.context_steps >= self.steps_per_context:
            self._switch_pending = True
        result = self._augment_info(info)
        result.update(posture)
        result.update(
            {
                "reward_native": float(native_reward),
                "reward_forward_native": native_forward,
                "reward_forward": shaped_forward,
                "reward_stabilized": reward,
                "action_scale": float(self.stabilization.action_scale),
                "policy_action_l2": float(np.linalg.norm(policy_action)),
                "executed_action_l2": float(np.linalg.norm(executed_action)),
                "native_terminated": bool(native_terminated),
                "high_z_termination": high_z_termination,
                "low_z_termination": low_z_termination,
                "nonfinite_termination": nonfinite_termination,
                "inverted": inverted,
                "inverted_steps": int(self._inverted_steps),
                "inverted_termination": inverted_termination,
                "functional_failure": functional_failure,
                # Historical training code calls this field ``flip``. Preserve
                # native semantics when V2 is disabled; V2 uses physical failure.
                "flip": (
                    functional_failure
                    if self.stabilization.enabled
                    else bool(native_terminated)
                ),
            }
        )
        result["switch_pending"] = bool(self._switch_pending)
        return self._observation(observation), reward, terminated, truncated, result

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()


class AntMazeTransferEnv(gym.Env):
    """Strict official Gymnasium-Robotics AntMaze-v5 task adapter.

    The policy receives Ant proprioception followed by four goal features.
    ``world`` preserves the historical ``achieved_goal`` plus world-frame
    delta representation. ``body_direction`` supplies the body-frame delta
    and its unit direction. These are observation-only transforms: the
    registered map, physics, reward, success definition, and episode
    termination are passed through unchanged. Posture and locomotion fields
    are diagnostics only and never modify the task.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        *,
        env_id: str = "AntMaze_UMazeDense-v5",
        render_mode: str | None = None,
        goal_feature_mode: str = "world",
        continuing_task: bool = True,
        stabilization: Mapping[str, object] | None = None,
        goal_velocity_reward_weight: float = 0.0,
        include_ant_control_contact_costs: bool = False,
        **maze_kwargs,
    ) -> None:
        super().__init__()
        if not (env_id.startswith("AntMaze_") and env_id.endswith("-v5")):
            raise ValueError(
                "AntMaze transfer is pinned to Gymnasium-Robotics *-v5; "
                f"got {env_id!r}"
            )
        if not math.isfinite(goal_velocity_reward_weight):
            raise ValueError(
                "goal_velocity_reward_weight must be finite and non-negative"
            )
        if float(goal_velocity_reward_weight) != 0.0:
            raise ValueError(
                "strict official AntMaze forbids goal-velocity reward shaping"
            )
        if include_ant_control_contact_costs:
            raise ValueError(
                "strict official AntMaze forbids adding Ant reward components"
            )
        self.stabilization = StabilizationSpec.from_mapping(stabilization)
        if self.stabilization.enabled:
            raise ValueError(
                "strict official AntMaze forbids stabilization intervention"
            )
        if self.stabilization.action_scale != 1.0:
            raise ValueError("strict official AntMaze requires unscaled native actions")
        if maze_kwargs:
            raise ValueError(
                "strict official AntMaze forbids environment overrides; "
                f"got {sorted(maze_kwargs)}"
            )
        if goal_feature_mode not in {"world", "body_direction"}:
            raise ValueError(
                "goal_feature_mode must be 'world' or 'body_direction'; "
                f"got {goal_feature_mode!r}"
            )
        if not isinstance(continuing_task, bool):
            raise TypeError("continuing_task must be a bool")

        try:
            import gymnasium_robotics
        except ImportError as exc:  # pragma: no cover - exercised in clean envs
            raise ImportError(
                "AntMaze requires gymnasium-robotics==1.4.2; install the "
                "ant-transfer optional dependency"
            ) from exc
        gym.register_envs(gymnasium_robotics)

        self.env_id = env_id
        self.goal_feature_mode = goal_feature_mode
        self.continuing_task = continuing_task
        self._inverted_steps = 0
        self._functional_failure_seen = False
        self._physical_failure_seen = False
        self._success_ever = False
        self._successful_steps = 0
        self.total_steps = 0
        # ``continuing_task`` is a documented AntMaze-v5 constructor option.
        # The episodic setting is used by the official Farama D4RL/Minari
        # generation protocol and changes only success termination: map,
        # physics, native reward, goal radius, and action space stay intact.
        self.base_env = gym.make(
            env_id, render_mode=render_mode, continuing_task=continuing_task
        )
        if not isinstance(self.base_env.observation_space, gym.spaces.Dict):
            self.base_env.close()
            raise TypeError("AntMaze-v5 must expose a Dict observation space")
        proprioception_space = self.base_env.observation_space["observation"]
        if not isinstance(proprioception_space, gym.spaces.Box):
            self.base_env.close()
            raise TypeError("AntMaze observation entry must be a Box")
        self.observation_space = _goal_augmented_space(proprioception_space)
        self.action_space = self.base_env.action_space
        self.render_mode = render_mode

        self._maze_env = self.base_env.unwrapped
        self._ant_env = self._maze_env.ant_env
        torso_body_id = mujoco.mj_name2id(
            self._ant_env.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        if torso_body_id < 0:
            self.base_env.close()
            raise RuntimeError("AntMaze Ant model has no body named 'torso'")
        self._torso_body_id = int(torso_body_id)
        self._wall_geom_ids = frozenset(
            index
            for index in range(self._ant_env.model.ngeom)
            if self._ant_env.model.geom(index).name.startswith("block_")
        )
        if not self._wall_geom_ids:
            self.base_env.close()
            raise RuntimeError("registered AntMaze-v5 model has no wall geoms")

    def _observation(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
        desired = np.asarray(observation["desired_goal"], dtype=np.float64)
        delta_world = desired - achieved
        if self.goal_feature_mode == "world":
            goal_features = np.concatenate([achieved, delta_world])
        else:
            rotation = np.asarray(
                self._ant_env.data.xmat[self._torso_body_id],
                dtype=np.float64,
            ).reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            cosine, sine = math.cos(yaw), math.sin(yaw)
            delta_body = np.asarray(
                [
                    cosine * delta_world[0] + sine * delta_world[1],
                    -sine * delta_world[0] + cosine * delta_world[1],
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(delta_world))
            unit_body = (
                delta_body / distance
                if distance > 1e-12
                else np.zeros(2, dtype=np.float64)
            )
            goal_features = np.concatenate([delta_body, unit_body])
        proprioception = np.asarray(observation["observation"], dtype=np.float64)
        return np.concatenate([proprioception, goal_features]).astype(
            self.observation_space.dtype, copy=False
        )

    def _context_label(self) -> str:
        return f"antmaze_{self.env_id.removesuffix('-v5').lower()}"

    def _augment_info(self, info: dict | None) -> dict:
        result = dict(info or {})
        result.update(
            {
                "terrain": self._context_label(),
                "context_id": 0,
                "friction": float("nan"),
                "context_steps": int(self.total_steps),
                "global_steps": int(self.total_steps),
                "context_changed": False,
            }
        )
        return result

    def get_state(self) -> dict:
        return {"format_version": 1, "total_steps": int(self.total_steps)}

    def set_state(self, state: dict) -> None:
        if int(state.get("format_version", 0)) != 1:
            raise ValueError("Unsupported AntMazeTransferEnv state format")
        total_steps = int(state["total_steps"])
        if total_steps < 0:
            raise ValueError("Environment step counter must be non-negative")
        self.total_steps = total_steps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._inverted_steps = 0
        self._functional_failure_seen = False
        self._physical_failure_seen = False
        self._success_ever = False
        self._successful_steps = 0
        observation, info = self.base_env.reset(seed=seed, options=options)
        achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
        desired = np.asarray(observation["desired_goal"], dtype=np.float64)
        result = self._augment_info(info)
        result.update(
            {
                "goal_distance": float(np.linalg.norm(desired - achieved)),
                "success_ever": False,
                "goals_reached": 0,
                "successful_steps": 0,
                "physical_failure": False,
                "achieved_goal": achieved.copy(),
                "desired_goal": desired.copy(),
                "torso_yaw": self._torso_yaw(),
            }
        )
        return self._observation(observation), result

    def step(self, action):
        observation, official_goal_reward, official_terminated, truncated, info = (
            self.base_env.step(action)
        )
        info = dict(info or {})
        posture = _torso_posture(
            self._ant_env.model, self._ant_env.data, self._torso_body_id
        )
        wall_contacts = _wall_contact_diagnostics(
            self._ant_env.model, self._ant_env.data, self._wall_geom_ids
        )
        inverted = bool(
            posture["torso_tilt_degrees"]
            > self.stabilization.inversion_threshold_degrees
        )
        self._inverted_steps = self._inverted_steps + 1 if inverted else 0
        persistent_inversion = bool(
            self._inverted_steps >= self.stabilization.inversion_persistence_steps
        )
        native_unhealthy = bool(not self._ant_env.is_healthy)

        achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
        desired = np.asarray(observation["desired_goal"], dtype=np.float64)
        delta = desired - achieved
        distance = float(np.linalg.norm(delta))
        direction = delta / distance if distance > 1e-12 else np.zeros(2)
        velocity_xy = np.asarray(
            [
                float(info.get("x_velocity", 0.0)),
                float(info.get("y_velocity", 0.0)),
            ]
        )
        goal_velocity = float(np.dot(velocity_xy, direction))

        healthy_low, healthy_high = self._ant_env._healthy_z_range
        high_z_termination = bool(
            native_unhealthy and posture["torso_height"] > healthy_high
        )
        low_z_termination = bool(
            native_unhealthy and posture["torso_height"] < healthy_low
        )
        nonfinite_termination = bool(native_unhealthy and not posture["finite_state"])
        functional_failure_step = bool(
            high_z_termination
            or low_z_termination
            or nonfinite_termination
            or persistent_inversion
        )
        self._functional_failure_seen = bool(
            self._functional_failure_seen or functional_failure_step
        )
        # A transient high-z excursion is common in bounding locomotion and
        # is retained as a diagnostic rather than treated as a fall.  It is
        # a physical failure only when the episode actually ends high.
        terminal_high_z = bool(
            high_z_termination and (official_terminated or truncated)
        )
        terminal_inversion = bool(
            (official_terminated or truncated)
            and inverted
            and not (self._success_ever or info.get("success", False))
        )
        physical_failure_step = bool(
            low_z_termination
            or nonfinite_termination
            or persistent_inversion
            or terminal_inversion
            or terminal_high_z
        )
        self._physical_failure_seen = bool(
            self._physical_failure_seen or physical_failure_step
        )

        self.total_steps += 1
        success_step = bool(info.get("success", False))
        self._success_ever = bool(self._success_ever or success_step)
        self._successful_steps += int(success_step)
        result = self._augment_info(info)
        result.update(posture)
        result.update(wall_contacts)
        result.update(
            {
                "reward_official_goal": float(official_goal_reward),
                "reward_goal_velocity": goal_velocity,
                "reward_stabilized": float(official_goal_reward),
                "goal_distance": distance,
                "goal_velocity": goal_velocity,
                "success_ever": self._success_ever,
                # Count the single distinct goal once. Continuing runs may dwell
                # in the radius; episodic runs terminate on the first success.
                "goals_reached": int(self._success_ever),
                "successful_steps": int(self._successful_steps),
                "achieved_goal": achieved.copy(),
                "desired_goal": desired.copy(),
                "torso_yaw": self._torso_yaw(),
                "goal_terminated": bool(official_terminated),
                "official_terminated": bool(official_terminated),
                "native_terminated": bool(official_terminated),
                "ant_unhealthy": native_unhealthy,
                "high_z_termination": high_z_termination,
                "low_z_termination": low_z_termination,
                "nonfinite_termination": nonfinite_termination,
                "inverted": inverted,
                "inverted_steps": int(self._inverted_steps),
                "persistent_inversion": persistent_inversion,
                "inverted_termination": False,
                "physical_termination": False,
                "physical_failure_step": physical_failure_step,
                "physical_failure": self._physical_failure_seen,
                "functional_failure_step": functional_failure_step,
                "functional_failure": self._functional_failure_seen,
                "flip": self._functional_failure_seen,
            }
        )
        return (
            self._observation(observation),
            float(official_goal_reward),
            bool(official_terminated),
            bool(truncated),
            result,
        )

    def render(self):
        return self.base_env.render()

    def _torso_yaw(self) -> float:
        rotation = np.asarray(
            self._ant_env.data.xmat[self._torso_body_id],
            dtype=np.float64,
        ).reshape(3, 3)
        return float(math.atan2(rotation[1, 0], rotation[0, 0]))

    def close(self) -> None:
        self.base_env.close()

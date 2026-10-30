"""Distil a released AntMaze-v5 SAC teacher into transfer actors.

The environment remains the registered Gymnasium-Robotics AntMaze task.  The
released Farama community controller supplies low-level actions for official
grid waypoints; DAgger then queries the same teacher on states visited by the
student.  This is a warm start for the subsequent TD3+HER experiment, not a
replacement for its online off-policy phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from stable_baselines3 import SAC

from body_control.graph import diagnose_connectome
from body_control.graph import build_connectome

from .maze_planner import OfficialAntMazeWaypointPlanner
from .env import AntMazeTransferEnv
from .evaluate_maze import (
    build_antmaze_gate,
    evaluate_antmaze_run,
    summarize_antmaze,
)
from .goals import encode_body_goal
from .train import (
    CONNECTOME_POLICY_NAMES,
    build_policy_from_config,
    canonical_hash,
    code_provenance,
    file_sha256,
    git_revision,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _normalise(observation: np.ndarray, state: dict, clip: float) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float32)
    variance = np.asarray(state["var"], dtype=np.float32)
    values = np.asarray(observation, dtype=np.float32)
    return np.clip(
        (values - mean) / np.sqrt(variance + 1e-8),
        -clip,
        clip,
    ).astype(np.float32, copy=False)


def validate_distillation_scope(config: dict) -> None:
    env = config.get("env") or {}
    policy = config.get("policy") or {}
    imitation = config.get("imitation") or {}
    evaluation = config.get("evaluation") or {}
    errors = {}
    if config.get("algorithm") != "antmaze_teacher_distillation":
        errors["algorithm"] = config.get("algorithm")
    expected_env = {
        "class_path": "body_control.env.AntMazeTransferEnv",
        "continuing_task": False,
    }
    for key, value in expected_env.items():
        if env.get(key) != value:
            errors[f"env.{key}"] = env.get(key)
    if env.get("env_id") not in {
        "AntMaze_UMaze-v5",
        "AntMaze_Open_Diverse_G-v5",
        "AntMaze_Medium_Diverse_GR-v5",
        "AntMaze_Large_Diverse_GR-v5",
    }:
        errors["env.env_id"] = env.get("env_id")
    if env.get("goal_feature_mode") not in {"body_direction", "world"}:
        errors["env.goal_feature_mode"] = env.get("goal_feature_mode")
    if set(env) != {*expected_env, "env_id", "goal_feature_mode"}:
        errors["env.keys"] = sorted(env)
    navigation = config.get("navigation") or {}
    if set(navigation) - {
        "waypoint_planner",
        "student_goal",
        "evaluation_waypoint_planner",
    }:
        errors["navigation.keys"] = sorted(navigation)
    if navigation.get("waypoint_planner") != "official_grid_waypoints":
        errors["navigation.waypoint_planner"] = navigation.get("waypoint_planner")
    student_goal = navigation.get("student_goal", "waypoint")
    if student_goal not in {"waypoint", "final"}:
        errors["navigation.student_goal"] = student_goal
    evaluation_waypoint_planner = navigation.get(
        "evaluation_waypoint_planner", "official_grid_waypoints"
    )
    if evaluation_waypoint_planner not in {
        "none",
        "official_grid_waypoints",
    }:
        errors["navigation.evaluation_waypoint_planner"] = evaluation_waypoint_planner
    if student_goal == "final" and evaluation_waypoint_planner != "none":
        errors["navigation.evaluation_waypoint_planner"] = (
            "final-goal students must be evaluated without waypoints"
        )
    if policy.get("name") not in {"connectome_rate", "mlp", "gru", "lstm", "transformer", "reservoir", "reservoir_ro"}:
        errors["policy.name"] = policy.get("name")
    if policy.get("name") == "connectome_rate":
        connectome = config.get("connectome") or {}
        if connectome.get("species") != "adult":
            errors["connectome.species"] = connectome.get("species")
        if connectome.get("kind") != "original":
            errors["connectome.kind"] = connectome.get("kind")
    integer_positive = (
        "demonstration_steps",
        "dagger_steps",
        "batch_size",
        "checkpoint_interval_updates",
    )
    for key in integer_positive:
        value = imitation.get(key)
        if not isinstance(value, int) or value <= 0:
            errors[f"imitation.{key}"] = value
    dagger_iterations = imitation.get("dagger_iterations")
    if not isinstance(dagger_iterations, int) or dagger_iterations < 0:
        errors["imitation.dagger_iterations"] = dagger_iterations
    updates_per_stage = imitation.get("updates_per_stage")
    if (
        not isinstance(updates_per_stage, int)
        or updates_per_stage < 0
        or (updates_per_stage == 0 and dagger_iterations == 0)
    ):
        errors["imitation.updates_per_stage"] = updates_per_stage
    dagger_updates = imitation.get("dagger_updates_per_stage")
    if dagger_updates is not None and (
        not isinstance(dagger_updates, int) or dagger_updates <= 0
    ):
        errors["imitation.dagger_updates_per_stage"] = dagger_updates
    rollout_action_smoothing = imitation.get("rollout_action_smoothing", 1.0)
    if (
        not isinstance(rollout_action_smoothing, (int, float))
        or not 0.0 < rollout_action_smoothing <= 1.0
    ):
        errors["imitation.rollout_action_smoothing"] = rollout_action_smoothing
    stop_after_failure = imitation.get("stop_after_student_physical_failure", False)
    if not isinstance(stop_after_failure, bool):
        errors["imitation.stop_after_student_physical_failure"] = stop_after_failure
    for key in ("learning_rate", "gradient_clip"):
        value = imitation.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            errors[f"imitation.{key}"] = value
    burn_in = imitation.get("burn_in")
    if not isinstance(burn_in, int) or burn_in < 0:
        errors["imitation.burn_in"] = burn_in
    if imitation.get("actor_train_mode", "all") not in {
        "all",
        "goal_input_only",
        "goal_interface_only",
        "goal_adapter_only",
        "goal_router_only",
        "base_all",
    }:
        errors["imitation.actor_train_mode"] = imitation.get("actor_train_mode")
    if imitation.get("loss_space", "action") not in {
        "action",
        "pre_tanh",
        "waypoint",
    }:
        errors["imitation.loss_space"] = imitation.get("loss_space")
    if (
        imitation.get("loss_space") == "waypoint"
        and imitation.get("actor_train_mode") != "goal_router_only"
    ):
        errors["imitation.actor_train_mode"] = "waypoint loss requires goal_router_only"
    waypoint_loss = imitation.get("waypoint_loss", "mse")
    if waypoint_loss not in {"mse", "smooth_l1"}:
        errors["imitation.waypoint_loss"] = waypoint_loss
    waypoint_huber_beta = imitation.get("waypoint_huber_beta", 1.0)
    if not isinstance(waypoint_huber_beta, (int, float)) or waypoint_huber_beta <= 0:
        errors["imitation.waypoint_huber_beta"] = waypoint_huber_beta
    if imitation.get("sampling_mode", "transition_uniform") not in {
        "transition_uniform",
        "episode_uniform",
    }:
        errors["imitation.sampling_mode"] = imitation.get("sampling_mode")
    if set(evaluation) - {
        "action_smoothing",
        "tilt_guard_threshold_degrees",
        "tilt_guard_action_scale",
    }:
        errors["evaluation.keys"] = sorted(evaluation)
    action_smoothing = evaluation.get("action_smoothing", 1.0)
    if (
        not isinstance(action_smoothing, (int, float))
        or not 0.0 < action_smoothing <= 1.0
    ):
        errors["evaluation.action_smoothing"] = action_smoothing
    tilt_threshold = evaluation.get("tilt_guard_threshold_degrees")
    if tilt_threshold is not None and (
        not isinstance(tilt_threshold, (int, float)) or not 0.0 < tilt_threshold < 90.0
    ):
        errors["evaluation.tilt_guard_threshold_degrees"] = tilt_threshold
    tilt_scale = evaluation.get("tilt_guard_action_scale", 1.0)
    if not isinstance(tilt_scale, (int, float)) or not 0.0 <= tilt_scale <= 1.0:
        errors["evaluation.tilt_guard_action_scale"] = tilt_scale
    teacher = config.get("teacher") or {}
    if teacher.get("source") != "Farama-Antmaze-Dataset-Generation":
        errors["teacher.source"] = teacher.get("source")
    checksum = teacher.get("checkpoint_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors["teacher.checkpoint_sha256"] = checksum
    exact_mlp = bool(teacher.get("initialize_mlp_exactly", False))
    if exact_mlp and (
        policy.get("name") != "mlp" or env.get("goal_feature_mode") != "world"
    ):
        errors["teacher.initialize_mlp_exactly"] = (
            "requires an MLP with world goal features"
        )
    if errors:
        raise ValueError(f"invalid AntMaze distillation protocol: {errors}")


def _updates_for_stage(imitation: dict, stage: int) -> int:
    """Return the planned update count, allowing weaker DAgger stages."""
    if stage > 0 and imitation.get("dagger_updates_per_stage") is not None:
        return int(imitation["dagger_updates_per_stage"])
    return int(imitation["updates_per_stage"])


def _teacher_observation(
    proprioception: np.ndarray,
    achieved_goal: np.ndarray,
    waypoint: np.ndarray,
) -> np.ndarray:
    """Observation used by the released GoalReachAnt SAC controller."""
    return np.concatenate(
        [
            np.asarray(proprioception, dtype=np.float32),
            np.asarray(waypoint, dtype=np.float32)
            - np.asarray(achieved_goal, dtype=np.float32),
        ]
    ).astype(np.float64, copy=False)


def _student_goal(
    waypoint: np.ndarray,
    final_goal: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Choose information exposed to the student, independently of teacher.

    The released controller may use the official grid route to provide
    low-level action labels.  A ``final`` student never observes those
    waypoints, during either teacher rollouts or DAgger rollouts.
    """
    if mode == "waypoint":
        return np.asarray(waypoint, dtype=np.float32)
    if mode == "final":
        return np.asarray(final_goal, dtype=np.float32)
    raise ValueError(f"unsupported student goal mode: {mode!r}")


def initialize_exact_teacher_mlp(policy, teacher: SAC) -> None:
    """Map the released four-layer SAC mean network into a 109-D MLP.

    AntMazeTransferEnv's world encoding is ``[proprio(105), achieved_xy,
    waypoint_delta_xy]`` while GoalReachAnt uses ``[proprio(105),
    waypoint_delta_xy]``.  The two achieved-position columns are therefore
    exactly zeroed; all other mean-network parameters are copied verbatim.
    """
    target_layers = [
        module for module in policy.actor.net if isinstance(module, torch.nn.Linear)
    ]
    teacher_layers = [
        module
        for module in teacher.policy.actor.latent_pi
        if isinstance(module, torch.nn.Linear)
    ] + [teacher.policy.actor.mu]
    if len(target_layers) != len(teacher_layers):
        raise ValueError("exact teacher MLP requires the released four hidden layers")
    with torch.no_grad():
        first_target, first_teacher = target_layers[0], teacher_layers[0]
        if (
            first_target.in_features != first_teacher.in_features + 2
            or first_target.out_features != first_teacher.out_features
        ):
            raise ValueError("teacher/student first-layer dimensions differ")
        first_target.weight.zero_()
        first_target.weight[:, :105].copy_(first_teacher.weight[:, :105])
        first_target.weight[:, -2:].copy_(first_teacher.weight[:, -2:])
        first_target.bias.copy_(first_teacher.bias)
        for target, source in zip(target_layers[1:], teacher_layers[1:]):
            if target.weight.shape != source.weight.shape:
                raise ValueError("teacher/student hidden dimensions differ")
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)


def load_source_policy(policy, source_state: dict[str, torch.Tensor]) -> None:
    """Load a source policy, optionally expanding its linear motor readout.

    A 109 -> 218 -> 8 ReLU network can reproduce a 109 -> 8 linear map
    exactly using ``[I; -I]`` followed by ``[W, -W]``.  This gives the adult
    connectome a nonlinear task interface without changing its initial gait.
    """
    try:
        policy.load_state_dict(source_state, strict=True)
        return
    except RuntimeError:
        pass
    target = policy.state_dict()
    # A goal-routing wrapper is a zero-residual addition around an existing
    # locomotion actor.  Re-key the source actor under ``actor.base`` while
    # retaining the freshly initialized router tensors.
    if hasattr(policy.actor, "goal_router") and hasattr(policy.actor, "base"):
        mapped = dict(target)
        translated = {
            (
                "actor.base." + key[len("actor.") :]
                if key.startswith("actor.")
                else key
            ): value
            for key, value in source_state.items()
        }
        if set(translated) <= set(target) and all(
            target[key].shape == value.shape for key, value in translated.items()
        ):
            mapped.update(translated)
            policy.load_state_dict(mapped, strict=True)
            return
    # A task adapter is initialized as an exact zero residual.  Copying every
    # matching source tensor therefore preserves the source controller while
    # retaining only the newly initialized adapter tensors.
    adapter_keys = {key for key in target if key.startswith("actor.goal_adapter.")}
    if (
        adapter_keys
        and set(source_state) == set(target) - adapter_keys
        and all(source_state[key].shape == target[key].shape for key in source_state)
    ):
        mapped = dict(target)
        mapped.update(source_state)
        policy.load_state_dict(mapped, strict=True)
        return
    old_weight_key = "actor.core.output_layer.weight"
    old_bias_key = "actor.core.output_layer.bias"
    first_weight_key = "actor.core.output_layer.0.weight"
    first_bias_key = "actor.core.output_layer.0.bias"
    final_weight_key = "actor.core.output_layer.2.weight"
    final_bias_key = "actor.core.output_layer.2.bias"
    required = {
        old_weight_key,
        old_bias_key,
        first_weight_key,
        first_bias_key,
        final_weight_key,
        final_bias_key,
    }
    if not (
        {old_weight_key, old_bias_key} <= source_state.keys()
        and {first_weight_key, first_bias_key, final_weight_key, final_bias_key}
        <= target.keys()
    ):
        raise ValueError("source and target policy structures are incompatible")
    old_weight = source_state[old_weight_key]
    old_bias = source_state[old_bias_key]
    width = old_weight.shape[1]
    if target[first_weight_key].shape != (2 * width, width):
        raise ValueError("source-preserving nonlinear readout requires hidden width 2O")
    mapped = {}
    replaced_target = {
        first_weight_key,
        first_bias_key,
        final_weight_key,
        final_bias_key,
    }
    for key, value in target.items():
        if key in replaced_target:
            continue
        if key not in source_state or source_state[key].shape != value.shape:
            raise ValueError(f"cannot map source policy tensor {key}")
        mapped[key] = source_state[key]
    identity = torch.eye(width, dtype=target[first_weight_key].dtype)
    mapped[first_weight_key] = torch.cat([identity, -identity], dim=0)
    mapped[first_bias_key] = torch.zeros_like(target[first_bias_key])
    mapped[final_weight_key] = torch.cat([old_weight, -old_weight], dim=1)
    mapped[final_bias_key] = old_bias
    policy.load_state_dict(mapped, strict=True)


def _policy_checkpoint(
    policy,
    normalizer: dict,
    *,
    stage: int,
    updates: int,
    protocol_hash: str,
    kind: str,
) -> dict:
    return {
        "format_version": 1,
        "algorithm": "antmaze_teacher_distillation",
        "checkpoint_kind": kind,
        "protocol_hash": protocol_hash,
        "stage": int(stage),
        "updates": int(updates),
        "timesteps": 0,
        "policy_state": {
            key: value.detach().cpu().clone()
            for key, value in policy.state_dict().items()
        },
        "obs_rms_state": deepcopy(normalizer),
    }


def _dataset_path(output: Path, stage: int) -> Path:
    return output / f"teacher_labels_stage{stage}.npz"


def _save_dataset(
    path: Path,
    observation: list[np.ndarray],
    action: list[np.ndarray],
    episode_start: list[bool],
    episode_id: list[int],
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        observation=np.asarray(observation, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        episode_start=np.asarray(episode_start, dtype=np.bool_),
        episode_id=np.asarray(episode_id, dtype=np.int32),
    )
    os.replace(temporary, path)


def _load_datasets(output: Path, final_stage: int) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "observation": [],
        "action": [],
        "episode_start": [],
        "episode_id": [],
    }
    episode_offset = 0
    for stage in range(final_stage + 1):
        with np.load(_dataset_path(output, stage)) as dataset:
            ids = dataset["episode_id"].astype(np.int64)
            for key in fields:
                value = dataset[key]
                if key == "episode_id":
                    value = ids + episode_offset
                fields[key].append(value)
            episode_offset += int(ids.max(initial=-1)) + 1
    return {key: np.concatenate(values) for key, values in fields.items()}


@torch.no_grad()
def _student_action(
    policy,
    observation: np.ndarray,
    hidden: torch.Tensor,
    normalizer: dict,
    clip: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    tensor = torch.as_tensor(
        _normalise(observation, normalizer, clip),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    mean, hidden = policy.actor(tensor, hidden)
    return torch.tanh(mean).squeeze(0).cpu().numpy(), hidden


def collect_teacher_labels(
    *,
    path: Path,
    target_steps: int,
    seed: int,
    teacher: SAC,
    policy,
    normalizer: dict,
    clip: float,
    device: torch.device,
    student_rollout: bool,
    goal_feature_mode: str,
    env_id: str = "AntMaze_UMaze-v5",
    student_goal_mode: str = "waypoint",
    label_space: str = "action",
    rollout_action_smoothing: float = 1.0,
    stop_after_student_physical_failure: bool = False,
) -> dict:
    """Collect teacher states or query labels on student-visited states."""
    if not 0.0 < rollout_action_smoothing <= 1.0:
        raise ValueError("rollout_action_smoothing must be in (0, 1]")
    env = AntMazeTransferEnv(
        env_id=env_id,
        goal_feature_mode=goal_feature_mode,
        continuing_task=False,
    )
    tracker = OfficialAntMazeWaypointPlanner(env._maze_env.maze).tracker()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    starts: list[bool] = []
    ids: list[int] = []
    successes = failures = physical_failures = 0
    student_physical_failure_stops = 0
    episode = 0
    try:
        while len(observations) < target_steps:
            observation, info = env.reset(seed=seed + episode)
            waypoint = tracker.reset(info["achieved_goal"], info["desired_goal"])
            final_goal = np.asarray(info["desired_goal"], dtype=np.float32)
            hidden = torch.zeros(
                1, policy.state_dim, dtype=torch.float32, device=device
            )
            previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
            terminated = truncated = False
            first = True
            while not (terminated or truncated) and len(observations) < target_steps:
                exposed_goal = _student_goal(waypoint, final_goal, student_goal_mode)
                student_observation = encode_body_goal(
                    observation[:-4],
                    info["achieved_goal"],
                    exposed_goal,
                    info["torso_yaw"],
                    goal_feature_mode,
                )
                teacher_observation = _teacher_observation(
                    observation[:-4], info["achieved_goal"], waypoint
                )
                teacher_action, _ = teacher.predict(
                    teacher_observation, deterministic=True
                )
                if student_rollout:
                    environment_action, hidden = _student_action(
                        policy, student_observation, hidden, normalizer, clip, device
                    )
                    environment_action = (
                        rollout_action_smoothing * environment_action
                        + (1.0 - rollout_action_smoothing) * previous_action
                    ).astype(np.float32, copy=False)
                    previous_action = environment_action
                else:
                    environment_action = np.asarray(teacher_action, dtype=np.float32)
                observations.append(student_observation)
                label = (
                    np.asarray(waypoint, dtype=np.float32)
                    - np.asarray(info["achieved_goal"], dtype=np.float32)
                    if label_space == "waypoint"
                    else np.asarray(teacher_action, dtype=np.float32)
                )
                actions.append(label)
                starts.append(first)
                ids.append(episode)
                first = False
                observation, _, terminated, truncated, info = env.step(
                    environment_action
                )
                waypoint, _, _ = tracker.advance(info["achieved_goal"])
                if (
                    student_rollout
                    and stop_after_student_physical_failure
                    and info.get("physical_failure", False)
                ):
                    truncated = True
                    student_physical_failure_stops += 1
            successes += int(info.get("success_ever", info.get("success", False)))
            failures += int(info.get("functional_failure", False))
            physical_failures += int(info.get("physical_failure", False))
            episode += 1
    finally:
        env.close()
    _save_dataset(path, observations, actions, starts, ids)
    return {
        "steps": len(observations),
        "episodes": episode,
        "seed": int(seed),
        "student_rollout": student_rollout,
        "environment_id": env_id,
        "student_goal": student_goal_mode,
        "label_space": label_space,
        "rollout_action_smoothing": float(rollout_action_smoothing),
        "stop_after_student_physical_failure": bool(
            stop_after_student_physical_failure
        ),
        "student_physical_failure_stops": student_physical_failure_stops,
        "online_success_rate": successes / episode,
        "online_physical_failure_rate": physical_failures / episode,
        "online_functional_failure_rate": failures / episode,
    }


def _episode_starts(dataset: dict[str, np.ndarray]) -> np.ndarray:
    starts = np.flatnonzero(dataset["episode_start"])
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("distillation data must start at an episode boundary")
    result = np.empty(len(dataset["observation"]), dtype=np.int64)
    ends = np.r_[starts[1:], len(result)]
    for start, end in zip(starts, ends):
        result[start:end] = start
    return result


def _sample_training_indices(
    dataset: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """Sample transitions uniformly, or give every route episode equal mass."""
    size = len(dataset["observation"])
    if mode == "transition_uniform":
        return rng.integers(0, size, size=batch_size)
    if mode != "episode_uniform":
        raise ValueError(f"unsupported distillation sampling mode: {mode}")
    episode_starts = np.flatnonzero(dataset["episode_start"])
    if len(episode_starts) == 0 or episode_starts[0] != 0:
        raise ValueError("distillation data must start at an episode boundary")
    episode_ends = np.r_[episode_starts[1:], size]
    episodes = rng.integers(0, len(episode_starts), size=batch_size)
    lengths = episode_ends[episodes] - episode_starts[episodes]
    offsets = (rng.random(batch_size) * lengths).astype(np.int64)
    return episode_starts[episodes] + offsets


def _actor_loss(
    policy,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    normalizer: dict,
    clip: float,
    burn_in: int,
    device: torch.device,
    starts: np.ndarray,
    loss_space: str = "action",
    waypoint_loss: str = "mse",
    waypoint_huber_beta: float = 1.0,
) -> torch.Tensor:
    observations = dataset["observation"]
    target_action = torch.as_tensor(
        dataset["action"][indices], dtype=torch.float32, device=device
    )
    if loss_space == "waypoint":
        router = getattr(policy.actor, "route_goal", None)
        if router is None:
            raise TypeError("waypoint loss requires an actor goal router")
        tensor = torch.as_tensor(
            _normalise(observations[indices], normalizer, clip),
            dtype=torch.float32,
            device=device,
        )
        goal_mean = torch.as_tensor(
            np.asarray(normalizer["mean"])[-2:], dtype=torch.float32, device=device
        )
        goal_scale = torch.as_tensor(
            np.sqrt(np.asarray(normalizer["var"])[-2:] + 1e-8),
            dtype=torch.float32,
            device=device,
        )
        target = ((target_action - goal_mean) / goal_scale).clamp(-clip, clip)
        prediction = router(tensor)
        if waypoint_loss == "mse":
            return F.mse_loss(prediction, target)
        if waypoint_loss == "smooth_l1":
            return F.smooth_l1_loss(prediction, target, beta=waypoint_huber_beta)
        raise ValueError(f"unsupported waypoint loss: {waypoint_loss}")
    if loss_space == "pre_tanh":
        target = torch.atanh(target_action.clamp(-0.9999, 0.9999))
    elif loss_space == "action":
        target = target_action
    else:
        raise ValueError(f"unsupported distillation loss space: {loss_space}")
    if not policy.has_state:
        tensor = torch.as_tensor(
            _normalise(observations[indices], normalizer, clip),
            dtype=torch.float32,
            device=device,
        )
        mean, _ = policy.actor(tensor, policy.actor.init_state(len(indices), device))
        prediction = mean if loss_space == "pre_tanh" else torch.tanh(mean)
        return F.mse_loss(prediction, target)

    sequence = np.zeros(
        (burn_in + 1, len(indices), observations.shape[1]), dtype=np.float32
    )
    valid = np.zeros((burn_in + 1, len(indices)), dtype=np.bool_)
    for column, index in enumerate(indices):
        begin = max(int(starts[index]), int(index) - burn_in)
        values = observations[begin : index + 1]
        sequence[-len(values) :, column] = values
        valid[-len(values) :, column] = True
    sequence = _normalise(sequence, normalizer, clip)
    hidden = policy.actor.init_state(len(indices), device)
    mean = None
    for step in range(burn_in + 1):
        tensor = torch.as_tensor(sequence[step], dtype=torch.float32, device=device)
        # Goal input reaches motor output only through recurrent connectome
        # edges.  Keep this short burn-in differentiable; detaching it makes
        # the goal-column gradient exactly zero for the adult graph.
        candidate_mean, candidate_hidden = policy.actor(tensor, hidden)
        mask = torch.as_tensor(valid[step], device=device)[:, None]
        hidden = torch.where(mask, candidate_hidden, hidden)
        mean = candidate_mean
    assert mean is not None
    prediction = mean if loss_space == "pre_tanh" else torch.tanh(mean)
    return F.mse_loss(prediction, target)


def configure_actor_training(
    policy,
    mode: str,
    goal_input_features: int = 4,
) -> tuple[list[torch.nn.Parameter], torch.nn.Parameter | None, int]:
    """Select all parameters or the preserved locomotion goal interface."""
    if mode == "all":
        parameters = [p for p in policy.actor.parameters() if p.requires_grad]
        return parameters, None, sum(parameter.numel() for parameter in parameters)
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    base = getattr(policy.actor, "base", None)
    if mode == "base_all":
        if base is None:
            raise TypeError("base_all requires a wrapped actor")
        parameters = list(base.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters, None, sum(parameter.numel() for parameter in parameters)
    if mode == "goal_adapter_only":
        actor = policy.actor
        while (
            not hasattr(actor, "goal_adapter")
            and getattr(actor, "base", None) is not None
        ):
            actor = actor.base
        adapter = getattr(actor, "goal_adapter", None)
        if adapter is None:
            raise TypeError("goal_adapter_only requires an actor goal_adapter")
        parameters = list(adapter.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters, None, sum(parameter.numel() for parameter in parameters)
    if mode == "goal_router_only":
        router = getattr(policy.actor, "goal_router", None)
        if router is None:
            raise TypeError("goal_router_only requires an actor goal_router")
        parameters = list(router.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters, None, sum(parameter.numel() for parameter in parameters)
    actor = policy.actor
    while (
        not hasattr(actor, "core")
        and not hasattr(actor, "net")
        and getattr(actor, "base", None) is not None
    ):
        actor = actor.base
    if hasattr(actor, "core") and hasattr(
        actor.core,
        "input_proj",
    ):
        input_layer = actor.core.input_proj
        output_layer = actor.core.output_layer
    elif hasattr(actor, "net"):
        input_layer = actor.net[0]
        output_layer = next(
            module
            for module in reversed(actor.net)
            if isinstance(module, torch.nn.Linear)
        )
    else:
        raise TypeError(f"cannot find policy interface for {type(policy.actor)}")
    input_layer.weight.requires_grad_(True)
    parameters: list[torch.nn.Parameter] = [input_layer.weight]
    if mode == "goal_interface_only":
        for parameter in output_layer.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    effective = input_layer.out_features * goal_input_features
    if mode == "goal_interface_only":
        effective += sum(parameter.numel() for parameter in output_layer.parameters())
    return parameters, input_layer.weight, effective


def train_stage(
    *,
    policy,
    optimizer,
    dataset: dict[str, np.ndarray],
    normalizer: dict,
    clip: float,
    burn_in: int,
    batch_size: int,
    gradient_clip: float,
    target_updates: int,
    checkpoint_interval: int,
    stage: int,
    output: Path,
    protocol_hash: str,
    rng: np.random.Generator,
    device: torch.device,
    start_update: int,
    goal_input_weight: torch.nn.Parameter | None,
    goal_input_features: int,
    loss_space: str,
    sampling_mode: str = "transition_uniform",
    waypoint_loss: str = "mse",
    waypoint_huber_beta: float = 1.0,
) -> list[dict]:
    starts = _episode_starts(dataset)
    history = []
    for update in range(start_update, target_updates):
        indices = _sample_training_indices(dataset, batch_size, rng, sampling_mode)
        optimizer.zero_grad(set_to_none=True)
        loss = _actor_loss(
            policy,
            dataset,
            indices,
            normalizer,
            clip,
            burn_in,
            device,
            starts,
            loss_space,
            waypoint_loss,
            waypoint_huber_beta,
        )
        loss.backward()
        if goal_input_weight is not None:
            if goal_input_weight.grad is None:
                raise RuntimeError("goal input projection received no gradient")
            goal_input_weight.grad[:, :-goal_input_features].zero_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.actor.parameters(), gradient_clip
        )
        optimizer.step()
        absolute_update = update + 1
        if absolute_update == 1 or absolute_update % 250 == 0:
            row = {
                "stage": stage,
                "update": absolute_update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if absolute_update % checkpoint_interval == 0:
            checkpoint = _policy_checkpoint(
                policy,
                normalizer,
                stage=stage,
                updates=absolute_update,
                protocol_hash=protocol_hash,
                kind="resume",
            )
            checkpoint.update(
                {
                    "optimizer_state": optimizer.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                }
            )
            _atomic_torch_save(checkpoint, output / "distill_resume.pt")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--collection-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--dagger-updates-per-stage", type=int)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if args.dagger_updates_per_stage is not None:
        config["imitation"]["dagger_updates_per_stage"] = args.dagger_updates_per_stage
    validate_distillation_scope(config)
    if file_sha256(args.teacher) != config["teacher"]["checkpoint_sha256"]:
        raise ValueError("teacher checkpoint checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    collection_seed = (
        args.collection_seed
        if args.collection_seed is not None
        else args.seed * 100_000
    )
    evaluation_seed = (
        args.evaluation_seed
        if args.evaluation_seed is not None
        else args.seed * 100_000 + 50_000
    )
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else args.seed
    resolved = {
        **config,
        "seed": args.seed,
        "seed_roles": {
            "optimization": args.seed,
            "sampling": sampling_seed,
            "collection": collection_seed,
            "evaluation": evaluation_seed,
        },
        "resolved_env": {
            key: value for key, value in config["env"].items() if key != "class_path"
        },
        "initialization": {
            "checkpoint": str(args.source.resolve()),
            "checkpoint_sha256": file_sha256(args.source),
        },
        "teacher": {
            **config["teacher"],
            "checkpoint": str(args.teacher.resolve()),
        },
    }
    protocol_hash = canonical_hash(resolved)
    resolved["provenance"] = {
        "source_config": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "protocol_sha256": protocol_hash,
        **code_provenance(),
    }
    (args.output / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    status = {
        "status": "running",
        "seed": args.seed,
        "seed_roles": resolved["seed_roles"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "device": args.device,
    }
    _write_json(args.output / "pipeline_status.json", status)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(sampling_seed)
    device = torch.device(args.device)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    normalizer = deepcopy(source["obs_rms_state"])
    if config["teacher"].get("initialize_mlp_exactly", False):
        # The released SB3 controller was trained on raw GoalReachAnt inputs.
        # A neutral state makes our evaluator feed the same values exactly.
        normalizer = {
            "mean": np.zeros_like(normalizer["mean"], dtype=np.float64),
            "var": np.ones_like(normalizer["var"], dtype=np.float64),
            "count": float(normalizer.get("count", 1.0)),
        }
    probe = AntMazeTransferEnv(
        env_id=config["env"]["env_id"],
        goal_feature_mode=config["env"]["goal_feature_mode"],
        continuing_task=False,
    )
    observation_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    probe.close()
    connectome = (
        build_connectome(config.get("connectome"))
        if config["policy"]["name"] in CONNECTOME_POLICY_NAMES
        else None
    )
    policy = build_policy_from_config(
        config, observation_dim, action_dim, connectome=connectome
    )
    configure_router = getattr(policy.actor, "configure_goal_router_normalizer", None)
    if configure_router is not None:
        configure_router(normalizer["mean"], normalizer["var"])
    teacher = SAC.load(args.teacher, device=device)
    teacher.policy.eval()
    if config["teacher"].get("initialize_mlp_exactly", False):
        initialize_exact_teacher_mlp(policy, teacher)
    else:
        load_source_policy(policy, source["policy_state"])
    policy.to(device).train()
    imitation = config["imitation"]
    clip = float(config.get("observation_clip", 10.0))
    actor_parameters, goal_input_weight, effective_actor_parameters = (
        configure_actor_training(
            policy,
            imitation.get("actor_train_mode", "all"),
            int(imitation.get("goal_input_features", 4)),
        )
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=float(imitation["learning_rate"]))
    start_stage = start_update = 0
    resume_path = args.output / "distill_resume.pt"
    if args.resume_if_available and resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("protocol_hash") != protocol_hash:
            raise ValueError("distillation resume protocol hash mismatch")
        policy.load_state_dict(resume["policy_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        rng.bit_generator.state = resume["numpy_rng_state"]
        start_stage = int(resume["stage"])
        start_update = int(resume["updates"])
    initial = _policy_checkpoint(
        policy,
        normalizer,
        stage=-1,
        updates=0,
        protocol_hash=protocol_hash,
        kind="initial",
    )
    if not (args.output / "initial_checkpoint.pt").exists():
        _atomic_torch_save(initial, args.output / "initial_checkpoint.pt")

    metadata = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "git_commit": git_revision(),
        "policy": config["policy"]["name"],
        "actor_class": policy.actor.__class__.__name__,
        "actor_parameters": sum(
            parameter.numel() for parameter in policy.actor.parameters()
        ),
        "effective_trainable_actor_parameters": effective_actor_parameters,
        "actor_train_mode": imitation.get("actor_train_mode", "all"),
        "sampling_mode": imitation.get("sampling_mode", "transition_uniform"),
        "seed_roles": resolved["seed_roles"],
        "teacher_checkpoint_sha256": file_sha256(args.teacher),
        "source_checkpoint_sha256": file_sha256(args.source),
        "protocol_sha256": protocol_hash,
        "source_files_sha256": code_provenance()["source_files_sha256"],
    }
    if connectome is not None:
        graph = diagnose_connectome(connectome)
        _write_json(args.output / "graph_diagnostics.json", graph)
        metadata["graph"] = {
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
            "storage": graph["storage"],
            "weight_multiset_sha256": graph["weights"]["sorted_multiset_sha256"],
        }
    _write_json(args.output / "run_metadata.json", metadata)

    started = time.time()
    collection = (
        _read_json(args.output / "collection_summary.json").get("stages", [])
        if (args.output / "collection_summary.json").exists()
        else []
    )
    history = (
        _read_json(args.output / "training_history.json").get("rows", [])
        if (args.output / "training_history.json").exists()
        else []
    )
    total_stages = int(imitation["dagger_iterations"]) + 1
    try:
        for stage in range(start_stage, total_stages):
            path = _dataset_path(args.output, stage)
            if not path.exists():
                status.update({"status": "collecting", "stage": stage})
                _write_json(args.output / "pipeline_status.json", status)
                collection.append(
                    collect_teacher_labels(
                        path=path,
                        target_steps=int(
                            imitation["demonstration_steps"]
                            if stage == 0
                            else imitation["dagger_steps"]
                        ),
                        seed=collection_seed + stage * 10_000,
                        teacher=teacher,
                        policy=policy,
                        normalizer=normalizer,
                        clip=clip,
                        device=device,
                        student_rollout=stage > 0,
                        goal_feature_mode=config["env"]["goal_feature_mode"],
                        env_id=config["env"]["env_id"],
                        student_goal_mode=(config.get("navigation") or {}).get(
                            "student_goal", "waypoint"
                        ),
                        label_space=imitation.get("loss_space", "action"),
                        rollout_action_smoothing=float(
                            imitation.get("rollout_action_smoothing", 1.0)
                        ),
                        stop_after_student_physical_failure=bool(
                            imitation.get("stop_after_student_physical_failure", False)
                        ),
                    )
                )
                _write_json(
                    args.output / "collection_summary.json", {"stages": collection}
                )
            dataset = _load_datasets(args.output, stage)
            status.update(
                {
                    "status": "training",
                    "stage": stage,
                    "aggregated_labels": len(dataset["observation"]),
                }
            )
            _write_json(args.output / "pipeline_status.json", status)
            history.extend(
                train_stage(
                    policy=policy,
                    optimizer=optimizer,
                    dataset=dataset,
                    normalizer=normalizer,
                    clip=clip,
                    burn_in=int(imitation["burn_in"]),
                    batch_size=int(imitation["batch_size"]),
                    gradient_clip=float(imitation["gradient_clip"]),
                    target_updates=_updates_for_stage(imitation, stage),
                    checkpoint_interval=int(imitation["checkpoint_interval_updates"]),
                    stage=stage,
                    output=args.output,
                    protocol_hash=protocol_hash,
                    rng=rng,
                    device=device,
                    start_update=start_update if stage == start_stage else 0,
                    goal_input_weight=goal_input_weight,
                    goal_input_features=int(imitation.get("goal_input_features", 4)),
                    loss_space=imitation.get("loss_space", "action"),
                    sampling_mode=imitation.get("sampling_mode", "transition_uniform"),
                    waypoint_loss=imitation.get("waypoint_loss", "mse"),
                    waypoint_huber_beta=float(
                        imitation.get("waypoint_huber_beta", 1.0)
                    ),
                )
            )
            _write_json(args.output / "training_history.json", {"rows": history})
            start_update = 0
            checkpoint = _policy_checkpoint(
                policy,
                normalizer,
                stage=stage,
                updates=_updates_for_stage(imitation, stage),
                protocol_hash=protocol_hash,
                kind="stage_complete",
            )
            _atomic_torch_save(checkpoint, args.output / "checkpoint.pt")
            _atomic_torch_save(checkpoint, args.output / "best_checkpoint.pt")
            # Point resume at the next stage so a watchdog restart does not
            # repeat a completed optimisation stage.
            next_resume = {
                **checkpoint,
                "stage": stage + 1,
                "updates": 0,
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
            }
            _atomic_torch_save(next_resume, resume_path)

        status["status"] = "evaluating"
        _write_json(args.output / "pipeline_status.json", status)
        records = evaluate_antmaze_run(
            args.output,
            config["env"]["env_id"],
            episodes=args.evaluation_episodes,
            seed=evaluation_seed,
            device=args.device,
            checkpoint="best",
            waypoint_planner=(
                (config.get("navigation") or {}).get(
                    "evaluation_waypoint_planner", "official_grid_waypoints"
                )
            ),
            continuing_task=False,
            action_smoothing=float(
                (config.get("evaluation") or {}).get("action_smoothing", 1.0)
            ),
            tilt_guard_threshold_degrees=(
                (config.get("evaluation") or {}).get("tilt_guard_threshold_degrees")
            ),
            tilt_guard_action_scale=float(
                (config.get("evaluation") or {}).get("tilt_guard_action_scale", 1.0)
            ),
        )
        _write_json(args.output / "eval_best_official_antmaze.json", records)
        summary = summarize_antmaze(records)
        _write_json(args.output / "antmaze_evaluation_summary.json", summary)
        gate = build_antmaze_gate(summary, environment_id=config["env"]["env_id"])
        _write_json(args.output / "antmaze_gate.json", gate)
        elapsed = time.time() - started
        result = {
            "seed": args.seed,
            "policy": config["policy"]["name"],
            "teacher_labels": sum(
                len(np.load(_dataset_path(args.output, stage))["observation"])
                for stage in range(total_stages)
            ),
            "gradient_updates": sum(
                _updates_for_stage(imitation, stage) for stage in range(total_stages)
            ),
            "elapsed_wall_seconds": elapsed,
            "gate_pass": gate["passed"],
            "protocol_sha256": protocol_hash,
        }
        _write_json(args.output / "summary.json", result)
        metadata.update(
            {
                **result,
                "status": "complete",
                "teacher_label_datasets": [
                    {
                        "stage": stage,
                        "steps": int(
                            len(
                                np.load(_dataset_path(args.output, stage))[
                                    "observation"
                                ]
                            )
                        ),
                        "sha256": file_sha256(_dataset_path(args.output, stage)),
                    }
                    for stage in range(total_stages)
                ],
                "cuda_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                ),
            }
        )
        _write_json(args.output / "run_metadata.json", metadata)
        status.update(
            {
                "status": "complete",
                "gate_pass": gate["passed"],
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(args.output / "pipeline_status.json", status)
        print(json.dumps({**result, "gate": gate}, indent=2))
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "error": repr(exc),
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(args.output / "pipeline_status.json", status)
        raise


if __name__ == "__main__":
    main()

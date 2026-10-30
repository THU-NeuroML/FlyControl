"""Frozen-policy evaluation on the official Gymnasium-Robotics AntMaze-v5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .maze_planner import OfficialAntMazeWaypointPlanner
from .env import AntMazeTransferEnv
from .evaluate import _normalize_observation, enable_deterministic_evaluation
from .goals import encode_body_goal
from .train import build_policy_from_config


ANTMAZE_GATE_SCHEMA_VERSION = 2
ANTMAZE_PHYSICAL_FAILURE_DEFINITION = (
    "persistent_inversion_seen OR inverted_completion OR "
    "low_z_failure_seen OR nonfinite_failure_seen OR terminal_high_z"
)


def antmaze_physical_failure(record: dict) -> bool:
    """Return the preregisterable physical-failure label for one episode.

    ``high_z_failure_seen`` is deliberately excluded: Ant locomotion can
    briefly raise the torso above the native healthy-height interval while
    bounding and then recover.  A high-z state counts only when it is the
    terminal state.  The broad historical ``functional_failure`` metric is
    retained separately for auditability.
    """
    return bool(
        record.get("persistent_inversion_seen", False)
        or record.get("inverted_completion", False)
        or record.get("low_z_failure_seen", False)
        or record.get("nonfinite_failure_seen", False)
        or record.get("high_z_termination", False)
    )


def build_antmaze_gate(
    summary: dict,
    *,
    environment_id: str | None = None,
    minimum_success_rate: float = 0.5,
    maximum_physical_failure_rate: float = 0.2,
) -> dict:
    """Build the versioned AntMaze gate and retain the legacy diagnostic."""
    success_rate = float(summary["success"]["mean"])
    physical_failure_rate = float(summary["physical_failure"]["mean"])
    legacy_failure_rate = float(summary["functional_failure"]["mean"])
    high_z_excursion_rate = float(summary["high_z_failure_seen"]["mean"])
    gate = {
        "schema_version": ANTMAZE_GATE_SCHEMA_VERSION,
        "gate_version": "antmaze_physical_failure_v2",
        "passed": bool(
            success_rate >= minimum_success_rate
            and physical_failure_rate <= maximum_physical_failure_rate
        ),
        "minimum_success_rate": minimum_success_rate,
        "maximum_physical_failure_rate": maximum_physical_failure_rate,
        "observed_success_rate": success_rate,
        "observed_physical_failure_rate": physical_failure_rate,
        "physical_failure_definition": ANTMAZE_PHYSICAL_FAILURE_DEFINITION,
        "transient_high_z_is_failure": False,
        "observed_high_z_excursion_rate": high_z_excursion_rate,
        "legacy_gate": {
            "passed": bool(
                success_rate >= minimum_success_rate
                and legacy_failure_rate <= maximum_physical_failure_rate
            ),
            "maximum_functional_failure_rate": (maximum_physical_failure_rate),
            "observed_functional_failure_rate": legacy_failure_rate,
        },
    }
    if environment_id is not None:
        gate["environment_id"] = environment_id
    return gate


def _adapter_kwargs(
    config: dict,
    env_id: str,
    continuing_task: bool | None = None,
) -> dict:
    env_config = config.get("resolved_env") or config.get("env") or {}
    return {
        "env_id": env_id,
        "goal_feature_mode": env_config.get("goal_feature_mode", "world"),
        "continuing_task": (
            env_config.get("continuing_task", True)
            if continuing_task is None
            else continuing_task
        ),
    }


def _actor_attribute_owner(actor, attribute: str):
    """Find an attribute through transparent policy wrappers."""
    current = actor
    while current is not None:
        if hasattr(current, attribute):
            return current
        current = getattr(current, "base", None)
    return None


def _selected_waypoint_planner(
    config: dict,
    override: str | None,
) -> str:
    """Select the deployment planner, not the label-collection planner."""
    if override is not None:
        return override
    navigation = config.get("navigation") or {}
    return navigation.get(
        "evaluation_waypoint_planner",
        navigation.get("waypoint_planner", "none"),
    )


def evaluate_antmaze_run(
    run_dir: Path,
    env_id: str,
    *,
    episodes: int,
    seed: int,
    device: str,
    checkpoint: str = "best",
    waypoint_planner: str | None = None,
    continuing_task: bool | None = None,
    goal_adapter_scale: float | None = None,
    goal_adapter_motor_scale: float | None = None,
    connectome_output_scale: float | None = None,
    action_smoothing: float = 1.0,
    goal_router_smoothing: float | None = None,
    action_scale: float = 1.0,
    tilt_guard_threshold_degrees: float | None = None,
    tilt_guard_action_scale: float = 1.0,
) -> list[dict]:
    """Evaluate the strict official task plus read-only locomotion metrics."""
    enable_deterministic_evaluation()
    if not 0.0 < action_smoothing <= 1.0:
        raise ValueError("action_smoothing must be in (0, 1]")
    if not 0.0 < action_scale <= 1.0:
        raise ValueError("action_scale must be in (0, 1]")
    if (
        tilt_guard_threshold_degrees is not None
        and not 0.0 < tilt_guard_threshold_degrees < 90.0
    ):
        raise ValueError("tilt_guard_threshold_degrees must be in (0, 90)")
    if not 0.0 <= tilt_guard_action_scale <= 1.0:
        raise ValueError("tilt_guard_action_scale must be in [0, 1]")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if goal_router_smoothing is not None:
        if not 0.0 < goal_router_smoothing <= 1.0:
            raise ValueError("goal_router_smoothing must be in (0, 1]")
        actor_kwargs = config["policy"].setdefault("actor_kwargs", {})
        if "goal_router_hidden" not in actor_kwargs:
            raise ValueError("goal_router_smoothing requires a learned goal router")
        actor_kwargs["goal_router_smoothing"] = float(goal_router_smoothing)
    checkpoint_paths = {
        "latest": run_dir / "checkpoint.pt",
        "best": run_dir / "best_checkpoint.pt",
        "initial": run_dir / "initial_checkpoint.pt",
    }
    checkpoint_path = checkpoint_paths[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_data = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    adapter_kwargs = _adapter_kwargs(config, env_id, continuing_task)
    probe = AntMazeTransferEnv(**adapter_kwargs)
    policy = build_policy_from_config(
        config, probe.observation_space.shape[0], probe.action_space.shape[0]
    )
    probe.close()
    configure_router = getattr(policy.actor, "configure_goal_router_normalizer", None)
    if configure_router is not None:
        normalizer = checkpoint_data["obs_rms_state"]
        configure_router(normalizer["mean"], normalizer["var"])
    offpolicy_actor = "actor_state" in checkpoint_data
    if offpolicy_actor:
        policy.actor.load_state_dict(checkpoint_data["actor_state"])
    else:
        policy.load_state_dict(checkpoint_data["policy_state"])
    if goal_adapter_scale is not None:
        owner = _actor_attribute_owner(policy.actor, "goal_adapter")
        if owner is None or owner.goal_adapter is None:
            raise ValueError("goal_adapter_scale requires a policy with a goal adapter")
        if goal_adapter_scale < 0.0:
            raise ValueError("goal_adapter_scale must be non-negative")
        owner.goal_adapter_scale = float(goal_adapter_scale)
    for value, attribute in (
        (goal_adapter_motor_scale, "goal_adapter_motor_scale"),
        (connectome_output_scale, "connectome_output_scale"),
    ):
        if value is None:
            continue
        owner = _actor_attribute_owner(policy.actor, attribute)
        if owner is None:
            raise ValueError(f"{attribute} requires a connectome-rate actor")
        if value < 0.0:
            raise ValueError(f"{attribute} must be non-negative")
        setattr(owner, attribute, float(value))
    policy.to(device).eval()

    records: list[dict] = []
    env = AntMazeTransferEnv(**adapter_kwargs)
    selected_planner = _selected_waypoint_planner(config, waypoint_planner)
    if selected_planner not in {"none", "official_grid_waypoints"}:
        raise ValueError(f"unsupported waypoint planner: {selected_planner}")
    diagnostic_planner = OfficialAntMazeWaypointPlanner(env._maze_env.maze)
    tracker = (
        diagnostic_planner.tracker()
        if selected_planner == "official_grid_waypoints"
        else None
    )
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, reset_info = env.reset(seed=episode_seed)
            waypoint = (
                tracker.reset(reset_info["achieved_goal"], reset_info["desired_goal"])
                if tracker is not None
                else None
            )
            hidden = torch.zeros(
                1, policy.state_dim, dtype=torch.float32, device=device
            )
            positions = [
                np.asarray(
                    [
                        reset_info["x_position"],
                        reset_info["y_position"],
                    ],
                    dtype=np.float64,
                )
            ]
            direct_distance = float(reset_info["goal_distance"])
            shortest_cells = diagnostic_planner.shortest_path(
                reset_info["achieved_goal"], reset_info["desired_goal"]
            )
            route_points = [
                np.asarray(reset_info["achieved_goal"], dtype=np.float64),
                *(
                    np.asarray(
                        diagnostic_planner.maze.cell_rowcol_to_xy(np.asarray(cell)),
                        dtype=np.float64,
                    )
                    for cell in shortest_cells[1:-1]
                ),
                np.asarray(reset_info["desired_goal"], dtype=np.float64),
            ]
            shortest_path_length = float(
                sum(
                    np.linalg.norm(right - left)
                    for left, right in zip(route_points[:-1], route_points[1:])
                )
            )
            grid_directions = np.diff(
                np.asarray(shortest_cells, dtype=np.int64), axis=0
            )
            shortest_path_turns = int(
                sum(
                    not np.array_equal(left, right)
                    for left, right in zip(grid_directions[:-1], grid_directions[1:])
                )
            )
            official_return = 0.0
            path_length_running = 0.0
            path_length_to_first_goal = None
            first_success_step = None
            successful_steps = 0
            action_costs: list[float] = []
            speeds: list[float] = []
            goal_velocities: list[float] = []
            tilts: list[float] = []
            heights: list[float] = []
            upright: list[float] = []
            wall_contact_counts: list[int] = []
            wall_contact_forces: list[float] = []
            inverted_steps = 0
            persistent_inversion_steps = 0
            native_unhealthy_steps = 0
            low_z_failure_steps = 0
            high_z_failure_steps = 0
            nonfinite_failure_steps = 0
            functional_failure_steps = 0
            terminated = truncated = False
            info: dict = {}
            length = 0
            waypoint_switches = 0
            waypoints_reached = 0
            info = reset_info
            previous_action: torch.Tensor | None = None
            tilt_guard_steps = 0
            while not (terminated or truncated):
                policy_observation = observation
                if tracker is not None:
                    assert waypoint is not None
                    policy_observation = encode_body_goal(
                        observation[:-4],
                        info["achieved_goal"],
                        waypoint,
                        info["torso_yaw"],
                        adapter_kwargs["goal_feature_mode"],
                    )
                obs_tensor = torch.as_tensor(
                    _normalize_observation(policy_observation, checkpoint_data, config),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    if offpolicy_actor:
                        mean, hidden = policy.actor(obs_tensor, hidden)
                        action = torch.tanh(mean)
                    else:
                        action, _, _, hidden = policy.act(
                            obs_tensor, hidden, deterministic=True
                        )
                    action = action_scale * action
                    if previous_action is None:
                        previous_action = torch.zeros_like(action)
                    action = (
                        action_smoothing * action
                        + (1.0 - action_smoothing) * previous_action
                    )
                    if (
                        tilt_guard_threshold_degrees is not None
                        and float(info.get("torso_tilt_degrees", 0.0))
                        >= tilt_guard_threshold_degrees
                    ):
                        action = tilt_guard_action_scale * action
                        tilt_guard_steps += 1
                    previous_action = action
                observation, reward, terminated, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                if tracker is not None:
                    waypoint, switched, newly_reached = tracker.advance(
                        info["achieved_goal"]
                    )
                    waypoint_switches += int(switched)
                    waypoints_reached += int(newly_reached)
                official_return += float(reward)
                length += 1
                position = np.asarray(
                    [
                        info["x_position"],
                        info["y_position"],
                    ],
                    dtype=np.float64,
                )
                path_length_running += float(np.linalg.norm(position - positions[-1]))
                positions.append(position)
                if info.get("success", False):
                    successful_steps += 1
                    if first_success_step is None:
                        first_success_step = length
                        path_length_to_first_goal = path_length_running
                action_costs.append(-float(info.get("reward_ctrl", 0.0)))
                speeds.append(
                    float(
                        np.hypot(
                            info.get("x_velocity", 0.0), info.get("y_velocity", 0.0)
                        )
                    )
                )
                goal_velocities.append(float(info["goal_velocity"]))
                tilts.append(float(info["torso_tilt_degrees"]))
                heights.append(float(info["torso_height"]))
                upright.append(float(info["upright_factor"]))
                wall_contact_counts.append(int(info["wall_contact_count"]))
                wall_contact_forces.append(float(info["wall_contact_normal_force"]))
                inverted_steps += int(info.get("inverted", False))
                persistent_inversion_steps += int(
                    info.get("persistent_inversion", False)
                )
                native_unhealthy_steps += int(info.get("ant_unhealthy", False))
                low_z_failure_steps += int(info.get("low_z_termination", False))
                high_z_failure_steps += int(info.get("high_z_termination", False))
                nonfinite_failure_steps += int(info.get("nonfinite_termination", False))
                functional_failure_steps += int(
                    info.get("functional_failure_step", False)
                )

            positions_array = np.asarray(positions, dtype=np.float64)
            segment_lengths = np.linalg.norm(np.diff(positions_array, axis=0), axis=1)
            path_length = float(segment_lengths.sum())
            final_goal_distance = float(info.get("goal_distance", np.nan))
            route_denominator = path_length + final_goal_distance
            success = successful_steps > 0
            final_inverted = bool(tilts and tilts[-1] > 90.0)
            inverted_completion = bool(truncated and not success and final_inverted)
            functional_failure = bool(
                info.get("functional_failure", False) or inverted_completion
            )
            physical_failure = bool(
                persistent_inversion_steps
                or inverted_completion
                or low_z_failure_steps
                or nonfinite_failure_steps
                or info.get("high_z_termination", False)
            )
            goal_velocities_array = np.asarray(goal_velocities, dtype=np.float64)
            speeds_array = np.asarray(speeds, dtype=np.float64)
            lateral_speeds = np.sqrt(
                np.maximum(speeds_array**2 - goal_velocities_array**2, 0.0)
            )
            records.append(
                {
                    "environment_id": env_id,
                    "checkpoint": checkpoint,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "goal_adapter_scale": float(
                        getattr(
                            _actor_attribute_owner(policy.actor, "goal_adapter_scale"),
                            "goal_adapter_scale",
                            1.0,
                        )
                    ),
                    "goal_adapter_motor_scale": float(
                        getattr(
                            _actor_attribute_owner(
                                policy.actor, "goal_adapter_motor_scale"
                            ),
                            "goal_adapter_motor_scale",
                            1.0,
                        )
                    ),
                    "connectome_output_scale": float(
                        getattr(
                            _actor_attribute_owner(
                                policy.actor, "connectome_output_scale"
                            ),
                            "connectome_output_scale",
                            1.0,
                        )
                    ),
                    "action_smoothing": float(action_smoothing),
                    "action_scale": float(action_scale),
                    "tilt_guard_threshold_degrees": (
                        float(tilt_guard_threshold_degrees)
                        if tilt_guard_threshold_degrees is not None
                        else None
                    ),
                    "tilt_guard_action_scale": float(tilt_guard_action_scale),
                    "tilt_guard_step_fraction": tilt_guard_steps / length,
                    "goal_router_smoothing": float(
                        getattr(policy.actor, "goal_router_smoothing", 1.0)
                    ),
                    "episode": episode,
                    "seed": episode_seed,
                    "official_return": official_return,
                    # Success is latched for compatibility with both documented
                    # protocols: continuing tasks may leave the radius again,
                    # whereas episodic tasks terminate on the successful step.
                    "success": success,
                    "terminal_step_success": bool(info.get("success", False)),
                    "goals_reached": int(success),
                    "successful_steps": successful_steps,
                    "success_step_fraction": successful_steps / length,
                    "length": length,
                    "time_to_goal_seconds": (
                        first_success_step * 0.05
                        if first_success_step is not None
                        else None
                    ),
                    "initial_direct_distance": direct_distance,
                    "shortest_path_length": shortest_path_length,
                    "shortest_path_cell_steps": len(shortest_cells) - 1,
                    "shortest_path_turns": shortest_path_turns,
                    "final_goal_distance": final_goal_distance,
                    "path_length": path_length,
                    "path_length_to_first_goal": path_length_to_first_goal,
                    "success_weighted_path_efficiency": (
                        direct_distance
                        / max(direct_distance, float(path_length_to_first_goal))
                        if path_length_to_first_goal is not None
                        else 0.0
                    ),
                    "success_weighted_geodesic_efficiency": (
                        shortest_path_length
                        / max(
                            shortest_path_length,
                            float(path_length_to_first_goal),
                        )
                        if path_length_to_first_goal is not None
                        else 0.0
                    ),
                    "time_to_goal_per_geodesic_meter": (
                        first_success_step * 0.05 / shortest_path_length
                        if first_success_step is not None
                        and shortest_path_length > 1e-12
                        else None
                    ),
                    # Includes the uncompleted straight-line remainder, so failed
                    # episodes cannot obtain efficiency > 1 merely by standing.
                    "direct_path_efficiency": (
                        direct_distance / route_denominator
                        if route_denominator > 1e-12
                        else 0.0
                    ),
                    "mean_speed_xy": float(np.mean(speeds)),
                    "mean_goal_velocity": float(np.mean(goal_velocities)),
                    "mean_lateral_speed_to_goal": float(np.mean(lateral_speeds)),
                    "backward_from_goal_fraction": float(
                        np.mean(goal_velocities_array < 0.0)
                    ),
                    "positive_goal_progress_fraction": float(
                        np.mean(goal_velocities_array > 0.0)
                    ),
                    "wall_contact_step_fraction": float(
                        np.mean(np.asarray(wall_contact_counts) > 0)
                    ),
                    "wall_contacts_per_step": float(np.mean(wall_contact_counts)),
                    # MuJoCo exposes the contact force at the final internal frame
                    # of each 0.05-s control step, so this is a force-time proxy
                    # rather than an exact integral over all five internal frames.
                    "wall_contact_impulse_proxy": float(
                        np.sum(wall_contact_forces) * 0.05
                    ),
                    "waypoint_switches": waypoint_switches,
                    "waypoints_reached": waypoints_reached,
                    "mean_action_cost": float(np.mean(action_costs)),
                    "mean_upright_factor": float(np.mean(upright)),
                    "min_upright_factor": float(np.min(upright)),
                    "mean_torso_tilt_degrees": float(np.mean(tilts)),
                    "max_torso_tilt_degrees": float(np.max(tilts)),
                    "inverted_step_fraction": inverted_steps / length,
                    "persistent_inversion_step_fraction": (
                        persistent_inversion_steps / length
                    ),
                    "max_torso_height": float(np.max(heights)),
                    "min_torso_height": float(np.min(heights)),
                    "native_unhealthy_step_fraction": (native_unhealthy_steps / length),
                    "functional_failure_step_fraction": (
                        functional_failure_steps / length
                    ),
                    "low_z_failure_seen": bool(low_z_failure_steps),
                    "high_z_failure_seen": bool(high_z_failure_steps),
                    "nonfinite_failure_seen": bool(nonfinite_failure_steps),
                    "persistent_inversion_seen": bool(persistent_inversion_steps),
                    "high_z_termination": bool(info.get("high_z_termination", False)),
                    "low_z_termination": bool(info.get("low_z_termination", False)),
                    "nonfinite_termination": bool(
                        info.get("nonfinite_termination", False)
                    ),
                    "inverted_completion": inverted_completion,
                    "physical_failure": physical_failure,
                    "functional_failure": functional_failure,
                }
            )
    finally:
        env.close()
    return records


def summarize_antmaze(records: list[dict]) -> dict:
    result = {"episodes": len(records)}
    for key in (
        "official_return",
        "success",
        "terminal_step_success",
        "goals_reached",
        "successful_steps",
        "success_step_fraction",
        "length",
        "initial_direct_distance",
        "final_goal_distance",
        "path_length",
        "direct_path_efficiency",
        "success_weighted_path_efficiency",
        "mean_speed_xy",
        "mean_goal_velocity",
        "mean_lateral_speed_to_goal",
        "backward_from_goal_fraction",
        "positive_goal_progress_fraction",
        "wall_contact_step_fraction",
        "wall_contacts_per_step",
        "wall_contact_impulse_proxy",
        "waypoint_switches",
        "waypoints_reached",
        "mean_action_cost",
        "mean_upright_factor",
        "min_upright_factor",
        "mean_torso_tilt_degrees",
        "max_torso_tilt_degrees",
        "inverted_step_fraction",
        "persistent_inversion_step_fraction",
        "max_torso_height",
        "min_torso_height",
        "native_unhealthy_step_fraction",
        "functional_failure_step_fraction",
        "low_z_failure_seen",
        "high_z_failure_seen",
        "nonfinite_failure_seen",
        "persistent_inversion_seen",
        "high_z_termination",
        "low_z_termination",
        "nonfinite_termination",
        "inverted_completion",
        "functional_failure",
    ):
        values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    for key in (
        "shortest_path_length",
        "shortest_path_cell_steps",
        "shortest_path_turns",
        "success_weighted_geodesic_efficiency",
        "tilt_guard_step_fraction",
    ):
        if all(key in row for row in records):
            values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
            result[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
    physical_failures = np.asarray(
        [float(antmaze_physical_failure(row)) for row in records], dtype=np.float64
    )
    result["physical_failure"] = {
        "mean": float(physical_failures.mean()),
        "std": float(physical_failures.std()),
        "median": float(np.median(physical_failures)),
        "q05": float(np.quantile(physical_failures, 0.05)),
        "q95": float(np.quantile(physical_failures, 0.95)),
    }
    successful_times = np.asarray(
        [
            row["time_to_goal_seconds"]
            for row in records
            if row["time_to_goal_seconds"] is not None
        ],
        dtype=np.float64,
    )
    result["time_to_goal_seconds_successes"] = {
        "n": int(successful_times.size),
        "mean": float(successful_times.mean()) if successful_times.size else None,
        "median": float(np.median(successful_times)) if successful_times.size else None,
    }
    if all("time_to_goal_per_geodesic_meter" in row for row in records):
        normalized_successful_times = np.asarray(
            [
                row["time_to_goal_per_geodesic_meter"]
                for row in records
                if row["time_to_goal_per_geodesic_meter"] is not None
            ],
            dtype=np.float64,
        )
        result["time_to_goal_per_geodesic_meter_successes"] = {
            "n": int(normalized_successful_times.size),
            "mean": (
                float(normalized_successful_times.mean())
                if normalized_successful_times.size
                else None
            ),
            "median": (
                float(np.median(normalized_successful_times))
                if normalized_successful_times.size
                else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint", choices=["latest", "best", "initial"], default="best"
    )
    parser.add_argument("--goal-adapter-scale", type=float)
    parser.add_argument("--goal-adapter-motor-scale", type=float)
    parser.add_argument("--connectome-output-scale", type=float)
    parser.add_argument("--action-smoothing", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--goal-router-smoothing", type=float)
    parser.add_argument("--tilt-guard-threshold-degrees", type=float)
    parser.add_argument("--tilt-guard-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--waypoint-planner",
        choices=["none", "official_grid_waypoints"],
        help=(
            "Override routing for diagnostic evaluation; formal learned-"
            "router runs use 'none'."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = evaluate_antmaze_run(
        args.run,
        args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        goal_adapter_scale=args.goal_adapter_scale,
        goal_adapter_motor_scale=args.goal_adapter_motor_scale,
        connectome_output_scale=args.connectome_output_scale,
        waypoint_planner=args.waypoint_planner,
        goal_router_smoothing=args.goal_router_smoothing,
        action_scale=args.action_scale,
        tilt_guard_threshold_degrees=args.tilt_guard_threshold_degrees,
        tilt_guard_action_scale=args.tilt_guard_action_scale,
        action_smoothing=args.action_smoothing,
    )
    output = args.output or args.run / f"eval_{args.checkpoint}_antmaze.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2))
    os.replace(temporary, output)
    print(json.dumps(summarize_antmaze(records), indent=2))


if __name__ == "__main__":
    main()

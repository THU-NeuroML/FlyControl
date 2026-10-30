"""Comprehensive deterministic motion-quality evaluation for Ant-v5.

The native Ant reward is deliberately not treated as a sufficient locomotion
metric.  This evaluator records progress, survival, posture, smoothness,
mechanical work, contacts, and gait timing directly from MuJoCo state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import torch
import yaml

from .env import ContinualAntEnv
from .evaluate import _normalize_observation, enable_deterministic_evaluation
from .train import build_policy_from_config


FOOT_GEOM_NAMES = {
    "front_left": "left_ankle_geom",
    "front_right": "right_ankle_geom",
    "back_left": "third_ankle_geom",
    "back_right": "fourth_ankle_geom",
}
FOOT_ORDER = tuple(FOOT_GEOM_NAMES)
MAX_EPISODE_STEPS = 1_000
GRAVITY = 9.81


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


def _rms(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def quaternion_posture(quaternion: np.ndarray) -> tuple[float, float, float, float]:
    """Return roll, pitch, yaw, and gravity-axis tilt in radians.

    MuJoCo free-joint quaternions use ``w, x, y, z`` ordering.
    """
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    # World z component of the torso-local z axis.
    local_z_dot_world_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = math.acos(float(np.clip(local_z_dot_world_z, -1.0, 1.0)))
    return roll, pitch, yaw, tilt


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a 2-D convex hull without adding a scipy dependency."""
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (
            a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def support_polygon_margin(point: np.ndarray, supports: np.ndarray) -> float:
    """Signed distance from a point to a convex support-polygon boundary.

    Positive values are inside.  Fewer than three non-collinear support points
    do not define a static polygon and return NaN.
    """
    hull = _convex_hull(supports)
    if len(hull) < 3:
        return float("nan")
    point = np.asarray(point, dtype=np.float64)
    distances = []
    signs = []
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        offset = point - start
        signed = float((edge[0] * offset[1] - edge[1] * offset[0]) / length)
        signs.append(signed)
        projection = float(np.clip(np.dot(point - start, edge) / (length ** 2), 0, 1))
        closest = start + projection * edge
        distances.append(float(np.linalg.norm(point - closest)))
    if not distances:
        return float("nan")
    inside = all(value >= -1e-12 for value in signs)
    return min(distances) if inside else -min(distances)


def phase_locking(touchdowns_a: np.ndarray, touchdowns_b: np.ndarray) -> dict:
    """Estimate relative touchdown phase and its circular concentration."""
    a = np.asarray(touchdowns_a, dtype=np.float64)
    b = np.asarray(touchdowns_b, dtype=np.float64)
    if len(a) < 3 or len(b) < 2:
        return {"phase": None, "strength": None, "samples": 0}
    phases = []
    for start, end in zip(a[:-1], a[1:]):
        period = end - start
        if period <= 0:
            continue
        candidates = b[(b >= start) & (b < end)]
        if candidates.size:
            phases.append(float((candidates[0] - start) / period))
    if not phases:
        return {"phase": None, "strength": None, "samples": 0}
    angles = 2 * np.pi * np.asarray(phases)
    vector = np.mean(np.exp(1j * angles))
    return {
        "phase": float((np.angle(vector) / (2 * np.pi)) % 1.0),
        "strength": float(abs(vector)),
        "samples": len(phases),
    }


def _contact_state(unwrapped, floor_id: int, foot_ids: dict[str, int]) -> tuple[
        dict[str, bool], dict[str, float], int, bool]:
    contacts = {name: False for name in FOOT_ORDER}
    forces = {name: 0.0 for name in FOOT_ORDER}
    nonfoot_floor_contacts = 0
    torso_floor_contact = False
    reverse = {geom_id: name for name, geom_id in foot_ids.items()}
    torso_geom = mujoco.mj_name2id(
        unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
    for index in range(unwrapped.data.ncon):
        contact = unwrapped.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair:
            continue
        other = int(contact.geom2 if contact.geom1 == floor_id else contact.geom1)
        if other in reverse:
            name = reverse[other]
            contacts[name] = True
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(unwrapped.model, unwrapped.data, index, force)
            forces[name] += abs(float(force[0]))
        else:
            nonfoot_floor_contacts += 1
            torso_floor_contact |= other == torso_geom
    return contacts, forces, nonfoot_floor_contacts, torso_floor_contact


@dataclass
class MotionAccumulator:
    dt: float
    mass: float
    initial_position: np.ndarray
    initial_yaw: float
    previous_position: np.ndarray
    previous_joint_velocity: np.ndarray
    previous_action: np.ndarray | None
    previous_foot_positions: dict[str, np.ndarray]
    positions: list[np.ndarray] = field(default_factory=list)
    velocities_x: list[float] = field(default_factory=list)
    velocities_y: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    rolls: list[float] = field(default_factory=list)
    pitches: list[float] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)
    tilts: list[float] = field(default_factory=list)
    angular_xy: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    action_rates: list[np.ndarray] = field(default_factory=list)
    joint_accelerations: list[np.ndarray] = field(default_factory=list)
    torques: list[np.ndarray] = field(default_factory=list)
    joint_velocities: list[np.ndarray] = field(default_factory=list)
    absolute_powers: list[float] = field(default_factory=list)
    positive_powers: list[float] = field(default_factory=list)
    reward_forward: list[float] = field(default_factory=list)
    reward_survive: list[float] = field(default_factory=list)
    reward_ctrl: list[float] = field(default_factory=list)
    reward_contact: list[float] = field(default_factory=list)
    contact_rows: list[dict[str, bool]] = field(default_factory=list)
    contact_force_rows: list[dict[str, float]] = field(default_factory=list)
    foot_slip_speeds: list[float] = field(default_factory=list)
    support_counts: list[int] = field(default_factory=list)
    support_margins: list[float] = field(default_factory=list)
    nonfoot_contact_frames: int = 0
    torso_contact_frames: int = 0
    touchdown_steps: dict[str, list[int]] = field(
        default_factory=lambda: {name: [] for name in FOOT_ORDER})
    previous_contacts: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in FOOT_ORDER})

    def add_step(
        self, *, unwrapped, info: dict, action: np.ndarray,
        foot_ids: dict[str, int], contacts: dict[str, bool],
        contact_forces: dict[str, float], nonfoot_contacts: int,
        torso_contact: bool,
    ) -> None:
        data = unwrapped.data
        position = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        roll, pitch, yaw, tilt = quaternion_posture(data.qpos[3:7])
        joint_velocity = np.asarray(data.qvel[6:14], dtype=np.float64).copy()
        torque = np.asarray(data.qfrc_actuator[6:14], dtype=np.float64).copy()
        action = np.asarray(action, dtype=np.float64).copy()
        power = torque * joint_velocity

        self.positions.append(position)
        self.velocities_x.append(float(info.get("x_velocity", data.qvel[0])))
        self.velocities_y.append(float(info.get("y_velocity", data.qvel[1])))
        self.heights.append(float(position[2]))
        self.rolls.append(roll)
        self.pitches.append(pitch)
        self.yaws.append(yaw)
        self.tilts.append(tilt)
        self.angular_xy.append(float(np.linalg.norm(data.qvel[3:5])))
        self.actions.append(action)
        if self.previous_action is not None:
            self.action_rates.append((action - self.previous_action) / self.dt)
        self.joint_accelerations.append(
            (joint_velocity - self.previous_joint_velocity) / self.dt)
        self.torques.append(torque)
        self.joint_velocities.append(joint_velocity)
        self.absolute_powers.append(float(np.abs(power).sum()))
        self.positive_powers.append(float(np.maximum(power, 0).sum()))
        self.reward_forward.append(float(info.get("reward_forward", np.nan)))
        self.reward_survive.append(float(info.get("reward_survive", np.nan)))
        self.reward_ctrl.append(float(info.get("reward_ctrl", np.nan)))
        self.reward_contact.append(float(info.get("reward_contact", np.nan)))
        self.contact_rows.append(dict(contacts))
        self.contact_force_rows.append(dict(contact_forces))
        support_names = [name for name in FOOT_ORDER if contacts[name]]
        self.support_counts.append(len(support_names))
        if len(support_names) >= 3:
            supports = np.asarray([
                unwrapped.data.geom_xpos[foot_ids[name], :2]
                for name in support_names])
            self.support_margins.append(
                support_polygon_margin(position[:2], supports))
        for name in FOOT_ORDER:
            foot_position = np.asarray(
                unwrapped.data.geom_xpos[foot_ids[name]],
                dtype=np.float64).copy()
            if contacts[name] and self.previous_contacts[name]:
                delta = foot_position[:2] - self.previous_foot_positions[name][:2]
                self.foot_slip_speeds.append(float(np.linalg.norm(delta) / self.dt))
            if contacts[name] and not self.previous_contacts[name]:
                self.touchdown_steps[name].append(len(self.positions) - 1)
            self.previous_foot_positions[name] = foot_position
        self.nonfoot_contact_frames += int(nonfoot_contacts > 0)
        self.torso_contact_frames += int(torso_contact)
        self.previous_contacts = dict(contacts)
        self.previous_position = position
        self.previous_joint_velocity = joint_velocity
        self.previous_action = action

    def finish(self, *, episode_return: float, terminated: bool) -> dict:
        length = len(self.positions)
        positions = np.asarray(self.positions, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        torques = np.asarray(self.torques, dtype=np.float64)
        elapsed = length * self.dt
        final_position = positions[-1] if length else self.initial_position
        position_rows = np.vstack([self.initial_position, positions])
        deltas = np.diff(position_rows[:, :2], axis=0)
        path_length = float(np.linalg.norm(deltas, axis=1).sum())
        progress = float(final_position[0] - self.initial_position[0])
        lateral = float(final_position[1] - self.initial_position[1])
        yaw_unwrapped = np.unwrap(np.asarray(self.yaws, dtype=np.float64))
        yaw_drift = (
            float(yaw_unwrapped[-1] - self.initial_yaw)
            if yaw_unwrapped.size else float("nan"))
        absolute_work = float(np.sum(self.absolute_powers) * self.dt)
        positive_work = float(np.sum(self.positive_powers) * self.dt)
        cot_denominator = self.mass * GRAVITY * progress
        mechanical_cot = (
            absolute_work / cot_denominator if progress > 0.1
            else float("nan"))

        contact_array = {
            name: np.asarray([row[name] for row in self.contact_rows], dtype=bool)
            for name in FOOT_ORDER}
        duty_factors = {
            name: float(values.mean()) if values.size else float("nan")
            for name, values in contact_array.items()}
        touchdown_counts = {
            name: len(self.touchdown_steps[name]) for name in FOOT_ORDER}
        step_frequencies = {
            name: touchdown_counts[name] / elapsed if elapsed > 0 else float("nan")
            for name in FOOT_ORDER}
        stride_cvs = []
        for name in FOOT_ORDER:
            intervals = np.diff(self.touchdown_steps[name]) * self.dt
            if intervals.size >= 2 and float(intervals.mean()) > 0:
                stride_cvs.append(float(intervals.std() / intervals.mean()))
        diagonal_locking = []
        ipsilateral_locking = []
        for a, b in (("front_left", "back_right"),
                     ("front_right", "back_left")):
            diagonal_locking.append(phase_locking(
                np.asarray(self.touchdown_steps[a]),
                np.asarray(self.touchdown_steps[b])))
        for a, b in (("front_left", "back_left"),
                     ("front_right", "back_right")):
            ipsilateral_locking.append(phase_locking(
                np.asarray(self.touchdown_steps[a]),
                np.asarray(self.touchdown_steps[b])))

        support_margins = np.asarray(self.support_margins, dtype=np.float64)
        contact_forces = np.asarray([
            sum(row.values()) for row in self.contact_force_rows], dtype=np.float64)
        body_weight = self.mass * GRAVITY
        duty_values = np.asarray(list(duty_factors.values()), dtype=np.float64)
        frequency_values = np.asarray(list(step_frequencies.values()), dtype=np.float64)
        velocity_x = np.asarray(self.velocities_x, dtype=np.float64)
        velocity_y = np.asarray(self.velocities_y, dtype=np.float64)
        tilts = np.asarray(self.tilts, dtype=np.float64)

        result = {
            "return": float(episode_return),
            "length_steps": length,
            "duration_seconds": elapsed,
            "fall": bool(terminated),
            "survival_fraction": length / MAX_EPISODE_STEPS,
            "forward_progress_m": progress,
            "mean_forward_velocity_mps": _mean(velocity_x),
            "forward_velocity_std_mps": float(velocity_x.std()),
            "forward_velocity_p05_mps": float(np.quantile(velocity_x, 0.05)),
            "forward_velocity_p95_mps": float(np.quantile(velocity_x, 0.95)),
            "backward_step_fraction": float(np.mean(velocity_x < 0)),
            "mean_abs_lateral_velocity_mps": _mean(np.abs(velocity_y)),
            "lateral_drift_m": abs(lateral),
            "lateral_drift_per_forward_m": (
                abs(lateral) / progress if progress > 0.1 else None),
            "planar_path_length_m": path_length,
            "path_efficiency": progress / path_length if path_length > 0 else None,
            "mean_torso_height_m": _mean(self.heights),
            "torso_height_std_m": float(np.std(self.heights)),
            "minimum_torso_height_m": float(np.min(self.heights)),
            "final_torso_height_m": float(self.heights[-1]),
            "rms_roll_deg": math.degrees(_rms(self.rolls)),
            "rms_pitch_deg": math.degrees(_rms(self.pitches)),
            "mean_tilt_deg": math.degrees(_mean(tilts)),
            "maximum_tilt_deg": math.degrees(float(np.max(tilts))),
            "final_tilt_deg": math.degrees(float(tilts[-1])),
            "high_tilt_fraction_30deg": float(np.mean(tilts > np.deg2rad(30))),
            "rms_roll_pitch_angular_velocity_radps": _rms(self.angular_xy),
            "absolute_yaw_drift_deg": abs(math.degrees(yaw_drift)),
            "rms_action": _rms(actions),
            "mean_abs_action": _mean(np.abs(actions)),
            "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.95)),
            "rms_action_rate_per_s": _rms(self.action_rates),
            "rms_joint_acceleration_radps2": _rms(self.joint_accelerations),
            "rms_joint_torque_nm": _rms(torques),
            "absolute_mechanical_work_j": absolute_work,
            "positive_mechanical_work_j": positive_work,
            "absolute_mechanical_power_w": _mean(self.absolute_powers),
            "mechanical_cost_of_transport": _finite_or_none(mechanical_cot),
            "mechanical_cost_per_planar_meter": (
                absolute_work / (self.mass * GRAVITY * path_length)
                if path_length > 0.1 else None),
            "robot_mass_kg": self.mass,
            "mean_reward_forward": _mean(self.reward_forward),
            "mean_reward_survive": _mean(self.reward_survive),
            "mean_reward_ctrl": _mean(self.reward_ctrl),
            "mean_reward_contact": _mean(self.reward_contact),
            "mean_support_feet": _mean(self.support_counts),
            "flight_fraction": float(np.mean(np.asarray(self.support_counts) == 0)),
            "four_foot_contact_fraction": float(
                np.mean(np.asarray(self.support_counts) == 4)),
            "nonfoot_ground_contact_fraction": self.nonfoot_contact_frames / length,
            "torso_ground_contact_fraction": self.torso_contact_frames / length,
            "support_polygon_available_fraction": len(support_margins) / length,
            "mean_static_support_margin_m": (
                _finite_or_none(_mean(support_margins)) if support_margins.size
                else None),
            "outside_static_support_fraction": (
                float(np.mean(support_margins < 0)) if support_margins.size
                else None),
            "mean_stance_foot_slip_mps": _finite_or_none(
                _mean(self.foot_slip_speeds)),
            "mean_contact_force_bodyweights": _mean(contact_forces) / body_weight,
            "peak_contact_force_bodyweights": float(np.max(contact_forces)) / body_weight,
            "mean_duty_factor": _mean(duty_values),
            "duty_factor_range": float(np.max(duty_values) - np.min(duty_values)),
            "left_right_duty_imbalance": abs(
                (duty_factors["front_left"] + duty_factors["back_left"]) / 2
                - (duty_factors["front_right"] + duty_factors["back_right"]) / 2),
            "diagonal_duty_imbalance": abs(
                (duty_factors["front_left"] + duty_factors["back_right"]) / 2
                - (duty_factors["front_right"] + duty_factors["back_left"]) / 2),
            "mean_step_frequency_hz": _mean(frequency_values),
            "step_frequency_range_hz": float(
                np.max(frequency_values) - np.min(frequency_values)),
            "mean_stride_interval_cv": (
                _mean(stride_cvs) if stride_cvs else None),
            "mean_diagonal_phase_locking": _mean([
                row["strength"] for row in diagonal_locking
                if row["strength"] is not None]),
            "mean_ipsilateral_phase_locking": _mean([
                row["strength"] for row in ipsilateral_locking
                if row["strength"] is not None]),
            "foot_duty_factors": duty_factors,
            "foot_step_frequencies_hz": step_frequencies,
            "foot_touchdown_counts": touchdown_counts,
            "diagonal_phase": diagonal_locking,
            "ipsilateral_phase": ipsilateral_locking,
        }
        return {
            key: _finite_or_none(value) if isinstance(value, float) else value
            for key, value in result.items()}


def load_evaluation_policy(
    run_dir: Path, *, device: str, checkpoint: str = "best",
):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    paths = {
        "best": run_dir / "best_checkpoint.pt",
        "latest": run_dir / "checkpoint.pt",
        "initial": run_dir / "initial_checkpoint.pt",
    }
    checkpoint_path = paths[checkpoint]
    checkpoint_data = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    training_env = config.get("resolved_env", config.get("env", {}))
    append_goal_features = bool(
        training_env.get("append_zero_goal_features", False))
    probe = ContinualAntEnv(
        friction_values=[1.0], steps_per_context=10**12,
        append_zero_goal_features=append_goal_features)
    try:
        policy = build_policy_from_config(
            config, probe.observation_space.shape[0], probe.action_space.shape[0])
    finally:
        probe.close()
    policy.load_state_dict(checkpoint_data["policy_state"])
    policy.to(device).eval()
    return policy, checkpoint_data, config, checkpoint_path


def evaluate_motion_quality(
    run_dir: Path,
    *,
    friction: float,
    episodes: int,
    seed: int,
    device: str,
    checkpoint: str = "best",
    action_scale: float | None = None,
    render_mode: str | None = None,
    frame_callback: Callable | None = None,
) -> list[dict]:
    enable_deterministic_evaluation()
    policy, checkpoint_data, config, checkpoint_path = load_evaluation_policy(
        run_dir, device=device, checkpoint=checkpoint)
    training_env = config.get("resolved_env", config.get("env", {}))
    append_goal_features = bool(
        training_env.get("append_zero_goal_features", False))
    training_stabilization = training_env.get("stabilization") or {}
    selected_action_scale = float(
        training_stabilization.get("action_scale", 1.0)
        if action_scale is None else action_scale)
    env = ContinualAntEnv(
        friction_values=[friction], steps_per_context=10**12,
        append_zero_goal_features=append_goal_features,
        stabilization={"enabled": False, "action_scale": selected_action_scale},
        render_mode=render_mode,
        **({"width": 640, "height": 480} if render_mode else {}),
    )
    unwrapped = env.base_env.unwrapped
    model = unwrapped.model
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for name, geom_name in FOOT_GEOM_NAMES.items()}
    if floor_id < 0 or any(value < 0 for value in foot_ids.values()):
        env.close()
        raise RuntimeError("Ant model is missing required floor/foot geoms")
    mass = float(model.body_subtreemass[1])
    records = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(seed=episode_seed)
            data = unwrapped.data
            _, _, initial_yaw, _ = quaternion_posture(data.qpos[3:7])
            foot_positions = {
                name: np.asarray(data.geom_xpos[geom_id]).copy()
                for name, geom_id in foot_ids.items()}
            accumulator = MotionAccumulator(
                dt=float(unwrapped.dt),
                mass=mass,
                initial_position=np.asarray(data.qpos[:3]).copy(),
                initial_yaw=initial_yaw,
                previous_position=np.asarray(data.qpos[:3]).copy(),
                previous_joint_velocity=np.asarray(data.qvel[6:14]).copy(),
                previous_action=None,
                previous_foot_positions=foot_positions,
            )
            hidden = torch.zeros(
                1, policy.state_dim, dtype=torch.float32, device=device)
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(
                    _normalize_observation(
                        observation, checkpoint_data, config),
                    dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, hidden = policy.act(
                        obs_tensor, hidden, deterministic=True)
                action_numpy = action.squeeze(0).cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(
                    action_numpy)
                contacts, contact_forces, nonfoot, torso = _contact_state(
                    unwrapped, floor_id, foot_ids)
                accumulator.add_step(
                    unwrapped=unwrapped, info=info,
                    action=action_numpy * selected_action_scale,
                    foot_ids=foot_ids, contacts=contacts,
                    contact_forces=contact_forces,
                    nonfoot_contacts=nonfoot, torso_contact=torso,
                )
                episode_return += float(reward)
                if frame_callback is not None:
                    frame_callback(
                        env=env, episode=episode, seed=episode_seed,
                        metrics=accumulator, terminated=terminated,
                        truncated=truncated)
            row = accumulator.finish(
                episode_return=episode_return, terminated=terminated)
            row.update({
                "checkpoint": checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "friction": float(friction),
                "episode": episode,
                "seed": episode_seed,
                "action_scale": selected_action_scale,
            })
            records.append(row)
    finally:
        env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint", choices=["best", "latest", "initial"],
        default="best")
    parser.add_argument(
        "--action-scale", type=float,
        help="Controller output scale; defaults to the run configuration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = evaluate_motion_quality(
        args.run, friction=args.friction, episodes=args.episodes,
        seed=args.seed, device=args.device, checkpoint=args.checkpoint,
        action_scale=args.action_scale)
    output = args.output or args.run / "motion_quality_best_native.json"
    _write_json(output, records)
    summary_fields = [
        "return", "mean_forward_velocity_mps", "fall",
        "survival_fraction", "path_efficiency", "mean_tilt_deg",
        "rms_action_rate_per_s", "mechanical_cost_of_transport",
        "mean_stance_foot_slip_mps", "mean_contact_force_bodyweights",
        "mean_duty_factor", "mean_step_frequency_hz",
    ]
    summary = {
        field: _mean([row[field] for row in records if row[field] is not None])
        for field in summary_fields}
    print(json.dumps({"episodes": len(records), **summary}, indent=2))


if __name__ == "__main__":
    main()

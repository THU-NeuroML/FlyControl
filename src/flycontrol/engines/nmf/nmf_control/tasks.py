"""Task target definitions for command-level locomotion experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TASK_OBS_DIM = 5


def _wrap_pi(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def steering_reward(
    task: str,
    forward_velocity: float,
    heading_rate_rad_s: float,
    target_turn_sign: float | None = None,
    turn_weight: float = 20.0,
    mode: str = "sign",
    steering_rate_max_rad_s: float = 2.5,
) -> float:
    if task == "steering_left":
        target_sign = 1.0
        target_command = target_sign
    elif task == "steering_right":
        target_sign = -1.0
        target_command = target_sign
    elif task == "steering_target":
        if target_turn_sign is None:
            raise ValueError("steering_target requires target_turn_sign")
        target_command = float(np.clip(target_turn_sign, -1.0, 1.0))
        target_sign = float(np.sign(target_command))
        if target_sign == 0.0:
            raise ValueError("target_turn_sign must be non-zero")
    else:
        raise ValueError(
            "steering_reward requires steering_left, steering_right, or steering_target"
        )
    if mode == "rate":
        target_rate = target_command * float(steering_rate_max_rad_s)
        rate_error = abs(float(heading_rate_rad_s) - target_rate)
        return float(
            forward_velocity
            + turn_weight * (float(steering_rate_max_rad_s) - rate_error)
        )
    if mode != "sign":
        raise ValueError("steering reward mode must be 'sign' or 'rate'")
    return float(forward_velocity + turn_weight * target_sign * heading_rate_rad_s)


def steering_target_observation(target_turn_sign: float) -> tuple[np.ndarray, dict]:
    target_command = float(np.clip(target_turn_sign, -1.0, 1.0))
    target_sign = float(np.sign(target_command))
    if target_sign == 0.0:
        raise ValueError("target_turn_sign must be non-zero")
    vector = np.array([target_command, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return vector, {
        "target_turn_sign": target_sign,
        "target_turn_strength": abs(target_command),
        "target_turn_command": target_command,
    }


def heading_target_observation(
    current_heading_rad: float,
    target_heading_rad: float,
    target_angle_rad: float,
) -> tuple[np.ndarray, dict]:
    heading_error = _wrap_pi(float(target_heading_rad) - float(current_heading_rad))
    target_angle = _wrap_pi(float(target_angle_rad))
    vector = np.array(
        [
            np.sin(heading_error),
            np.cos(heading_error),
            np.clip(target_angle / np.pi, -1.0, 1.0),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return vector, {
        "target_angle_rad": target_angle,
        "target_heading_rad": _wrap_pi(float(target_heading_rad)),
        "heading_error_rad": heading_error,
        "heading_error_abs_rad": abs(heading_error),
    }


def heading_target_reward(
    forward_velocity: float,
    heading_error_rad: float,
    forward_weight: float = 1.0,
    heading_error_weight: float = 20.0,
) -> float:
    return float(
        forward_weight * float(forward_velocity)
        - heading_error_weight * abs(float(heading_error_rad))
    )


def heading_speed_target_observation(
    current_heading_rad: float,
    target_heading_rad: float,
    target_speed_mm_s: float,
    current_speed_mm_s: float,
    speed_scale_mm_s: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Joint speed/heading target for bounded-drive recovery experiments."""
    target_speed = float(target_speed_mm_s)
    speed_scale = float(speed_scale_mm_s)
    if target_speed <= 0.0:
        raise ValueError("target_speed_mm_s must be positive")
    if speed_scale <= 0.0:
        raise ValueError("speed_scale_mm_s must be positive")
    current_speed = float(current_speed_mm_s)
    speed_error = target_speed - current_speed
    heading_error = _wrap_pi(
        float(target_heading_rad) - float(current_heading_rad)
    )
    vector = np.asarray(
        [
            np.clip(target_speed / speed_scale, 0.0, 1.0),
            np.clip(current_speed / speed_scale, -1.0, 1.0),
            np.clip(speed_error / speed_scale, -1.0, 1.0),
            np.sin(heading_error),
            np.cos(heading_error),
        ],
        dtype=np.float32,
    )
    return vector, {
        "target_speed_mm_s": target_speed,
        "current_speed_mm_s": current_speed,
        "speed_error_mm_s": speed_error,
        "target_angle_rad": 0.0,
        "target_heading_rad": _wrap_pi(float(target_heading_rad)),
        "heading_error_rad": heading_error,
        "heading_error_abs_rad": abs(heading_error),
    }


def speed_target_observation(
    target_speed_mm_s: float,
    speed_scale_mm_s: float = 20.0,
    current_speed_mm_s: float = 0.0,
    heading_rate_rad_s: float = 0.0,
    heading_rate_scale_rad_s: float = 5.0,
) -> tuple[np.ndarray, dict]:
    target_speed = float(target_speed_mm_s)
    if target_speed <= 0.0:
        raise ValueError("target_speed_mm_s must be positive")
    speed_scale = float(speed_scale_mm_s)
    if speed_scale <= 0.0:
        raise ValueError("speed_scale_mm_s must be positive")
    current_speed = float(current_speed_mm_s)
    speed_error = target_speed - current_speed
    heading_rate_scale = float(heading_rate_scale_rad_s)
    if heading_rate_scale <= 0.0:
        raise ValueError("heading_rate_scale_rad_s must be positive")
    vector = np.array(
        [
            np.clip(target_speed / speed_scale, 0.0, 1.0),
            np.clip(current_speed / speed_scale, -1.0, 1.0),
            np.clip(speed_error / speed_scale, -1.0, 1.0),
            np.clip(float(heading_rate_rad_s) / heading_rate_scale, -1.0, 1.0),
            0.0,
        ],
        dtype=np.float32,
    )
    return vector, {
        "target_speed_mm_s": target_speed,
        "current_speed_mm_s": current_speed,
        "speed_error_mm_s": speed_error,
    }


def speed_control_reward(
    forward_velocity: float,
    heading_rate_rad_s: float,
    target_speed_mm_s: float,
    speed_error_weight: float = 2.0,
    heading_rate_weight: float = 0.25,
    huber_delta_mm_s: float = 1.0,
) -> float:
    speed_error = abs(float(forward_velocity) - float(target_speed_mm_s))
    delta = float(huber_delta_mm_s)
    if delta <= 0.0:
        raise ValueError("huber_delta_mm_s must be positive")
    speed_loss = (
        0.5 * speed_error * speed_error / delta
        if speed_error <= delta
        else speed_error - 0.5 * delta
    )
    return float(
        -speed_error_weight * speed_loss
        - heading_rate_weight * abs(float(heading_rate_rad_s))
    )


def heading_speed_target_reward(
    forward_velocity: float,
    heading_rate_rad_s: float,
    heading_error_rad: float,
    target_speed_mm_s: float,
    speed_error_weight: float = 2.0,
    heading_rate_weight: float = 0.25,
    heading_error_weight: float = 20.0,
    huber_delta_mm_s: float = 1.0,
) -> float:
    """Track a finite forward speed while recovering a target heading."""
    if heading_error_weight < 0.0:
        raise ValueError("heading_error_weight must be non-negative")
    speed_term = speed_control_reward(
        forward_velocity,
        heading_rate_rad_s,
        target_speed_mm_s,
        speed_error_weight=speed_error_weight,
        heading_rate_weight=heading_rate_weight,
        huber_delta_mm_s=huber_delta_mm_s,
    )
    return float(
        speed_term - heading_error_weight * abs(float(heading_error_rad))
    )


def stumble_recovery_reward(
    forward_velocity: float,
    heading_rate_rad_s: float,
    perturbation_active: bool,
    local_action_l2: float = 0.0,
    heading_rate_weight: float = 0.5,
    action_weight: float = 0.02,
) -> float:
    """Dense reward for localized leg-perturbation recovery.

    The reward never exposes the disturbed leg identity.  The controller must
    infer it from per-leg position/contact observations.  A small residual
    penalty keeps the neutral CPG solution preferred outside the perturbation.
    """
    recovery_weight = 1.0 if perturbation_active else 0.25
    return float(
        forward_velocity
        - recovery_weight * heading_rate_weight * abs(heading_rate_rad_s)
        - action_weight * float(local_action_l2)
    )


def path_following_reward(
    forward_velocity: float,
    path_error_mm: float,
    heading_error_rad: float,
    path_error_weight: float = 1.0,
    heading_error_weight: float = 5.0,
) -> float:
    return float(
        forward_velocity
        - path_error_weight * abs(path_error_mm)
        - heading_error_weight * abs(heading_error_rad)
    )


@dataclass(frozen=True)
class SinePathTarget:
    amplitude_mm: float = 5.0
    wavelength_mm: float = 40.0
    error_scale_mm: float = 20.0

    def observe(self, raw_obs) -> tuple[np.ndarray, dict]:
        xy = np.asarray(raw_obs["fly"][0, :2], dtype=np.float64)
        orientation = np.asarray(raw_obs["fly_orientation"][:2], dtype=np.float64)

        phase = 2 * np.pi * xy[0] / self.wavelength_mm
        target_y = self.amplitude_mm * np.sin(phase)
        dy_dx = (
            self.amplitude_mm
            * (2 * np.pi / self.wavelength_mm)
            * np.cos(phase)
        )
        target_heading = np.arctan2(dy_dx, 1.0)

        heading = np.arctan2(orientation[1], orientation[0])
        heading_error = _wrap_pi(float(heading - target_heading))
        path_error = float(xy[1] - target_y)

        vector = np.array(
            [
                np.clip(path_error / self.error_scale_mm, -1.0, 1.0),
                np.sin(heading_error),
                np.cos(heading_error),
                np.sin(phase),
                np.cos(phase),
            ],
            dtype=np.float32,
        )
        info = {
            "path_target_y_mm": float(target_y),
            "path_error_mm": path_error,
            "path_heading_error_rad": heading_error,
            "path_phase_rad": float(phase),
        }
        return vector, info

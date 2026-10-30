"""Original body/goal observation encoding."""

import numpy as np


def encode_body_goal(
    proprioception: np.ndarray,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    torso_yaw: np.ndarray,
    goal_feature_mode: str = "body_direction",
) -> np.ndarray:
    """Rebuild the policy observation for a real or hindsight goal."""
    proprioception = np.asarray(proprioception, dtype=np.float32)
    achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
    desired_goal = np.asarray(desired_goal, dtype=np.float32)
    torso_yaw = np.asarray(torso_yaw, dtype=np.float32)
    delta = desired_goal - achieved_goal
    if goal_feature_mode == "world":
        goal_features = np.concatenate([achieved_goal, delta], axis=-1)
        return np.concatenate([proprioception, goal_features], axis=-1).astype(
            np.float32, copy=False
        )
    if goal_feature_mode != "body_direction":
        raise ValueError(f"unsupported goal feature mode: {goal_feature_mode!r}")
    cosine, sine = np.cos(torso_yaw), np.sin(torso_yaw)
    body = np.stack(
        [
            cosine * delta[..., 0] + sine * delta[..., 1],
            -sine * delta[..., 0] + cosine * delta[..., 1],
        ],
        axis=-1,
    )
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    unit = np.divide(body, distance, out=np.zeros_like(body), where=distance > 1e-12)
    return np.concatenate([proprioception, body, unit], axis=-1).astype(
        np.float32, copy=False
    )

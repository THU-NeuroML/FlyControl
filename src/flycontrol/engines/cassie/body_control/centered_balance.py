"""Cassie support objective with observable position drift, no assistance."""

import gymnasium as gym
import numpy as np
from .env import TaskEnv


class CenteredCassieEnv(TaskEnv):
    def __init__(
        self,
        body,
        n_envs=4,
        horizon_s=20.0,
        eval_speed=None,
        interface="cassie_centered_curriculum",
    ):
        if body != "cassie":
            raise ValueError("Cassie only")
        # Keep the parent's action-coordinate map and competence schedule intact.
        super().__init__(body, n_envs, horizon_s, eval_speed, "cassie_curriculum")
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.physics.observation_space.shape[0] + 1,),
            dtype=np.float32,
        )

    def observation(self):
        x = float(self.data.qpos[0] - self.physics.origin[0])
        return np.r_[super().observation(), x / 0.3].astype(np.float32)

    def step(self, raw_action):
        standing = self.command < 0.02
        obs, reward, terminated, truncated, info = super().step(raw_action)
        displacement = self.data.qpos[:2] - self.physics.origin
        cost = (
            0.5
            * min(float(np.sum((displacement / 0.3) ** 2)), 4.0)
            * info["control_duration_s"]
            if standing
            else 0.0
        )
        info.update(
            reward_support_position=-cost, body_interface="cassie_centered_curriculum"
        )
        return obs, float(reward - cost), terminated, truncated, info

"""Path-position objective for native joint policies, shared by both methods."""

from .env import TaskEnv


class PathAlignedNativeEnv(TaskEnv):
    def __init__(
        self,
        body,
        n_envs=4,
        horizon_s=20.0,
        eval_speed=None,
        interface="go1_rhythm_path",
    ):
        parent = {"go1_rhythm_path": ("go1", "go1_rhythm")}
        required, base = parent[interface]
        if body != required:
            raise ValueError(f"{interface} requires {required}")
        super().__init__(body, n_envs, horizon_s, eval_speed, base)
        self.interface = interface

    def step(self, raw_action):
        obs, reward, terminated, truncated, info = super().step(raw_action)
        lateral = float(self.data.qpos[1] - self.physics.origin[1])
        cost = 0.75 * min((lateral / 0.75) ** 2, 4.0) * info["control_duration_s"]
        info["reward_path_alignment"] = -cost
        return obs, float(reward - cost), terminated, truncated, info

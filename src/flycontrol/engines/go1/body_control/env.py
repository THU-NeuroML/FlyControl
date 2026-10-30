"""Robot-specific decision timing, physical observations and time-scaled reward."""

from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
from fly_control.environments.joint_targets import JointTargetEnv, SPEEDS

BODY_CONFIG = {
    "go1": dict(
        repeat=1,
        initial_speed=0.1,
        speed=SPEEDS["go1"],
        log_std=-1.5,
        ramp_steps=524288,
    )
}


class TaskEnv(gym.Env):

    def __init__(
        self, body, n_envs=4, horizon_s=20.0, eval_speed=None, interface="go1_rhythm"
    ):
        if body != "go1" or interface != "go1_rhythm":
            raise ValueError("Unsupported body/interface for this experiment")
        super().__init__()
        self.body = body
        self.config = BODY_CONFIG[body].copy()
        self.interface = interface
        self.physics = JointTargetEnv(body, steps=round(horizon_s / 0.02))
        (self.model, self.data) = (self.physics.model, self.physics.data)
        self.repeat = self.config["repeat"]
        self.dt = self.physics.dt * self.repeat
        (self.n_envs, self.eval_speed) = (n_envs, eval_speed)
        self.total_decisions = 0
        self.nu = self.model.nu
        self.physics.span = np.tile([0.2, 0.4, 0.5], 4)
        self.physics.span = np.tile([0.3, 0.7, 0.9], 4)
        self.config.update(repeat=2, log_std=-1.0, clock_hz=2.0)
        self.repeat = self.config["repeat"]
        self.dt = self.physics.dt * self.repeat
        self.clock_phase = 0.0
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, (self.nu,), dtype=np.float32
        )
        self.observation_space = self.physics.observation_space
        if self.config.get("clock_hz"):
            self.observation_space = gym.spaces.Box(
                -np.inf,
                np.inf,
                (self.physics.observation_space.shape[0] + 2,),
                dtype=np.float32,
            )
        self.reference_height = 1.0
        self.command = self.config["initial_speed"]

    def get_state(self):
        return dict(total_decisions=self.total_decisions)

    def set_state(self, state):
        self.total_decisions = int(state["total_decisions"])

    def current_command(self):
        if self.eval_speed is not None:
            return float(self.eval_speed)
        fraction = min(
            1.0, self.total_decisions * self.n_envs / self.config["ramp_steps"]
        )
        return self.config["initial_speed"] + fraction * (
            self.config["speed"] - self.config["initial_speed"]
        )

    def observation(self):
        env = self.physics
        rot = self.data.xmat[env.trunk_id].reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        scale = max(0.3, self.config["speed"])
        observation = np.r_[
            rot.T @ self.data.qvel[:3] / scale,
            self.data.qvel[3:6] / 5.0,
            rot[2],
            self.data.qpos[2] / self.reference_height - 1.0,
            (self.data.qpos[7:] - env.qref) / np.pi,
            self.data.qvel[6:] / 10.0,
            (env.target - env.nominal) / env.span,
            env.previous_action,
            self.command / scale,
            (self.data.qpos[1] - env.origin[1]) / 0.75,
            math.sin(yaw),
            math.cos(yaw),
        ].astype(np.float32)
        if self.config.get("clock_hz"):
            phase = (
                self.clock_phase + 2 * np.pi * self.config["clock_hz"] * self.data.time
            )
            observation = np.r_[observation, np.sin(phase), np.cos(phase)].astype(
                np.float32
            )
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        (_, info) = self.physics.reset(seed=seed)
        self.reference_height = float(self.data.qpos[2])
        if self.config.get("clock_hz"):
            self.clock_phase = float(self.np_random.uniform(0, 2 * np.pi))
        self.command = self.current_command()
        return (self.observation(), info)

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (self.nu,) or not np.isfinite(raw_action).all():
            raise ValueError("Expected a finite Gaussian-coordinate action")
        action = np.tanh(raw_action)
        reward = 0.0
        reward_forward = 0.0
        reward_support = 0.0
        elapsed = 0.0
        start = self.data.qpos[:2].copy()
        for _ in range(self.repeat):
            previous = self.physics.target.copy()
            (_, _, terminated, truncated, info) = self.physics.step(action)
            rot = self.data.xmat[self.physics.trunk_id].reshape(3, 3)
            yaw = math.atan2(rot[1, 0], rot[0, 0])
            upright = np.clip((rot[2, 2] - 0.4) / 0.6, 0, 1)
            support = upright * np.clip(
                (self.data.qpos[2] / self.reference_height - 0.5) / 0.4, 0, 1
            )
            (vx, vy) = (info["x_velocity"], info["y_velocity"])
            if self.command < 0.02:
                forward_rate = -0.5 * min((vx / 0.2) ** 2, 4)
            else:
                forward_rate = 1.5 * float(np.clip(vx / max(self.command, 0.15), -1, 1))
            progress = max(forward_rate, 0) * support + min(forward_rate, 0)
            stable = 0.5 * support
            target_delta = (self.physics.target - previous) / (
                self.physics.rates * self.physics.dt
            )
            costs = (
                0.2 * min((vy / 0.3) ** 2, 4)
                + 0.2 * (1 - math.cos(yaw))
                + 0.05
                * min(((self.data.qpos[1] - self.physics.origin[1]) / 0.75) ** 2, 4)
                + 0.03 * min(float(np.mean(self.data.qvel[3:5] ** 2)), 9)
                + 0.01 * float(np.mean(action**2))
                + 0.01 * float(np.mean(target_delta**2))
            )
            reward += self.physics.dt * (progress + stable - costs)
            reward_forward += self.physics.dt * progress
            reward_support += self.physics.dt * stable
            elapsed += self.physics.dt
            if terminated:
                reward -= 5.0
            if terminated or truncated:
                break
        self.total_decisions += 1
        self.command = self.current_command()
        info.update(
            x_velocity=float((self.data.qpos[0] - start[0]) / elapsed),
            y_velocity=float((self.data.qpos[1] - start[1]) / elapsed),
            reward_forward=reward_forward,
            reward_survive=reward_support,
            teacher_used=False,
            cpg_used=False,
            command_speed_mps=self.command,
            control_duration_s=elapsed,
            raw_gaussian_action=True,
            body_interface=self.interface,
            rhythmic_clock_used=bool(self.config.get("clock_hz")),
            prescribed_joint_gait=False,
        )
        return (
            self.observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        self.physics.close()


def make_env(
    body, n_envs=4, horizon_s=20.0, eval_speed=None, interface="go1_rhythm_path"
):
    if body != "go1" or interface != "go1_rhythm_path":
        raise ValueError("Unsupported body/interface for this experiment")
    from .path_objective import PathAlignedNativeEnv

    return PathAlignedNativeEnv(body, n_envs, horizon_s, eval_speed, interface)

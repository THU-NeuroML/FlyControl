"""Robot-specific decision timing, physical observations and time-scaled reward."""

from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
from fly_control.environments.joint_targets import JointTargetEnv, SPEEDS

BODY_CONFIG = {
    "cassie": dict(
        repeat=1,
        initial_speed=0.0,
        speed=SPEEDS["cassie"],
        log_std=-1.7,
        ramp_steps=524288,
    )
}


class TaskEnv(gym.Env):

    def __init__(
        self,
        body,
        n_envs=4,
        horizon_s=20.0,
        eval_speed=None,
        interface="cassie_curriculum",
    ):
        if body != "cassie" or interface != "cassie_curriculum":
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
        self.physics.span = np.tile([0.25, 0.25, 1.0, 1.6, 0.6], 2)
        self.physics.lower[:] = -np.inf
        self.physics.upper[:] = np.inf
        self.config.update(log_std=-1.0, ramp_steps=1048576)
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
        self.training_command = self.config["initial_speed"]
        self.curriculum_outcomes = []
        self.episode_max_lateral = 0.0

    def get_state(self):
        return dict(
            total_decisions=self.total_decisions,
            training_command=self.training_command,
            curriculum_outcomes=self.curriculum_outcomes.copy(),
        )

    def set_state(self, state):
        self.total_decisions = int(state["total_decisions"])
        self.training_command = float(
            state.get("training_command", self.config["initial_speed"])
        )
        self.curriculum_outcomes = list(state.get("curriculum_outcomes", []))

    def current_command(self):
        if self.eval_speed is not None:
            return float(self.eval_speed)
        return self.training_command
        fraction = min(
            1.0, self.total_decisions * self.n_envs / self.config["ramp_steps"]
        )
        fraction = max(0.0, (fraction - 0.125) / 0.875)
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
        self.episode_max_lateral = 0.0
        if self.config.get("clock_hz"):
            self.clock_phase = float(self.np_random.uniform(0, 2 * np.pi))
        self.command = self.current_command()
        return (self.observation(), info)

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (self.nu,) or not np.isfinite(raw_action).all():
            raise ValueError("Expected a finite Gaussian-coordinate action")
        raw_action = bilateral_coordinates(raw_action)
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
            self.episode_max_lateral = max(
                self.episode_max_lateral,
                abs(float(self.data.qpos[1] - self.physics.origin[1])),
            )
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
            if (
                self.interface in {"cassie_support", "cassie_curriculum"}
                and self.command < 0.02
            ):
                progress = 0.0
                stable = (
                    1.0
                    + upright
                    + float(
                        np.clip(
                            (self.data.qpos[2] / self.reference_height - 0.5) / 0.4,
                            0,
                            1,
                        )
                    )
                )
                costs = (
                    0.05 * min((vx / 0.3) ** 2 + (vy / 0.3) ** 2, 4)
                    + 0.05 * min(float(np.mean(self.data.qvel[3:5] ** 2)), 9)
                    + 0.01 * float(np.mean(action**2))
                    + 0.01 * float(np.mean(target_delta**2))
                )
            else:
                stable = (
                    1.0
                    + upright
                    + float(
                        np.clip(
                            (self.data.qpos[2] / self.reference_height - 0.5) / 0.4,
                            0,
                            1,
                        )
                    )
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
        if (
            (terminated or truncated)
            and self.interface == "cassie_curriculum"
            and (self.eval_speed is None)
        ):
            duration = self.physics.steps * self.physics.dt
            speed = float(
                (self.data.qpos[0] - self.physics.origin[0]) / max(duration, 1e-08)
            )
            success = bool(
                not terminated
                and duration >= 20 - 1e-06
                and (self.episode_max_lateral <= 0.75)
                and (
                    abs(speed) <= 0.05
                    if self.command < 0.02
                    else speed >= 0.7 * self.command
                )
            )
            self.curriculum_outcomes = (self.curriculum_outcomes + [success])[-8:]
            if (
                len(self.curriculum_outcomes) == 8
                and sum(self.curriculum_outcomes) >= 6
            ):
                self.training_command = min(
                    self.config["speed"], round(self.training_command + 0.05, 8)
                )
                self.curriculum_outcomes = []
        if not (self.interface == "cassie_curriculum" and (terminated or truncated)):
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


def bilateral_coordinates(raw_action, differential_scale=0.25):
    """Full-rank common/difference coordinates; no prescribed trajectory.

    Mirrored Cassie joint axes all have opposite signs, verified from the
    native model's left/right foot Jacobians. Every actuator stays controllable.
    This invertible mixing precedes tanh; PPO retains its original Gaussian
    sample so its density ratio needs no inverse or Jacobian correction.
    """
    (common, difference) = np.asarray(raw_action).reshape(2, 5)
    scale = np.sqrt(1 + differential_scale**2)
    return np.r_[
        (common + differential_scale * difference) / scale,
        -(common - differential_scale * difference) / scale,
    ]


def make_env(
    body,
    n_envs=4,
    horizon_s=20.0,
    eval_speed=None,
    interface="cassie_centered_curriculum",
):
    if body != "cassie" or interface != "cassie_centered_curriculum":
        raise ValueError("Unsupported body/interface for this experiment")
    from .centered_balance import CenteredCassieEnv

    return CenteredCassieEnv(body, n_envs, horizon_s, eval_speed, interface)

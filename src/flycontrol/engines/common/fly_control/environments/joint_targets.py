"""Robot-native PD targets learned only through task reward.

No teacher, CPG, prerecorded gait, inherited actor or automatic steering.
The complete joint target is the policy output. Target slew respects native
joint speed limits where available; its state is included in observations.
"""

from __future__ import annotations
import math
import gymnasium as gym
import numpy as np

SPEEDS = dict(hexapod=0.5, tripod=0.45, go1=0.8, cassie=0.5)
SPANS = dict(
    hexapod=[0.35, 0.4, 0.45] * 6,
    tripod=[1.0, 1.0] * 3,
    go1=[0.3, 0.7, 0.9] * 4,
    cassie=[0.15, 0.2, 0.5, 0.5, 0.4] * 2,
)


class JointTargetEnv(gym.Env):
    def __init__(self, body="hexapod", steps=1000, **kwargs):
        super().__init__()
        self.body = body
        self.limit = steps
        if body == "go1":
            from fly_control.environments.go1 import Go1Env, STAND

            self.robot = Go1Env(command=[SPEEDS[body], 0, 0], max_episode_steps=steps)
            self.nominal = STAND.copy()
            self.qids = np.arange(7, 19)
            self.vids = np.arange(6, 18)
            self.angle_scale = 1.5
            self.rates = np.full(12, 6.0)
        elif body == "tripod":
            from fly_control.environments.tripod import TripodEnv

            self.robot = TripodEnv(steps=steps)
            self.nominal = np.zeros(6)
            self.qids = np.arange(7, 13)
            self.vids = np.arange(6, 12)
            self.angle_scale = 1.2
            self.rates = np.full(6, 6.0)
        else:
            from fly_control.environments.cassie_hexapod import EmbodimentEnv

            self.robot = EmbodimentEnv(body, steps=steps, command=[SPEEDS[body], 0, 0])
            native = self.robot.native
            self.nominal = native.nominal_joint_positions.copy()
            self.qids = (
                np.arange(7, self.robot.model.nq)
                if body == "hexapod"
                else native.joint_mask_qpos
            )
            self.vids = (
                np.arange(6, self.robot.model.nv)
                if body == "hexapod"
                else native.joint_mask_qvel
            )
            self.angle_scale = (
                self.robot.action_scale * native.control_function.seen_scaling_factor
            )
            self.rates = np.asarray(native.max_joint_velocities, float).copy()
        self.model, self.data = self.robot.model, self.robot.data
        self.dt = self.robot.dt
        self.trunk_id = self.robot.trunk_id
        self.nu = self.model.nu
        self.span = np.asarray(SPANS[body], float)
        joint_ids = self.model.actuator_trnid[:, 0]
        limited = self.model.jnt_limited[joint_ids].astype(bool)
        self.lower = np.where(
            limited, self.model.jnt_range[joint_ids, 0] + 1e-4, -np.inf
        )
        self.upper = np.where(
            limited, self.model.jnt_range[joint_ids, 1] - 1e-4, np.inf
        )
        assert np.all(self.nominal >= self.lower) and np.all(self.nominal <= self.upper)
        self.qref = self.model.qpos0[7:].copy()
        self.action_space = gym.spaces.Box(-1, 1, (self.nu,), np.float32)
        # Include passive joint positions/velocities as well (especially Cassie).
        obs_dim = 14 + (self.model.nq - 7) + (self.model.nv - 6) + 2 * self.nu
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.target = self.nominal.copy()
        self.previous_action = np.zeros(self.nu)
        self.total_steps = 0
        self.steps = 0
        self.origin = np.zeros(2)

    def observation(self):
        rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        return np.r_[
            rot.T @ self.data.qvel[:3],
            self.data.qvel[3:6],
            rot[2],
            self.data.qpos[2],
            self.data.qpos[7:] - self.qref,
            self.data.qvel[6:],
            (self.target - self.nominal) / self.span,
            self.previous_action,
            SPEEDS[self.body],
            self.data.qpos[1] - self.origin[1],
            math.sin(yaw),
            math.cos(yaw),
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # A persistent, independent per-environment reset stream, including auto-reset.
        draw = int(self.np_random.integers(0, 2**31 - 1))
        self.robot.reset(seed=draw)
        self.origin = self.data.qpos[:2].copy()
        self.target = self.nominal.copy()
        self.previous_action[:] = 0
        self.steps = 0
        self.initial_height = float(self.data.qpos[2])
        return self.observation(), dict(reset_seed=draw)

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (self.nu,) or not np.isfinite(action).all():
            raise ValueError("Invalid policy action")
        action = np.clip(action, -1, 1)
        wanted = np.clip(self.nominal + self.span * action, self.lower, self.upper)
        old_target = self.target.copy()
        self.target += np.clip(
            wanted - self.target, -self.rates * self.dt, self.rates * self.dt
        )
        # Inverse of the unchanged native PD target coordinates, not inverse dynamics.
        executed = (
            (self.target - self.nominal) / self.angle_scale
            if self.body != "tripod"
            else self.target / 1.2
        )
        assert np.max(np.abs(executed)) <= 1 + 1e-6
        before = self.data.qpos[:2].copy()
        _, _, terminated, truncated, info = self.robot.step(executed)
        self.steps += 1
        self.total_steps += 1
        self.previous_action = action.copy()
        rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        up = float(rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        velocity = (self.data.qpos[:2] - before) / self.dt
        target = SPEEDS[self.body]
        upright = float(np.clip((up - 0.4) / 0.6, 0, 1))
        forward = float(np.clip(velocity[0] / target, -2, 1.5))
        # As in stable Ant: attenuate positive progress when tilted, retain backward cost.
        progress = 2 * (max(0, forward) * upright + min(0, forward))
        lateral = float(self.data.qpos[1] - self.origin[1])
        reward = (
            progress
            + 0.1 * upright
            - 0.1 * min((velocity[1] / target) ** 2, 4)
            - 0.1 * (1 - math.cos(yaw))
            - 0.05 * min((lateral / 0.75) ** 2, 4)
            - 0.005 * float(np.mean(action**2))
            - 0.005
            * float(np.mean(((self.target - old_target) / (self.rates * self.dt)) ** 2))
        )
        failed = bool(terminated or info.get("physical_failure", False))
        if failed:
            reward -= 2.0
        info.update(
            x_velocity=float(velocity[0]),
            y_velocity=float(velocity[1]),
            upright_factor=upright,
            functional_failure=failed,
            physical_failure=failed,
            flip=bool(up < 0.4),
            terrain=self.body,
            context_id=0,
            global_steps=self.total_steps,
            reward_forward=progress,
            reward_survive=0.1 * upright,
            teacher_at_evaluation=False,
            teacher_used=False,
            cpg_used=False,
            policy_controls_all_joints=True,
        )
        obs = self.observation()
        assert obs.shape == self.observation_space.shape
        return (
            obs,
            float(reward),
            failed,
            bool(truncated or self.steps >= self.limit),
            info,
        )

    def close(self):
        self.robot.close()


def make_env(body, steps=1000):
    return JointTargetEnv(body, steps)

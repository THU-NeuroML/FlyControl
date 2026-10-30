"""Shared analytic hexapod gait with reward-trained joint corrections.

No actor, action labels, demonstration trajectories or fitted teacher are used.
The analytic CPG is part of both methods' explicitly declared motor interface.
"""

import gymnasium as gym
import numpy as np

from .gait import HexapodCPG
from fly_control.environments.joint_targets import JointTargetEnv
from .env import TaskEnv


class CPGJointTargets(JointTargetEnv):
    def __init__(self, body, steps):
        if body != "hexapod":
            raise ValueError("This gait is specific to the hexapod kinematics")
        super().__init__(body, steps)
        self.cpg = HexapodCPG(self.robot, dict(frequency=2.4, stride=0.065))
        self.residual_span = np.tile([0.15, 0.2, 0.2], 6)
        # Restore the native position-reference range. Travel and motor forces
        # remain enforced by the original model, without an extra slew filter.
        self.span = np.minimum(
            self.angle_scale,
            np.maximum(self.upper - self.nominal, self.nominal - self.lower),
        )
        self.rates = 2 * self.span / self.dt

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self.cpg.reset()
        phase = self.np_random.uniform(0, 2 * np.pi)
        self.cpg.cpg.curr_phases += phase
        return self.observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.nu,) or not np.isfinite(action).all():
            raise ValueError("Invalid residual action")
        residual = np.clip(action, -1, 1)
        base = self.cpg.action(feedback=False)
        target_offset = base * self.angle_scale + residual * self.residual_span
        _, reward, term, trunc, info = super().step(target_offset / self.span)
        self.previous_action = residual.copy()
        info.update(
            cpg_used=True,
            prescribed_joint_gait=True,
            policy_controls_all_joints=False,
            policy_controls_joint_corrections=True,
        )
        return self.observation(), reward, term, trunc, info


class HexapodResidualEnv(TaskEnv):
    def __init__(
        self,
        body,
        n_envs=4,
        horizon_s=20.0,
        eval_speed=None,
        interface="hexapod_cpg_residual",
    ):
        super().__init__(body, n_envs, horizon_s, eval_speed, interface="native")
        self.interface = interface
        self.config.update(initial_speed=0.5, ramp_steps=1)
        self.command = self.training_command = 0.5
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self.physics.observation_space.shape[0] + 12,),
            dtype=np.float32,
        )

    @staticmethod
    def make_physics(body, steps):
        return CPGJointTargets(body, steps)

    def observation(self):
        phase = self.physics.cpg.cpg.curr_phases
        return np.r_[super().observation(), np.sin(phase), np.cos(phase)].astype(
            np.float32
        )

    def step(self, raw_action):
        obs, reward, term, trunc, info = super().step(raw_action)
        info.update(
            cpg_used=True,
            prescribed_joint_gait=True,
            policy_controls_all_joints=False,
            policy_controls_joint_corrections=True,
            cpg_handwritten_feedback=False,
        )
        return obs, reward, term, trunc, info


class PathAlignedHexapodEnv(HexapodResidualEnv):
    """Same gait and physics; explicitly price accumulated path displacement."""

    def __init__(
        self,
        body,
        n_envs=4,
        horizon_s=20.0,
        eval_speed=None,
        interface="hexapod_cpg_path",
    ):
        super().__init__(body, n_envs, horizon_s, eval_speed, interface)

    def step(self, raw_action):
        obs, reward, term, trunc, info = super().step(raw_action)
        lateral = float(self.data.qpos[1] - self.physics.origin[1])
        cost = 0.75 * min((lateral / 0.75) ** 2, 4.0) * info["control_duration_s"]
        info["reward_path_alignment"] = -cost
        return obs, float(reward - cost), term, trunc, info

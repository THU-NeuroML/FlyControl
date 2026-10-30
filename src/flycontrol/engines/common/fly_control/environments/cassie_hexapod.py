"""Native MuJoCo adapters for the pinned One Policy robot environments.

The upstream environment code and robot physics are retained. Its optional
RL-X experiment registry is bypassed by importing just the environment package.
No teacher policy is imported or evaluated by this module.
"""

from __future__ import annotations
import os
import importlib
import sys
import types
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np

ASSET_ROOT = Path(os.environ["FIG6_ASSET_ROOT"]).resolve()
UPSTREAM = ASSET_ROOT / "one_policy_to_run_them_all"
UPSTREAM_REVISION = "7c1819444028c529b0a964fe3f32878333b1310a"


class _NumpyCompatibility:
    """Local spelling compatibility for upstream's NumPy 1.x np.Inf."""

    def __getattr__(self, name):
        return np.inf if name == "Inf" else getattr(np, name)


def load_robot_class(robot):
    if robot not in {"hexapod", "cassie"}:
        raise ValueError(robot)
    prefix = "one_policy_to_run_them_all"
    for suffix in ("", ".environments", f".environments.{robot}"):
        name = prefix + suffix
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(UPSTREAM / name.replace(".", "/"))]
            sys.modules[name] = package
    module = importlib.import_module(f"{prefix}.environments.{robot}.environment")
    module.np = _NumpyCompatibility()
    return getattr(module, {"hexapod": "Hexapod", "cassie": "Cassie"}[robot])


class EmbodimentEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot="hexapod",
        *,
        command=None,
        steps=1000,
        reset_noise=0.01,
        action_scale=10.0,
    ):
        super().__init__()
        self.robot_name = robot
        cls = load_robot_class(robot)
        self.native = cls(
            seed=0,
            render=False,
            mode="train",
            control_type="rudin2022",
            command_type="random",
            command_sampling_type="none",
            initial_state_type="default",
            reward_type="rudin_own_var",
            termination_type=(
                "trunk_and_hip_collision" if robot == "hexapod" else "height"
            ),
            domain_randomization_sampling_type="none",
            domain_randomization_action_delay_type="none",
            domain_randomization_mujoco_model_type="default",
            domain_randomization_seen_robot_type="default",
            domain_randomization_unseen_robot_type="default",
            domain_randomization_perturbation_type="none",
            domain_randomization_perturbation_sampling_type="none",
            observation_noise_type="none",
            observation_dropout_type="none",
            terrain_type="plane",
            missing_value=0.0,
            add_goal_arrow=False,
            timestep=0.005,
            episode_length_in_seconds=steps / 50,
            total_nr_envs=1,
        )
        (self.model, self.data) = (self.native.model, self.native.data)
        self.dt = self.native.dt
        self.trunk_id = self.model.body("trunk").id
        (self.limit, self.reset_noise) = (int(steps), float(reset_noise))
        self.action_scale = float(action_scale)
        self.fixed_command = np.array(
            (
                command
                if command is not None
                else [0.25, 0, 0] if robot == "hexapod" else [0.5, 0, 0]
            ),
            float,
        )
        self.command = self.fixed_command.copy()
        self.native.command_function.get_next_command = lambda: tuple(self.command)
        self.native.eval = True
        self.native.eval_at_last_setup = True
        self.indices = np.array(
            [
                i
                for (name, i) in self.native.observation_name_to_id.items()
                if "_description_" not in name
                and name
                not in {
                    "p_gain",
                    "d_gain",
                    "action_scaling_factor",
                    "mass",
                    "robot_length",
                    "robot_width",
                    "robot_height",
                }
            ],
            dtype=np.int64,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (len(self.indices),), np.float32
        )
        self.action_space = gym.spaces.Box(-1, 1, (self.model.nu,), np.float32)
        self.renderer = None
        self.steps = 0

    def set_command(self, command):
        self.command[:] = command
        (
            self.native.goal_x_velocity,
            self.native.goal_y_velocity,
            self.native.goal_yaw_velocity,
        ) = self.command

    def observation(self):
        return self.native.get_observation()[self.indices].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.native.np_rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        self.native.reset(seed=seed)
        self.steps = 0
        self.set_command(self.fixed_command)
        for name in self.native.joint_names:
            address = self.model.jnt_qposadr[self.model.joint(name).id]
            self.data.qpos[address] += self.np_random.uniform(
                -self.reset_noise, self.reset_noise
            )
        self.data.qvel[:] += self.np_random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nv
        )
        mujoco.mj_forward(self.model, self.data)
        self.native.update_orientation_attributes()
        return (self.observation(), {})

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite action for every actuator")
        (_, reward, terminated, truncated, info) = self.native.step(
            np.clip(action, -1, 1) * self.action_scale
        )
        self.steps += 1
        up = float(self.data.xmat[self.trunk_id].reshape(3, 3)[2, 2])
        finite = bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        failed = bool(terminated or not finite or up < 0.4)
        info.update(
            physical_failure=failed,
            torso_tilt_degrees=float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
            command=self.command.copy(),
            torso_height=float(self.data.qpos[2]),
        )
        return (self.observation(), float(reward), failed, bool(truncated), info)

    def render(self):
        if self.renderer is None:
            self.model.vis.global_.offwidth = 1280
            self.model.vis.global_.offheight = 720
            for texture in range(self.model.ntex):
                if self.model.tex_type[texture] != mujoco.mjtTexture.mjTEXTURE_2D:
                    continue
                start = self.model.tex_adr[texture]
                channels = int(self.model.tex_nchannel[texture])
                count = (
                    self.model.tex_width[texture]
                    * self.model.tex_height[texture]
                    * channels
                )
                pixels = self.model.tex_data[start : start + count].reshape(
                    -1, channels
                )[:, :3]
                bright = pixels.mean(1) > pixels.mean()
                (pixels[~bright], pixels[bright]) = ([46, 59, 71], [66, 82, 96])
            self.model.mat_reflectance[:] = 0.03
            if self.robot_name == "cassie":
                self.model.vis.headlight.ambient[:] = 0.35
                self.model.vis.headlight.diffuse[:] = 0.8
                self.model.vis.headlight.specular[:] = 0.15
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.trunk_id]
        if self.robot_name == "cassie":
            camera.lookat[2] -= 0.25
        camera.distance = 1.5 if self.robot_name == "hexapod" else 3.1
        camera.azimuth = 125
        camera.elevation = -25
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

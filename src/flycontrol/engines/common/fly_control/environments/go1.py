"""Pinned Go1 physics with twelve policy-controlled joint targets.

MuJoCo integration, reset noise, PD gains and termination thresholds preserve
the Figure 6 environment. Task reward and command timing are applied by the
body-control interface. No navigation planner or pretrained policy is loaded.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import os

ASSET_ROOT = Path(os.environ["FIG6_ASSET_ROOT"])


FEET = ("FR", "FL", "RR", "RL")
STAND = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)
CONTROL_DT = 0.02
ACTION_SCALE = 1.5


def prepare_scene(asset_root: Path = ASSET_ROOT) -> Path:
    """Resolve the pinned model, floor, sensors and physics settings."""
    robot_dir = (asset_root / "unitree_go1").resolve()
    root = ET.parse(robot_dir / "go1.xml").getroot()
    root.find("compiler").set("meshdir", str(robot_dir / "assets"))
    scene = ET.parse(robot_dir / "scene.xml").getroot()
    for child in scene:
        if child.tag == "include":
            continue
        existing = root.find(child.tag)
        if existing is None:
            root.append(child)
        else:
            existing.extend(child)
    # Preserve the registered low-level PD integration settings.
    option = root.find("option")
    option.attrib.update(
        timestep="0.004",
        integrator="implicitfast",
        cone="pyramidal",
        impratio="1",
        solver="PGS",
        iterations="10",
    )
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="gyro", site="imu")
    ET.SubElement(sensors, "velocimeter", name="vel", site="imu")
    ET.SubElement(
        sensors,
        "framezaxis",
        name="up",
        objtype="xbody",
        objname="world",
        reftype="xbody",
        refname="trunk",
    )
    for name in FEET:
        ET.SubElement(
            sensors,
            "framelinvel",
            name=name + "_velocity",
            objtype="site",
            objname=name,
        )
    output = asset_root.resolve() / "go1_control.xml"
    ET.indent(root)
    content = ET.tostring(root, encoding="unicode")
    if not output.is_file() or output.read_text() != content:
        output.write_text(content)
    return output


class Go1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        asset_root: str = str(ASSET_ROOT),
        command=(0.8, 0.0, 0.0),
        max_episode_steps: int = 1500,
        reset_noise_scale: float = 0.03,
        render_mode: str | None = None,
        width: int = 960,
        height: int = 640,
    ):
        super().__init__()
        if max_episode_steps <= 0 or reset_noise_scale < 0:
            raise ValueError("invalid episode length or reset noise")
        self.task, self.render_mode = "flat", render_mode
        self.asset_root = Path(asset_root)
        scene = prepare_scene(self.asset_root)
        self.fixed_command = np.asarray(command, dtype=np.float64)
        if (
            self.fixed_command.shape != (3,)
            or not np.isfinite(self.fixed_command).all()
        ):
            raise ValueError("command must be three finite velocities")
        self.command = self.fixed_command.copy()
        self.max_episode_steps = int(max_episode_steps)
        self.reset_noise_scale = float(reset_noise_scale)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.renderer = None
        self.width, self.height = width, height
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            self.close()
            raise ValueError("expected the pinned 12-actuator Go1 model")
        self.trunk_id = self.model.body("trunk").id
        self.foot_ids = np.array([self.model.geom(name).id for name in FEET])
        self.foot_sites = np.array([self.model.site(name).id for name in FEET])
        for joint_id in range(1, self.model.njnt):
            dof_id = self.model.jnt_dofadr[joint_id]
            gear = 9 if self.model.joint(joint_id).name.endswith("calf_joint") else 6
            self.model.dof_damping[dof_id] = 0
            self.model.dof_frictionloss[dof_id] = 0
            self.model.dof_armature[dof_id] = 0.000111842 * gear**2
        self.model.actuator_gainprm[:, 0] = 35.0
        self.model.actuator_biasprm[:, 1] = -35.0
        self.model.actuator_biasprm[:, 2] = -0.5
        self.model.actuator_ctrllimited[:] = False
        self.model.geom_condim[self.foot_ids] = 3
        self.model.geom_solimp[self.foot_ids] = [0.9, 0.95, 0.001, 0.5, 2.0]
        self.model.geom_margin[self.foot_ids] = 0.0
        self.init_qpos = self.model.key_qpos[0].copy()
        self.init_qpos[7:] = STAND
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (50,), np.float32)
        self.steps = self.total_steps = 0
        self.phase = 0.0
        self.previous_action = np.zeros(12)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _observation(self):
        moving = np.linalg.norm(self.command) > 0.01
        return np.concatenate(
            [
                self.data.sensor("vel").data,
                self.data.sensor("gyro").data,
                self.data.sensor("up").data,
                self.data.qpos[7:] - STAND,
                self.data.qvel[6:],
                self.previous_action,
                self.command,
                [
                    math.sin(2 * math.pi * self.phase) * moving,
                    math.cos(2 * math.pi * self.phase) * moving,
                ],
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[7:] = STAND + self.np_random.uniform(
            -self.reset_noise_scale, self.reset_noise_scale, 12
        )
        self.data.qvel[:] = self.np_random.uniform(
            -self.reset_noise_scale, self.reset_noise_scale, self.model.nv
        )
        self.data.ctrl[:] = STAND
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] -= self.data.site_xpos[self.foot_sites, 2].min() - 0.024
        mujoco.mj_forward(self.model, self.data)
        self.phase = float(self.np_random.uniform())
        self.previous_action[:] = 0
        self.steps = 0
        self.command = self.fixed_command.copy()
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Go1 needs twelve finite normalized joint targets")
        action = np.clip(action, -1, 1)
        targets = STAND + ACTION_SCALE * action
        before = self.data.xpos[self.trunk_id, :2].copy()
        self.data.ctrl[:] = targets
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        xy_velocity = (self.data.xpos[self.trunk_id, :2] - before) / self.dt
        healthy = bool(
            np.isfinite(np.r_[self.data.qpos, self.data.qvel]).all()
            and 0.16 <= self.data.qpos[2] <= 0.65
        )
        native_reward = float(xy_velocity[0]) + float(healthy)
        terminated = not healthy
        truncated = self.steps + 1 >= self.max_episode_steps
        info = dict(
            x_position=self.data.qpos[0],
            y_position=self.data.qpos[1],
            distance_from_origin=np.linalg.norm(self.data.qpos[:2]),
            x_velocity=xy_velocity[0],
            y_velocity=xy_velocity[1],
            reward_forward=xy_velocity[0],
            reward_survive=float(healthy),
            reward_ctrl=-0.0,
            reward_contact=-0.0,
        )
        self.steps += 1
        self.total_steps += 1
        self.phase = (self.phase + 2.0 * self.dt) % 1.0
        self.previous_action = (
            3.0 * action
        )  # target offset in the original 0.5-radian units
        up = float(self.data.sensor("up").data[2])
        finite = bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        fallen = bool(not finite or up < 0.4 or self.data.qpos[2] < 0.16)
        velocity = self.data.sensor("vel").data
        tracking = float(np.exp(-np.sum((velocity[:2] - self.command[:2]) ** 2) / 0.25))
        yaw_tracking = float(
            np.exp(-((self.data.sensor("gyro").data[2] - self.command[2]) ** 2) / 0.25)
        )
        reward = float(native_reward)
        terminated = bool(terminated or fallen)
        info = dict(info)
        info.update(
            robot="Unitree Go1",
            task=self.task,
            functional_failure=fallen,
            physical_failure=fallen,
            flip=fallen,
            torso_height=float(self.data.qpos[2]),
            torso_up_z=up,
            torso_tilt_degrees=float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
            command=self.command.copy(),
            command_tracking=tracking,
            action_targets=targets.copy(),
            terrain="go1_" + self.task,
            context_id=0,
            global_steps=self.total_steps,
        )
        return self._observation(), reward, terminated, bool(truncated), info

    def get_state(self):
        return {"format_version": 1, "total_steps": self.total_steps}

    def set_state(self, state):
        if state.get("format_version") != 1 or int(state["total_steps"]) < 0:
            raise ValueError("invalid Go1 protocol state")
        self.total_steps = int(state["total_steps"])

    def render(self):
        if self.renderer is None:
            self.model.vis.global_.offwidth = self.width
            self.model.vis.global_.offheight = self.height
            self.renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.trunk_id]
        camera.distance, camera.azimuth, camera.elevation = 3.2, 130, -22
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    print(prepare_scene())

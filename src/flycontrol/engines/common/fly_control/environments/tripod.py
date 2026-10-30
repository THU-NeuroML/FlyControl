"""Three-leg, six-actuator creature based on DeepMind's PyMJCF tutorial.

Geometry, axes, link lengths, joint damping and position gains follow the
tutorial's Leg and make_creature definitions. Adds a free root, ground, finite
joint/actuator limits and a locomotion task. There is no gait in env.step().
"""

from __future__ import annotations
import math
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np

DEFAULT_FREQUENCY = 1.7984182939242415


def tripod_xml():
    root = ET.Element("mujoco", model="radial_tripod")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="2", type="hinge", range="-1.2 1.2")
    ET.SubElement(default, "geom", friction="0.9 0.005 0.0001", density="1000")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="256",
        height="256",
        rgb1="0.18 0.23 0.28",
        rgb2="0.24 0.30 0.35",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor",
        texture="grid",
        texrepeat="3 3",
        texuniform="true",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="0 0 .1", material="floor"
    )
    torso = ET.SubElement(world, "body", name="trunk", pos="0 0 0.125")
    ET.SubElement(torso, "freejoint")
    ET.SubElement(
        torso,
        "geom",
        name="torso",
        type="ellipsoid",
        size=".1 .1 .05",
        rgba=".18 .35 .43 1",
    )
    actuators = ET.SubElement(root, "actuator")
    for i in range(3):
        theta = 2 * math.pi * i / 3
        thigh = ET.SubElement(
            torso,
            "body",
            name=f"thigh_{i}",
            pos=f"{0.1 * math.cos(theta)} {0.1 * math.sin(theta)} 0",
            euler=f"0 0 {theta}",
        )
        ET.SubElement(thigh, "joint", name=f"hip_{i}", axis="0 0 1")
        ET.SubElement(
            thigh,
            "geom",
            name=f"thigh_geom_{i}",
            type="capsule",
            fromto="0 0 0 .1 0 0",
            size=".025",
            rgba=".18 .65 .62 1",
        )
        shin = ET.SubElement(thigh, "body", name=f"shin_{i}", pos=".1 0 0")
        ET.SubElement(shin, "joint", name=f"knee_{i}", axis="0 1 0")
        ET.SubElement(
            shin,
            "geom",
            name=f"foot_{i}",
            type="capsule",
            fromto="0 0 0 0 0 -.1",
            size=".02",
            rgba=".75 .82 .85 1",
        )
        for joint in (f"hip_{i}", f"knee_{i}"):
            ET.SubElement(
                actuators,
                "position",
                name=joint,
                joint=joint,
                kp="10",
                ctrlrange="-1.2 1.2",
                forcerange="-10 10",
            )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


class TripodEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    robot_name = "tripod"
    dt = 0.02

    def __init__(self, *, steps=1000, reset_noise=0.01, frequency=None, **kwargs):
        super().__init__()
        if frequency is None:
            frequency = DEFAULT_FREQUENCY
        self.model = mujoco.MjModel.from_xml_string(tripod_xml())
        self.data = mujoco.MjData(self.model)
        self.trunk_id = self.model.body("trunk").id
        (self.limit, self.reset_noise, self.frequency) = (steps, reset_noise, frequency)
        self.action_space = gym.spaces.Box(-1, 1, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (30,), np.float32)
        self.indices = np.arange(30)
        self.previous_action = np.zeros(6)
        self.phase = 0.0
        self.steps = 0
        self.renderer = None

    def observation(self):
        rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        return np.r_[
            self.data.qpos[2],
            rot.T @ self.data.qvel[:3],
            self.data.qvel[3:6],
            rot[2],
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.previous_action,
            math.sin(self.phase),
            math.cos(self.phase),
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.np_random.uniform(
            -self.reset_noise, self.reset_noise, 6
        )
        self.data.qvel[:] = self.np_random.uniform(
            -self.reset_noise, self.reset_noise, self.model.nv
        )
        self.previous_action[:] = 0
        (self.steps, self.phase) = (0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        return (self.observation(), {})

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Tripod needs six finite actions")
        action = np.clip(action, -1, 1)
        previous_x = self.data.qpos[0]
        self.data.ctrl[:] = 1.2 * action
        mujoco.mj_step(self.model, self.data, nstep=10)
        self.steps += 1
        self.phase = (self.phase + 2 * math.pi * self.frequency * self.dt) % (
            2 * math.pi
        )
        self.previous_action = action.copy()
        up = self.data.xmat[self.trunk_id].reshape(3, 3)[2, 2]
        failed = bool(
            up < 0.4
            or self.data.qpos[2] < 0.065
            or (not np.isfinite(self.data.qvel).all())
        )
        reward = (self.data.qpos[0] - previous_x) / self.dt
        return (
            self.observation(),
            float(reward),
            failed,
            self.steps >= self.limit,
            dict(
                physical_failure=failed,
                torso_tilt_degrees=float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
                torso_height=float(self.data.qpos[2]),
            ),
        )

    def render(self):
        if self.renderer is None:
            (self.model.vis.global_.offwidth, self.model.vis.global_.offheight) = (
                1280,
                720,
            )
            self.model.vis.headlight.ambient[:] = 0.4
            self.model.vis.headlight.diffuse[:] = 0.8
            self.model.vis.headlight.specular[:] = 0.15
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.xpos[self.trunk_id]
        (cam.distance, cam.azimuth, cam.elevation) = (1.1, 160, -55)
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

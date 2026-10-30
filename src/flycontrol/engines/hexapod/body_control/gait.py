"""Robot-scaled FlyGym tripod CPG with bounded sensory joint corrections.

The same interface is used for both learned policy families. MuJoCo model
parameters, actuator limits and physical integration are never changed.
"""

from __future__ import annotations
import math
import mujoco
import numpy as np
from .oscillator import CPGNetwork


class HexapodCPG:
    # Native order is LF, RF, LM, RM, LH, RH.
    biases = np.pi * np.array([0, 1, 1, 0, 0, 1], dtype=float)
    defaults = dict(
        frequency=1.8,
        stride=0.065,
        lift=0.035,
        height=0.155,
        stance=0.58,
        fore_offset=0.0,
        heading_gain=0.8,
        feedback_gain=0.5,
    )

    def __init__(self, env, parameters=None):
        self.p = {**self.defaults, **(parameters or {})}
        self.env = env
        self.nominal = env.native.nominal_joint_positions.copy()
        self.action_scale = (
            env.action_scale * env.native.control_function.seen_scaling_factor
        )
        self.body_ids = np.array(
            [env.model.body(f"leg_{i}_link_2").id for i in range(6)]
        )
        self.geom_ids = np.array([env.model.geom(f"foot_{i}").id for i in range(6)])
        self.point_local = []
        reference = mujoco.MjData(env.model)
        reference.qpos[:] = env.model.qpos0
        reference.qpos[:3] = [0, 0, self.p["height"]]
        reference.qpos[3:7] = [1, 0, 0, 0]
        reference.qpos[7:] = self.nominal
        mujoco.mj_forward(env.model, reference)
        for gid, bid in zip(self.geom_ids, self.body_ids):
            mesh = env.model.geom_dataid[gid]
            start = env.model.mesh_vertadr[mesh]
            vertices = env.model.mesh_vert[start : start + env.model.mesh_vertnum[mesh]]
            world = (
                vertices @ reference.geom_xmat[gid].reshape(3, 3).T
                + reference.geom_xpos[gid]
            )
            toe = world[np.argmin(world[:, 2])]
            self.point_local.append(
                reference.xmat[bid].reshape(3, 3).T @ (toe - reference.xpos[bid])
            )
        self.point_local = np.asarray(self.point_local)
        self.centers = self.points(reference)
        self.centers[:, 0] += self.p["fore_offset"]
        self.centers[:, 2] = 0.005
        self.table, self.z_jacobian = self._template(reference)
        self.reset()

    def points(self, data):
        return np.asarray(
            [
                data.xpos[b] + data.xmat[b].reshape(3, 3) @ p
                for b, p in zip(self.body_ids, self.point_local)
            ]
        )

    def _template(self, data):
        n = 96
        table = np.zeros((6, n, 3))
        dz = np.zeros((6, 3))
        jp = np.zeros((3, self.env.model.nv))
        jr = np.zeros_like(jp)
        for leg, bid in enumerate(self.body_ids):
            for k in range(n):
                phase = k / n
                stance = self.p["stance"]
                if phase < stance:
                    t = phase / stance
                    x, z = self.p["stride"] * (1 - 2 * t), 0.0
                else:
                    t = (phase - stance) / (1 - stance)
                    # Smooth swing endpoint velocities; stance is constant speed.
                    x = self.p["stride"] * (-1 + 2 * (3 * t * t - 2 * t * t * t))
                    z = self.p["lift"] * math.sin(math.pi * t)
                target = self.centers[leg] + [x, 0, z]
                qslice = slice(7 + 3 * leg, 10 + 3 * leg)
                vslice = slice(6 + 3 * leg, 9 + 3 * leg)
                for _ in range(14):
                    point = (
                        data.xpos[bid]
                        + data.xmat[bid].reshape(3, 3) @ self.point_local[leg]
                    )
                    error = target - point
                    if np.linalg.norm(error) < 2e-5:
                        break
                    mujoco.mj_jac(self.env.model, data, jp, jr, point, bid)
                    jac = jp[:, vslice]
                    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(3), error)
                    data.qpos[qslice] = np.clip(
                        data.qpos[qslice] + np.clip(dq, -0.2, 0.2), -2.3, 2.3
                    )
                    mujoco.mj_forward(self.env.model, data)
                table[leg, k] = data.qpos[qslice]
                if k == 0:
                    dz[leg] = np.linalg.pinv(jp[:, vslice])[:, 2]
        return table, dz

    def reset(self):
        phase_biases = self.biases[None, :] - self.biases[:, None]
        self.cpg = CPGNetwork(
            timestep=self.env.dt,
            intrinsic_freqs=np.full(6, self.p["frequency"]),
            intrinsic_amps=np.ones(6),
            coupling_weights=np.full((6, 6), 4.0),
            phase_biases=phase_biases,
            convergence_coefs=np.full(6, 12.0),
            init_phases=self.biases.copy(),
            init_magnitudes=np.ones(6),
        )
        self.elapsed = 0.0
        self.retraction = np.zeros(6)
        self.last_correction = np.zeros(18)

    def action(self, yaw_rate=0.0, *, feedback=True, advance=True):
        if feedback and not advance:
            raise ValueError(
                "Feedback updates its sensor memory; previews must disable feedback"
            )
        phase = np.mod(self.cpg.curr_phases, 2 * np.pi) / (2 * np.pi)
        index = phase * len(self.table[0])
        low = index.astype(int)
        fraction = index - low
        q = np.stack(
            [
                (1 - fraction[i]) * self.table[i, low[i]]
                + fraction[i] * self.table[i, (low[i] + 1) % len(self.table[i])]
                for i in range(6)
            ]
        )
        # Differential stride modulation, analogous to FlyGym turning magnitudes.
        side = np.sign(self.centers[:, 1])
        gains = np.clip(1 - self.p["heading_gain"] * side * yaw_rate, 0.4, 1.6)
        q[:, 0] = self.nominal.reshape(6, 3)[:, 0] + gains * (
            q[:, 0] - self.nominal.reshape(6, 3)[:, 0]
        )
        correction = np.zeros((6, 3))
        if feedback:
            rot = self.env.data.xmat[self.env.trunk_id].reshape(3, 3)
            roll = math.atan2(rot[2, 1], rot[2, 2])
            pitch = math.asin(np.clip(-rot[2, 0], -1, 1))
            points = self.points(self.env.data)
            relative_z = self.env.data.qpos[2] - points[:, 2]
            # A low, contacting swing leg is lifted; an abnormally extended leg
            # receives persistent retraction. These are kinematic/contact sensors.
            contact = np.array(
                [
                    self.env.native.check_collision("floor", f"foot_{i}")
                    for i in range(6)
                ]
            )
            swing = phase > self.p["stance"] + 0.06
            retract = relative_z > np.median(relative_z) + 0.04
            stumble = swing & contact & (phase < 0.94)
            self.retraction = np.clip(
                self.retraction
                + self.env.dt * np.where(retract | stumble, 0.08, -0.12),
                0,
                0.025,
            )
            dz = np.clip(
                self.p["feedback_gain"]
                * (-roll * self.centers[:, 1] + pitch * self.centers[:, 0]),
                -0.025,
                0.025,
            )
            dz += self.retraction
            correction = np.clip(dz[:, None] * self.z_jacobian, -0.18, 0.18)
        if advance:
            self.last_correction = correction.ravel() / self.action_scale
        q += correction
        blend = min(1.0, self.elapsed / 0.4)
        action = blend * (q.ravel() - self.nominal) / self.action_scale
        if advance:
            self.cpg.step()
            self.elapsed += self.env.dt
        return np.clip(action, -1, 1).astype(np.float32)

    def features(self):
        return np.r_[
            np.sin(self.cpg.curr_phases),
            np.cos(self.cpg.curr_phases),
            self.last_correction,
        ].astype(np.float32)

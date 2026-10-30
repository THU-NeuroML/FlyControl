"""Exact body/flip observations each physics tick; full proprioception each10ms.

Only for this fixed command2 interface with no visual/olfactory/head reflex.
Validated against original observations, rewards, qpos and qvel at every10ms.
"""

import types
import numpy as np
from nmf_sim.fly import R as Rotation


def install(env):
    fly = env._sim.flies[0]
    if getattr(fly, "_cs_fast_owner", None) is env:
        return
    assert not hasattr(fly, "_cs_fast_owner"), "Unexpected shared fly object"
    fly._cs_fast_owner = env
    original = fly.get_observation
    simstep = env._sim.step
    assert (
        fly.head_stabilization_model is None
        and not fly.enable_vision
        and not fly.enable_olfaction
    )
    assert (
        env.action_mode == "command2"
        and env.observation_delay == 0
        and env.action_delay == 0
    )
    env._fast_observation_active = False

    def observation(self, sim):
        if (
            not env._fast_observation_active
            or (env._physics_steps + 1) % env.decision_interval == 0
        ):
            return original(sim)
        physics = sim.physics
        card = np.array(
            [physics.bind(self._body_sensors[i]).sensordata.copy() for i in [4, 5, 6]]
        )
        if card[2, 2] < 0:
            return original(sim)
        pos = physics.bind(self._body_sensors[0]).sensordata
        vel = physics.bind(self._body_sensors[1]).sensordata
        quat = physics.bind(self._body_sensors[2]).sensordata
        angles = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("ZYX")
        angular = physics.bind(self._body_sensors[3]).sensordata
        self.last_obs["rot"] = angles
        self.last_obs["pos"] = pos
        return dict(
            fly=np.array([pos, vel, angles, angular]).astype(np.float32),
            fly_orientation=card[0].astype(np.float32),
            cardinal_vectors=card.astype(np.float32),
        )

    def step(action):
        env._fast_observation_active = True
        try:
            return simstep(action)
        finally:
            env._fast_observation_active = False

    fly.get_observation = types.MethodType(observation, fly)
    env._sim.step = step

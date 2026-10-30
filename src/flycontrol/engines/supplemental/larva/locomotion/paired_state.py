"""Exact in-process snapshots for the fixed-body, flat-terrain command2 task.

This deliberately restricts the supported environment. Copying qpos/qvel alone
is insufficient: solver warm starts, actuator controls, adhesion bookkeeping,
CPG state, observation buffers and RNGs must also be restored. MuJoCo's C API
mj_copyData copies the complete mjData without running forward dynamics.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch


_LIB = ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))))
_COPY_DATA = _LIB.mj_copyData
_COPY_DATA.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_COPY_DATA.restype = ctypes.c_void_p
_ENV_HANDLES = {'_sim', '_fly', '_controller', '_sim_cache', 'action_space', 'observation_space'}
_FLY_RUNTIME = ('last_obs', '_last_observation', '_last_neck_actuation', '_last_adhesion',
                '_active_adhesion', '_last_vision_update_time', '_curr_raw_visual_input',
                '_curr_visual_input', '_vision_update_mask', '_np_random', '_np_random_seed')


def _plain_tree(value):
    if isinstance(value, dict):
        return {k: _plain_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain_tree(v) for v in value)
    if hasattr(value, '__array__') and hasattr(value, 'shape'):
        return np.array(value, copy=True)
    return value


def _validate(env):
    if env._sim is None or env._controller is None:
        raise ValueError('snapshot requires a reset environment')
    if env.task != 'heading_hold' or env.action_mode != 'command2' or env._current_terrain != 'flat':
        raise ValueError('snapshot currently supports flat heading_hold command2 only')
    if env._sim.cameras or env._fly.enable_vision or env._fly.enable_olfaction:
        raise ValueError('snapshot does not support cameras, vision or olfaction')
    if env._fly.head_stabilization_model is not None:
        raise ValueError('snapshot does not support a stateful head stabilization model')


def state_digest(env, observation, hidden):
    physics = env._sim.physics
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(physics.model.ptr, spec))
    mujoco.mj_getState(physics.model.ptr, physics.data.ptr, state, spec)
    h = hashlib.sha256()
    for a in (state, observation, hidden.detach().cpu().numpy(),
              physics.data.qacc, physics.data.sensordata, physics.data.cfrc_ext):
        h.update(np.ascontiguousarray(a).tobytes())
    # Include all controller state and the environment's RNG/history, not just
    # the directly visible phase. Exclude constant handles bound to MuJoCo.
    h.update(pickle.dumps(env._controller.__dict__, protocol=5))
    h.update(pickle.dumps({k: v for k, v in env.__dict__.items() if k not in _ENV_HANDLES}, protocol=5))
    h.update(pickle.dumps(_plain_tree({k: getattr(env._fly, k) for k in _FLY_RUNTIME if hasattr(env._fly, k)}), protocol=5))
    h.update(pickle.dumps((env._sim.curr_time, env._sim.np_random.bit_generator.state), protocol=5))
    return h.hexdigest()


@dataclass
class PairedSnapshot:
    data: object
    model_address: int
    env_state: dict
    controller_state: dict
    fly_state: dict
    sim_time: float
    sim_rng: object
    observation: np.ndarray
    hidden: torch.Tensor
    numpy_rng: tuple
    python_rng: tuple
    torch_rng: torch.Tensor
    cuda_rng: list | None
    digest: str

    @classmethod
    def capture(cls, env, observation, hidden):
        _validate(env)
        # Deep-copy the Python payload in one operation to retain aliasing.
        payload = copy.deepcopy({
            'env': {k: v for k, v in env.__dict__.items() if k not in _ENV_HANDLES},
            'controller': env._controller.__dict__,
            'fly': _plain_tree({k: getattr(env._fly, k) for k in _FLY_RUNTIME if hasattr(env._fly, k)}),
            'sim_rng': env._sim.np_random,
        })
        return cls(copy.copy(env._sim.physics.data.ptr), env._sim.physics.model.ptr._address,
                   payload['env'], payload['controller'], payload['fly'], env._sim.curr_time,
                   payload['sim_rng'], np.array(observation, copy=True), hidden.detach().clone(),
                   np.random.get_state(), random.getstate(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if hidden.is_cuda else None,
                   state_digest(env, observation, hidden))

    def restore(self, env):
        _validate(env)
        physics = env._sim.physics
        if physics.model.ptr._address != self.model_address:
            raise ValueError('snapshot belongs to a different compiled model')
        _COPY_DATA(physics.data.ptr._address, physics.model.ptr._address, self.data._address)
        payload = copy.deepcopy({'env': self.env_state, 'controller': self.controller_state,
                                 'fly': self.fly_state, 'sim_rng': self.sim_rng})
        for k in set(env.__dict__) - _ENV_HANDLES - set(payload['env']):
            del env.__dict__[k]
        env.__dict__.update(payload['env'])
        env._controller.__dict__.clear()
        env._controller.__dict__.update(payload['controller'])
        for k in _FLY_RUNTIME:
            if k in payload['fly']:
                setattr(env._fly, k, payload['fly'][k])
            elif hasattr(env._fly, k):
                delattr(env._fly, k)
        env._sim.curr_time = self.sim_time
        env._sim._np_random = payload['sim_rng']
        np.random.set_state(self.numpy_rng)
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        obs, hidden = self.observation.copy(), self.hidden.clone()
        digest = state_digest(env, obs, hidden)
        if digest != self.digest:
            raise RuntimeError(f'state restoration digest mismatch: {digest} != {self.digest}')
        return obs, hidden

    def archive(self):
        """Serializable audit payload; restoration also requires the same model."""
        return {'mujoco_data': self.data, 'env': self.env_state, 'controller': self.controller_state,
                'fly': self.fly_state, 'sim_time': self.sim_time, 'sim_rng': self.sim_rng,
                'observation': self.observation, 'hidden': self.hidden.detach().cpu().numpy(),
                'numpy_rng': self.numpy_rng, 'python_rng': self.python_rng, 'torch_rng': self.torch_rng,
                'cuda_rng': self.cuda_rng, 'digest': self.digest}

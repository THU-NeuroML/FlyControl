"""Frozen March 2026 experiment with explicit graph and execution adapters."""
from pathlib import Path
import sys
import json
import hashlib
import os
import time

ROOT = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
SOURCE = ROOT / 'source_original'

import numpy as np
import torch
from cpg_rl_env import CPGRLEnv
from connectome_policy import ConnectomeFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from recurrent_policy import RecurrentCfCPolicy


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure_process():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ManagedLegacyEnv(CPGRLEnv):
    """Only release replaced simulators and count existing physics steps."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_physics_steps = 0

    def _build_sim(self, terrain_name, seed):
        if self._sim is not None:
            self._sim.close()
        super()._build_sim(terrain_name, seed)

    def step(self, action):
        before = self._physics_steps
        obs, reward, terminated, truncated, info = super().step(action)
        delta = self._physics_steps - before
        self.total_physics_steps += delta
        info['executed_physics_steps'] = delta
        info['executed_physics_steps_total'] = self.total_physics_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None


class FrozenGraphExtractor(ConnectomeFeaturesExtractor):
    """Keep the original CfC computation; replace only the fixed graph buffers."""
    def __init__(self, observation_space, graph_path, **kwargs):
        if kwargs.pop('use_lora', False):
            raise ValueError('LoRA must be disabled in both conditions')
        super().__init__(observation_space, use_lora=False, **kwargs)
        w = torch.from_numpy(np.load(graph_path, allow_pickle=False))
        if w.shape != self.rnn.W.shape or not torch.isfinite(w).all():
            raise ValueError('Invalid graph')
        with torch.no_grad():
            self.rnn.W.copy_(w)
            self.rnn.W_init_buf.copy_(w)
            self.rnn.sparsity_mask.copy_((w != 0).float())
        assert not self.rnn.W.requires_grad
        assert not any('lora_' in name for name, _ in self.named_parameters())

    def forward(self, observations):
        raise RuntimeError('CfC requires explicit recurrent state; use advance()')

    def advance(self, observations, state):
        """Original three settling steps, starting from the carried membrane state.

        With LoRA disabled W already has exactly the masked support. Reusing W
        avoids retaining a separate 35 MB W*mask tensor at every BPTT step.
        """
        rnn = self.rnn
        assert not rnn.use_lora and not rnn.noise_std
        s, i = rnn.sensory_dim, rnn.internal_dim
        for _ in range(3):
            r = torch.tanh(state)
            drive = r @ rnn.W
            external = torch.zeros_like(state)
            external[:, s:s+i] = rnn.input_proj(observations)
            full_drive = drive + external
            target = torch.tanh(full_drive)
            gate = rnn._compute_gate(state, r, full_drive)
            state = (1.0 - gate) * state + gate * target
        raw = torch.tanh(rnn.output_layer(torch.tanh(state)[:, s+i:]))
        return state, self.output_proj(raw)


def graph_path(kind, seed):
    return ROOT / 'graphs' / ('connectome.npy' if kind == 'connectome' else f'er_{seed:02d}.npy')


def make_env(rank, seed, monitor_dir=None):
    def factory():
        env = ManagedLegacyEnv(terrain='mixed', terrain_seed=seed + rank,
                               randomize_terrain=True, max_time=1.5,
                               decision_interval=100, timestep=1e-4,
                               max_correction=10., flip_penalty=10.,
                               correction_energy_penalty=1e-3)
        filename = str(Path(monitor_dir) / f'env_{rank}') if monitor_dir else None
        return Monitor(env, filename=filename,
                       info_keywords=('executed_physics_steps_total',))
    return factory


def build_vec(seed, monitor_dir=None, subprocesses=True, n_envs=8):
    factories = [make_env(i, seed, monitor_dir) for i in range(n_envs)]
    raw = (SubprocVecEnv(factories, start_method='spawn') if subprocesses
           else DummyVecEnv(factories))
    return VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.,
                        clip_reward=10., gamma=.99)


def build_model(kind, seed, env, device, n_steps=1024):
    kwargs = {
        'features_extractor_class': FrozenGraphExtractor,
        'features_extractor_kwargs': {
            'graph_path': str(graph_path(kind, seed)),
            'connectome_path': str(SOURCE / 'larva/data/ad_connectivity_matrix.csv'),
            'annotation_path': str(SOURCE / 'larva/data/science.add9330_data_s2.csv'),
            'mode': 'cfc', 'features_dim': 64, 'lora_rank': 16, 'dt': 1e-4,
        },
        'net_arch': dict(pi=[64], vf=[64]),
        'activation_fn': torch.nn.Tanh,
    }
    return RecurrentPPO(RecurrentCfCPolicy, env, learning_rate=3e-4, n_steps=n_steps,
               batch_size=256, n_epochs=10, gamma=.99, gae_lambda=.95,
               clip_range=.2, ent_coef=.01, seed=seed, device=device,
               policy_kwargs=kwargs, verbose=0)


def graph_buffers(model):
    r = model.policy.features_extractor.rnn
    return {name: getattr(r, name).detach().cpu().clone()
            for name in ['W', 'W_init_buf', 'sparsity_mask']}


def validate_frozen(model, reference):
    r = model.policy.features_extractor.rnn
    return all(torch.equal(getattr(r, k).detach().cpu(), v)
               for k, v in reference.items())


def experiment_config():
    return json.loads((ROOT / 'CONFIG.json').read_text())

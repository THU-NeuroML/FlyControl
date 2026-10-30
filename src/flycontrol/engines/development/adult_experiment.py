"""Frozen adult CfC core, explicit sensory input and descending-only readout.

Reuses the audited legacy recurrent PPO interface and CPG environment. The
historical adult sparse kernel is accelerated with an equivalent fixed CSR map.
"""
from pathlib import Path
import sys
import json
import numpy as np
import scipy.sparse as sp
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ROOT = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
LEGACY = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
from experiment import ManagedLegacyEnv,write_json,sha256,configure_process
from recurrent_policy import RecurrentCfCPolicy
from fresh_reset import FreshEpisodeReset,RecordedLegacyEnv
from adult_source.connectome import ConnectomeData
from adult_source.cfc import ConnectomeCfC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv,VecNormalize
from sb3_contrib import RecurrentPPO


def load_graph():
    audit=json.loads((ROOT/'inputs/graph.json').read_text())
    if sha256(ROOT/'inputs/graph.npz')!=audit['sha256']:
        raise RuntimeError('Frozen FlyWire graph artifact changed')
    a=np.load(ROOT/'inputs/graph.npz')
    s=int(a['sensory_dim']);i=int(a['internal_dim']);o=int(a['output_dim'])
    ids=a['root_ids'].tolist()
    return ConnectomeData(W=None,sensory_dim=s,internal_dim=i,output_dim=o,
        sensory_ids=tuple(ids[:s]),internal_ids=tuple(ids[s:s+i]),output_ids=tuple(ids[s+i:]),
        edge_pre=a['edge_pre'],edge_post=a['edge_post'],edge_weight=a['edge_weight'],metadata={'species':'adult'})


class AdultSparseCfC(ConnectomeCfC):
    def __init__(self,connectome,**kwargs):
        super().__init__(connectome,**kwargs)
        self.total_dim=self.N
        wt=sp.csr_matrix((connectome.edge_weight,(connectome.edge_post,connectome.edge_pre)),
                         shape=(self.N,self.N))
        wt.sort_indices()
        self.register_buffer('_transport',torch.sparse_csr_tensor(
            torch.from_numpy(wt.indptr.astype(np.int64)),torch.from_numpy(wt.indices.astype(np.int64)),
            torch.from_numpy(wt.data.copy()),size=wt.shape),persistent=False)

    def _recurrent_drive(self,r):
        return torch.sparse.mm(self._transport,r.T.contiguous()).T


class AdultFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space,features_dim=64):
        super().__init__(observation_space,features_dim)
        self.rnn=AdultSparseCfC(load_graph(),input_dim=observation_space.shape[0],output_dim=features_dim,
            mode='cfc',dt=.01,tau_init=.02,cfc_init='heterogeneous_tau',cfc_tau_log_std=.5,
            cfc_gate_mod_init=.01,use_lora=False,trainable_W=False,noise_std=0.,input_target='sensory')

    def advance(self,observations,state):
        return self.rnn.step(state,observations)

    def forward(self,observations):
        raise RuntimeError('Explicit carried recurrent state required')


class GapBlockEnv(RecordedLegacyEnv):
    def reset(self,*,seed=None,options=None):
        if seed is None:raise ValueError('FreshEpisodeReset must supply an episode seed')
        self.terrain_name=str(np.random.default_rng(seed).choice(['gapped','blocks']))
        self.randomize_terrain=False
        return super().reset(seed=seed,options=options)


def make_env(rank,seed,run,n_envs=16):
    def factory():
        env=GapBlockEnv(terrain='gapped',terrain_seed=seed+rank,randomize_terrain=False,
            max_time=1.5,timestep=1e-4,decision_interval=100,max_correction=10.,
            flip_penalty=10.,correction_energy_penalty=1e-3)
        env=FreshEpisodeReset(env,seed,rank,n_envs,log_path=Path(run)/'monitor'/f'env_{rank}_resets.jsonl')
        return Monitor(env,filename=str(Path(run)/'monitor'/f'env_{rank}'),
            info_keywords=('executed_physics_steps_total','episode_seed','episode_index','terrain'))
    return factory


def build_vec(seed,run,n_envs=16):
    env=SubprocVecEnv([make_env(i,seed,run,n_envs) for i in range(n_envs)],start_method='spawn')
    return VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.,clip_reward=10.,gamma=.99)


def build_model(seed,env,device='cuda:0',n_steps=160):
    return RecurrentPPO(RecurrentCfCPolicy,env,learning_rate=3e-4,n_steps=n_steps,batch_size=256,
        n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.01,vf_coef=.5,
        max_grad_norm=.5,seed=seed,device=device,verbose=0,
        policy_kwargs={'features_extractor_class':AdultFeaturesExtractor,'features_extractor_kwargs':{'features_dim':64},
            'net_arch':dict(pi=[64],vf=[64]),'activation_fn':torch.nn.Tanh})


def frozen_digest(model):
    import hashlib
    core=model.policy.features_extractor.rnn
    result={k:hashlib.sha256(getattr(core,k).detach().cpu().numpy().tobytes()).hexdigest()
            for k in ['edge_pre','edge_post','edge_weight']}
    for k,value in [('csr_values',core._transport.values()),('csr_rows',core._transport.crow_indices()),
                    ('csr_columns',core._transport.col_indices())]:
        result[k]=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    return result

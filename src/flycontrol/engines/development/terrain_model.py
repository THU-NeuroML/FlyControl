"""Adult-matched terrain PPO with a native Larva core or an active MLP."""
from pathlib import Path
import sys, hashlib, json
import numpy as np
import torch
from torch import nn
O = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
from adult_experiment import (AdultSparseCfC, BaseFeaturesExtractor, ConnectomeData,
    RecurrentCfCPolicy, RecurrentPPO, build_vec, configure_process, write_json)

class FeedForwardCore(nn.Module):
    total_dim=1
    def __init__(self):
        super().__init__();widths=[46,878,899,889,64];layers=[]
        for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(widths)-2:layers.append(nn.Tanh())
        self.net=nn.Sequential(*layers)
    def step(self,state,obs):return torch.zeros_like(state),self.net(obs)

class BaselineFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,kind,graph_path=None,features_dim=64):
        super().__init__(observation_space,features_dim)
        if kind=='mlp':self.rnn=FeedForwardCore()
        else:
            with np.load(graph_path) as a:
                s,i,o=(int(a[k]) for k in ['sensory_dim','internal_dim','output_dim']);ids=a['root_ids'].tolist()
                graph=ConnectomeData(W=None,sensory_dim=s,internal_dim=i,output_dim=o,sensory_ids=tuple(ids[:s]),internal_ids=tuple(ids[s:s+i]),output_ids=tuple(ids[s+i:]),edge_pre=a['edge_pre'],edge_post=a['edge_post'],edge_weight=a['edge_weight'],metadata={'species':'larva','normalization':'row_l1','global_gain':1.})
            self.rnn=AdultSparseCfC(graph,input_dim=46,output_dim=64,mode='cfc',dt=.01,tau_init=.02,cfc_init='heterogeneous_tau',cfc_tau_log_std=.5,cfc_gate_mod_init=.01,use_lora=False,trainable_W=False,noise_std=0.,input_target='sensory')
    def advance(self,observations,state):return self.rnn.step(state,observations)
    def forward(self,observations):raise RuntimeError('Use the shared PPO sequence interface')

def build(kind,seed,env,device='cuda:0'):
    model=RecurrentPPO(RecurrentCfCPolicy,env,learning_rate=3e-4,n_steps=160,batch_size=256,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.01,vf_coef=.5,max_grad_norm=.5,seed=seed,device=device,verbose=0,
        policy_kwargs=dict(features_extractor_class=BaselineFeatures,features_extractor_kwargs=dict(kind=kind,graph_path=str(O/'inputs/larva_terrain.npz'),features_dim=64),net_arch=dict(pi=[64],vf=[64]),activation_fn=nn.Tanh))
    assert not any(isinstance(m,nn.LSTM) for m in model.policy.modules())
    if kind=='mlp':assert sum(p.numel() for p in model.policy.parameters())==1697328
    return model

def frozen_digest(model):
    core=model.policy.features_extractor.rnn
    return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in core.named_buffers() if k!='_transport'}

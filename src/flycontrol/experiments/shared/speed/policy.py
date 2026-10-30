"""Frozen 10-ms supplemental policy evaluation and native replay checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from larva.locomotion.env import TerrainLocomotionEnv
from larva.locomotion.tasks import speed_target_observation
from larva.locomotion.train import build_connectome
from larva.locomotion.models import build_policy
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class AxialEnv(TerrainLocomotionEnv):
 def _apply_external_perturbation(self,elapsed_s):
  self._clear_external_forces()
  if self._sim is not None and 1.5<=elapsed_s<1.54:
   self._sim.physics.bind(self._fly.thorax).xfrc_applied=np.array([self.axial_force,0.,0.,0.,0.,0.])
  if self._sim is not None:self.impulse+=float(self._sim.physics.data.xfrc_applied[:,0].sum())*self.timestep
def load(cp):
 a=torch.load(cp,map_location='cpu',weights_only=False);s=a['policy_state'];ar=lambda k:s['actor.core.'+k].numpy()
 cfg=a.get('config',a.get('cfg',{}))
 if 'policy' not in cfg:cfg=yaml.safe_load((Path(cp).parent/'config.yaml').read_text())
 if cfg['policy']['name']=='mlp':
  layers=[(s[f'actor.net.{i}.weight'].numpy(),s[f'actor.net.{i}.bias'].numpy()) for i in [0,2,4]]
  return dict(N=0,cfg=cfg,path=str(cp),sha=sha(cp),state=s,graph=None,layers=layers)
 n=int(ar('input_proj.weight').shape[0]+0) # full N is stored in rate state partition via config/graph.
 cfg=a.get('config',a.get('cfg',{}))
 if 'connectome' not in cfg:cfg=yaml.safe_load((Path(cp).parent/'config.yaml').read_text())
 g=build_connectome(cfg['connectome']);N=g.N;S=g.sensory_dim;D=g.output_dim
 assert ar('input_proj.weight').shape==(S,51) and ar('output_layer.weight').shape==(2,D)
 assert not cfg['policy']['actor_kwargs'].get('goal_adapter_hidden')
 m=dict(N=N,S=S,D=D,W=(csr_matrix(ar('W').T) if 'actor.core.W' in s else csr_matrix((ar('edge_weight'),(ar('edge_post'),ar('edge_pre'))),shape=(N,N))),E=ar('input_proj.weight'),eb=ar('input_proj.bias'),C=ar('output_layer.weight'),cb=ar('output_layer.bias'),gate=np.float32(.01/float(ar('rate_tau'))),cfg=cfg,path=str(cp),sha=sha(cp),state=s,graph=g)
 return m
def step(m,h,o):
 if 'layers' in m:
  z=o
  for i,(w,b) in enumerate(m['layers']):
   z=w@z+b
   if i<2:z=np.tanh(z)
  return h,np.tanh(z)
 u=m['W']@np.tanh(h);u[:m['S']]+=m['E']@o[:,None]+m['eb'][:,None]
 hn=(1-m['gate'])*h+m['gate']*np.tanh(u)
 raw=m['C']@np.tanh(hn[-m['D']:])+m['cb'][:,None]
 return hn,np.tanh(raw)[:,0]
def replay_check(m):
 p=build_policy(m['cfg']['policy']['name'],51,2,actor_kwargs=m['cfg']['policy']['actor_kwargs'],value_hidden=[128,128],connectome=m['graph']);p.load_state_dict(m['state']);p.eval()
 h=np.zeros((m['N'],1),np.float32);ht=torch.zeros(1,m['N']);error=0.;rng=np.random.default_rng(884)
 with torch.no_grad():
  for _ in range(24):
   obs=rng.normal(0,.2,51).astype(np.float32);h,act=step(m,h,obs);a,_,_,ht=p.act(torch.from_numpy(obs[None]),ht,deterministic=True)
   error=max(error,float(abs(act-a.numpy()[0]).max()))
 assert error<2e-5,error
 return dict(max_action_abs_error=error,tolerance=2e-5,steps=24,reference='native Policy.act on CPU',evaluation='float32 SciPy sparse exact recurrence')
def digest(env,obs,h):
 arrays=[env._sim.physics.data.qpos,env._sim.physics.data.qvel,obs,h,env._controller.cpg.curr_phases]
 return hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in arrays)).hexdigest()
def target_at(task,direction,t,target):
 if task=='static':return float(target)
 lo,hi=(8.,16.) if direction=='up' else (16.,8.)
 if task=='step':return lo if t<1.-1e-9 else hi
 return float(np.interp(t,[0.,.5,2.,2.5,4.,4.5],[lo,lo,hi,hi,lo,lo]))

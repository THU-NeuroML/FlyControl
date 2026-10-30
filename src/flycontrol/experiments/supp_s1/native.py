import os,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from flycontrol.paths import run_root
from larva.ant_transfer.reward_transfer_v2.policy import build_policy
ROOT=run_root();DEVICE=os.environ.get('FLYCONTROL_DEVICE','cpu')
torch.set_num_threads(1)
def load(seed,method):
    row=next(x for x in json.loads((ROOT/'training/models.json').read_text()) if x['seed']==seed and x['method']==method)
    cp=ROOT/row['checkpoint'];assert sha(cp)==row['sha256']
    c=torch.load(cp,map_location='cpu',weights_only=False)
    p=build_policy(ROOT/'inputs/graph.npz',688,6,method,seed,DEVICE,'normalized',-1.)
    p.load_state_dict(c['policy_state']);p.eval()
    for w in p.parameters():w.requires_grad_(False)
    return p,dict(row,path=str(cp))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def mean_from_rates(core,rate):return 3*torch.tanh(core.output_layer(rate[:,-1290:])/3)
def native(core,obs):
 b=obs.shape[0];h=core.init_state(b,DEVICE);out=[]
 for t in range(52):
  # Native history at reset contains four copies; subsequent frames are consumed once.
  for _ in range(4 if t==0 else 1):
   prev=h;rate=torch.tanh(prev);drive=core.fixed_graph(rate)
   drive[:,:2279]+=core.input_proj(obs[:,t,-172:]);target=torch.tanh(drive);h=.5*prev+.5*target
  out.append(dict(prev=prev,prev_slope=1-rate**2,target_slope=1-target**2,rate=torch.tanh(h),h=h,mean=mean_from_rates(core,torch.tanh(h))))
 return out
def gate(position):
 u=np.clip((position[:,:,0]-1.8)/.8,0,1);return u*u*(3-2*u)

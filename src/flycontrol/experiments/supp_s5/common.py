"""Frozen speed models and verified atlas coordinates."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,pickle
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from flycontrol.paths import run_root, read_json
ROOT=run_root();O=ROOT/'mechanisms/banc'
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
MODELS=read_json(ROOT/'run.json')['methods']
RUNS=[dict(run_id=f"{m}_seed{s}",model=m,train_seed=s,output=str(ROOT/f'training/speed/{m}_seed{s}'),config=str(ROOT/f'training/configs/speed/{m}_seed{s}.yaml')) for m in MODELS for s in read_json(ROOT/'run.json')['seeds']]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def load(run,stage):
    cp=Path(run['output'])/('initial_checkpoint.pt' if stage=='initial' else 'policy.pt')
    a=torch.load(cp,map_location='cpu',weights_only=False);state=a['policy_state']
    get=lambda k:state['actor.core.'+k].numpy()
    model=run['model'];gname=model
    # Every graph cache is pinned by the registered configuration.
    import yaml
    cfg=yaml.safe_load(Path(run['config']).read_text());gp=Path(cfg['connectome']['frozen_graph_path'])
    assert sha(gp)==cfg['connectome']['frozen_graph_sha256']
    g=pickle.load(gp.open('rb'));pre,post,w=[get(k) for k in ['edge_pre','edge_post','edge_weight']]
    assert np.array_equal(pre,g.edge_pre) and np.array_equal(post,g.edge_post) and np.array_equal(w,g.edge_weight)
    return dict(N=g.N,S=g.sensory_dim,D=g.output_dim,E=get('input_proj.weight'),eb=get('input_proj.bias'),C=get('output_layer.weight'),cb=get('output_layer.bias'),gate=.01/float(get('rate_tau')),sigma=np.exp(state['log_std'].numpy()),W=csr_matrix((w,(post,pre)),shape=(g.N,g.N)),graph=g,path=str(cp),sha=sha(cp),edge_sha=hashlib.sha256(pre.tobytes()+post.tobytes()+w.tobytes()).hexdigest())
def mapping(model):
    key='adult' if model=='er' else model
    return pd.read_csv(O/'source_data'/f'{key}_nodes.csv.gz',dtype={'root_id':str}).fillna('unassigned').sort_values('state_index').reset_index(drop=True)
def masks(model,S):
    m=mapping(model).iloc[:S]
    groups={g:d.index.to_numpy() for g,d in m.groupby('refined_group')}
    assert sum(len(x) for x in groups.values())==S
    return groups
def sparse(w,dev):
    w=w.tocsr();w.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(w.indptr,device=dev,dtype=torch.int64),torch.as_tensor(w.indices,device=dev,dtype=torch.int64),torch.as_tensor(w.data,device=dev,dtype=torch.float32),size=w.shape)
def cpu_step(m,h,obs):
    u=m['W']@np.tanh(h);u[:m['S']]+=m['E']@obs+m['eb'][:,None]
    hn=(1-m['gate'])*h+m['gate']*np.tanh(u)
    raw=m['C']@np.tanh(hn[-m['D']:])+m['cb'][:,None]
    return hn,np.tanh(raw)
def pool():
    rows=json.loads((O/'source_data/history_manifest.json').read_text())
    return [(r,np.load(r['path'])['obs'].astype(np.float32)) for r in rows]

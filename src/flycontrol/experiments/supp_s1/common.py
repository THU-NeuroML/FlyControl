import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,torch
from flycontrol.paths import run_root
from flycontrol.connectomes.anatomy import adult_nodes
from flycontrol.experiments.supp_s1 import native as old, adjoint as scan
ROOT=run_root();W=ROOT/'mechanisms/historical';DEVICE=os.environ.get('FLYCONTROL_DEVICE','cpu')
M,_=adult_nodes(ROOT);N=len(M)
ids=np.arange(N)
GROUPS={'mechanosensory':ids[M['class'].eq('mechanosensory')],
        'AN':ids[M.super_class.eq('ascending')],
        'SA':ids[M.super_class.eq('sensory_ascending')&M['class'].eq('AN')],
        'unknown_sensory':ids[M['class'].eq('unknown_sensory')],
        **{k:ids[M['class'].eq(k)] for k in ['gustatory','visual','hygrosensory','thermosensory','olfactory']}}
AFF=list(GROUPS);assert sum(map(len,GROUPS.values()))==18756
GROUPS['DN']=ids[ids>=N-1290]
GROUPS['other_internal']=np.setdiff1d(ids,np.concatenate(list(GROUPS.values())))
assert len(np.unique(np.concatenate(list(GROUPS.values()))))==sum(map(len,GROUPS.values()))==N
PART=list(GROUPS)
GROUPS.update(brain_motor=ids[M['class'].eq('brain_motor_neuron')],**{k:ids[M.primary_type.eq(k)] for k in ['DNge037','DNg35','DNg78']})
def save(path,x):old.write(path,x)
def cases(seed,smoke=False):
    c=pd.read_csv(ROOT/'evaluation/history_index.csv').query('training_seed==@seed').to_dict('records')
    return c
def batch_load(batch):
    traces=[]
    for r in batch:
        assert old.sha(r['trace'])==r['trace_sha256']
        with np.load(r['trace']) as z:
            assert len(z['time_s'])>=52 and np.allclose(z['time_s'][:52],np.arange(52)*.01)
            traces.append({k:z[k][:52].copy() for k in ['observation','action','position']})
    obs=torch.as_tensor(np.stack([x['observation'] for x in traces]),device=DEVICE)
    gates=torch.as_tensor(old.gate(np.stack([x['position'] for x in traces])),device=DEVICE)
    return obs,gates,traces
def labels(r):return {k:r[k] for k in ['training_seed','source_method','height_mm','phase_seed']}
def replay_error(base,gates,traces,batch):
    err=0.
    for t in range(0,48,4):
        pred=(torch.tanh(base[t]['mean'])*gates[:,t,None]).cpu().numpy()
        for i,r in enumerate(batch):
            if r['source_method']=='connectome':err=max(err,float(np.abs(traces[i]['action'][t+1:t+5]-pred[i]).max()))
    assert err<2e-4
    return err
def prepare():
    (W/'source_data').mkdir(parents=True,exist_ok=True)
    (W/'raw').mkdir(parents=True,exist_ok=True)
    rows=[]
    for k,v in GROUPS.items():rows.append(dict(category=k,n_nodes=len(v),exhaustive_partition=k in PART,afferent=k in AFF))
    pd.DataFrame(rows).to_csv(W/'source_data/GROUP_COUNTS.csv',index=False)
    np.savez_compressed(W/'source_data/GROUPS.npz',**GROUPS)

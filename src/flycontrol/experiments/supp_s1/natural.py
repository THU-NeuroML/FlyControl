"""Actual matched trajectories plus two explicit cue-counterfactual replays."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json,argparse
import numpy as np,pandas as pd,torch
from flycontrol.experiments.supp_s1 import common as C
from flycontrol.paths import read_json
W=C.W/'natural_response';old=C.old
SPEC=read_json(C.ROOT/'run.json')
HEIGHTS=[.2,.3,.45,.525];GROUPS=C.PART
METRICS=['flat_activity2','obstacle_activity2','natural_difference2','same_high_cue_difference2','same_zero_cue_difference2','flat_cue_effect2','obstacle_cue_effect2','interaction2']
@torch.no_grad()
def run(seed,smoke=False):
    started=time.time();index=pd.read_csv(W/'source_data/HISTORY_INDEX.csv').query('seed==@seed')
    phases=sorted(index.phase_seed.unique());phases=phases[:1] if smoke else phases
    assert len(phases)==(1 if smoke else 64)
    policy,model=old.load(seed,'connectome');core=policy.actor.core;N=C.N
    matrix=torch.zeros(N,len(GROUPS),device=C.DEVICE)
    counts=[]
    for j,k in enumerate(GROUPS):
        matrix[C.GROUPS[k],j]=1.;counts.append(len(C.GROUPS[k]))
    # Summed energy per annotation, normalized by group size only during analysis.
    output=np.full((len(phases),4,121,len(METRICS),len(GROUPS)),np.nan,dtype=np.float32)
    valid=np.zeros((len(phases),4,121),bool);errors=[];initial_matched=[]
    for start in range(0,len(phases),2):
        chosen=phases[start:start+2];B=len(chosen);arrays=[];positions=[];actions=[];lengths=[];pairlength=[]
        for phase in chosen:
            sub=index[index.phase_seed==phase].set_index('height_mm');obslist=[];poslist=[];actlist=[];nn=[]
            for height in [0,*HEIGHTS]:
                r=sub.loc[height]
                with np.load(r.trace) as z:
                    n=int(r.n_frames);nn.append(n)
                    obslist.append(z['observation'][:n,-172:].copy());poslist.append(z['position'][:n].copy());actlist.append(z['action'][:n].copy())
            length=nn[0];x=np.zeros((13,length,172),np.float32);pos=np.zeros((5,length,3),np.float32);act=np.zeros((5,length,6),np.float32)
            for j in range(5):
                m=min(length,nn[j]);x[j,:m]=obslist[j][:m];pos[j,:m]=poslist[j][:m];act[j,:m]=actlist[j][:m]
            for j,height in enumerate(HEIGHTS):
                x[5+j]=x[0];x[5+j,:,169]=height
                x[9+j]=x[1+j];x[9+j,:,169]=0
                # Identical initial mechanics: same-cue inputs must coincide at reset.
                assert np.array_equal(x[1+j,0],x[5+j,0])
                assert np.array_equal(x[9+j,0],x[0,0])
            arrays.append(x);positions.append(pos);actions.append(act);lengths.append(nn);pairlength.append([min(nn[0],nn[j+1]) for j in range(4)])
        Tmax=max(x.shape[1] for x in arrays)
        arr=np.zeros((B,13,Tmax,172),np.float32)
        for b,x in enumerate(arrays):arr[b,:,:x.shape[1]]=x
        obs=torch.as_tensor(arr.reshape(B*13,Tmax,172),device=C.DEVICE);h=core.init_state(B*13,C.DEVICE)
        err=0.
        for t in range(Tmax):
            for _ in range(4 if t==0 else 1):
                drive=core.fixed_graph(torch.tanh(h));drive[:,:2279]+=core.input_proj(obs[:,t]);h=.5*h+.5*torch.tanh(drive)
            rate=torch.tanh(h).reshape(B,13,N)
            flat=rate[:,0:1].expand(-1,4,-1);obstacle=rate[:,1:5];flatcue=rate[:,5:9];obszero=rate[:,9:13]
            values=torch.stack([flat.square(),obstacle.square(),(obstacle-flat).square(),(obstacle-flatcue).square(),(obszero-flat).square(),(flatcue-flat).square(),(obstacle-obszero).square(),(obstacle-flatcue-obszero+flat).square()],dim=2)
            energy=(values@matrix).cpu().numpy()
            for b in range(B):
                for j in range(4):
                    if t<pairlength[b][j]:output[start+b,j,t]=energy[b,j];valid[start+b,j,t]=True
            if t==0:initial_matched.append(float(energy[:,:,3:5].max()))
            if t%4==0:
                mean=old.mean_from_rates(core,rate[:,:5].reshape(B*5,N)).reshape(B,5,6).cpu().numpy()
                for b in range(B):
                    for j in range(5):
                        n=min(lengths[b][j],positions[b].shape[1]);stop=min(t+5,n)
                        if t+1>=stop:continue
                        u=np.clip((positions[b][j,t,0]-1.8)/.8,0,1);gate=u*u*(3-2*u)
                        cmd=np.tanh(mean[b,j])*gate
                        err=max(err,float(abs(actions[b][j,t+1:stop]-cmd).max()))
        assert err<2e-4,(seed,start,err)
        errors.append(err)
    assert max(initial_matched)==0
    path=W/'raw'/f'seed{seed}.npz'
    np.savez_compressed(path,energy=output,valid=valid,phases=phases,heights=HEIGHTS,groups=GROUPS,counts=counts,metrics=METRICS,time_s=np.arange(121)*.01)
    qa=dict(seed=seed,phases=len(phases),native_trajectories=5*len(phases),paired_contrasts=4*len(phases),max_saved_action_error=max(errors),same_cue_initial_difference2=max(initial_matched),seconds=time.time()-started,model=model,status='PASS')
    old.write(W/'qa'/f'replay_seed{seed}.json',qa)
    print(json.dumps({k:v for k,v in qa.items() if k!='model'}),flush=True)


def prepare():
    rows=[];pairs=[]
    for name in ('source_data','raw','qa'):(W/name).mkdir(parents=True,exist_ok=True)
    for item in read_json(C.ROOT/'evaluation/natural_histories.json'):
        with np.load(item['trace']) as z:
            times=z['time_s'];bad=np.flatnonzero(abs(times-np.arange(len(times))*.01)>1e-8)
            assert len(bad)==0 or np.array_equal(bad,[len(times)-1])
            n=len(times)-int(bool(len(bad)));ev=np.flatnonzero(z['feet'][:n,:,0].max(1)>=3.)
            row=dict(item,n_frames=n,partial_terminal_frame=bool(len(bad)),encounter_frame=int(ev[0]) if len(ev) else -1,initial_digest=__import__('hashlib').sha256(z['qpos'][0].tobytes()).hexdigest());rows.append(row)
    d=pd.DataFrame(rows)
    for (seed,phase),g in d.groupby(['seed','phase_seed']):
        g=g.set_index('height_mm');flat=g.loc[0.]
        for height in HEIGHTS:
            r=g.loc[height];assert r.initial_digest==flat.initial_digest
            pairs.append(dict(seed=seed,phase_seed=phase,height_mm=height,encounter_frame=int(r.encounter_frame),common_frames=min(r.n_frames,flat.n_frames)))
    d.to_csv(W/'source_data/HISTORY_INDEX.csv',index=False);pd.DataFrame(pairs).to_csv(W/'source_data/PAIRS.csv',index=False)
    pd.DataFrame([dict(group=k,n_nodes=len(C.GROUPS[k]),afferent=k in C.AFF) for k in GROUPS]).to_csv(W/'source_data/GROUP_COUNTS.csv',index=False)

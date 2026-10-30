"""Exact six-output adjoints at dense horizons; unit and activity-scaled states."""
from flycontrol.experiments.supp_s1.common import *
import argparse
@torch.no_grad()
def run(seed,smoke=False):
    t0=time.time();p,model=old.load(seed,'connectome');core=p.actor.core
    dest=W/'raw'/f'sensitivity_{seed}{"_smoke" if smoke else ""}';dest.mkdir(exist_ok=True)
    cc=cases(seed,smoke);idx={k:torch.as_tensor(v,device=DEVICE) for k,v in GROUPS.items()};rows=[];audits=[];maxerr=0
    for offset in range(0,len(cc),4):
        batch=cc[offset:offset+4];obs,gates,traces=batch_load(batch);base=old.native(core,obs);b=len(batch)
        maxerr=max(maxerr,replay_error(base,gates,traces,batch))
        for start,horizons in [(12,range(1,33)),(32,range(1,17))]:
            for lag in horizons:
                end=start+lag;gg=scan.output_gradient(core,base[end]['h'])
                for t in range(end,start,-1):
                    v=base[t]
                    gg=.5*gg+.5*v['prev_slope'][:,None,:]*scan.adjoint(core.fixed_graph,(gg*v['target_slope'][:,None,:]).reshape(b*6,N)).reshape(b,6,N)
                assert torch.isfinite(gg).all()
                for scale,e in [('unit_state',gg.square().sum(1)),('activity_scaled',gg.square().sum(1)*base[start]['h'].square())]:
                    energy=torch.stack([e.index_select(-1,idx[k]).sum(-1) for k in GROUPS],-1).cpu().numpy()
                    full=energy[:,:11].sum(-1);aff=energy[:,:9].sum(-1)
                    for i,c in enumerate(batch):
                        for j,(k,ii) in enumerate(GROUPS.items()):
                            value=float(energy[i,j]);nc=len(ii)
                            rows.append(dict(**labels(c),pulse_time_s=start*.01,horizon_ms=lag*10,scale=scale,category=k,n_nodes=nc,
                                energy=value,rms=np.sqrt(value/nc),full_share=value/full[i],
                                afferent_enrichment=value/nc/(aff[i]/18756) if k in AFF and aff[i]>0 else np.nan))
                if offset==0 and start==12 and lag in [1,8,32]:
                    rng=np.random.default_rng(161050+seed+lag);v=rng.normal(size=(b,N));v/=np.linalg.norm(v,axis=1,keepdims=True)
                    v=torch.as_tensor(v,device=DEVICE,dtype=torch.float64);analytic=torch.einsum('ban,bn->ba',gg.double(),v);errs=[]
                    for eps in [.01,.03]:
                        plus=scan.propagate(core,base[start]['h'].double()+eps*v,obs,start,end)
                        minus=scan.propagate(core,base[start]['h'].double()-eps*v,obs,start,end)
                        fd=(plus-minus)/(2*eps);errs.append(float(torch.linalg.vector_norm(fd-analytic)/torch.linalg.vector_norm(analytic).clamp(min=1e-20)))
                    assert min(errs)<.015,errs
                    audits.append(dict(lag_ms=lag*10,finite_difference_relative_errors=errs))
        save(dest/'STATUS.json',dict(processed=offset+b,total=len(cc),seconds=time.time()-t0))
    pd.DataFrame(rows).to_csv(dest/'SENSITIVITY.csv.gz',index=False)
    assert old.sha(model['path'])==model['sha256']
    save(dest/'COMPLETE.json',dict(seed=seed,histories=len(cc),rows=len(rows),native_replay_error=maxerr,fd=audits,model=model,model_unchanged=True,seconds=time.time()-t0))
    print('SENSITIVITY FINISHED',seed,len(cc),round(time.time()-t0,1),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();prepare()
    for seed in ([a.seed] if a.seed else range(2011,2017)):run(seed,a.smoke)

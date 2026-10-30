import numpy as np,pandas as pd
from . import speed_metrics as metrics
ID=['run_id','model','train_seed','eval_seed','case_id','family','protocol']
def validate_trace(f,r,p):
    assert len(f)>0 and np.all(np.diff(f.time_s)>=0),r['trace']
    assert (f.dt_s>=0).all() and np.allclose(f.dt_s,np.diff(np.r_[0.,f.time_s]),rtol=1e-7,atol=1e-8)
    zero=np.flatnonzero(f.dt_s.to_numpy()<=1e-12)
    # The frozen environment explicitly counts only successful physics steps.
    # A PhysicsError before the next step can therefore produce a zero-time
    # terminal event. Keep the failed case and charge all unobserved time.
    if len(zero):assert r['numerical_failure'] and list(zero)==[len(f)-1]
    assert abs(f.time_s.iloc[-1]-r['duration_s'])<1e-8
    assert f.time_s.iloc[-1]<=p['duration']+1e-8
    assert np.isfinite(f.target_mm_s).all() and f.target_mm_s.between(8-1e-8,16+1e-8).all()
    if r['full_duration']:
        assert not r['flip'] and not r['numerical_failure'] and r['duration_s']>=p['duration']-1e-8
        assert np.isfinite(f[['speed_mm_s','raw_speed_mm_s','drive','turn']]).all().all()
def push_metrics(pushed,control,profile):
    """Separate force response from the unforced response to the same setpoint."""
    a=pd.read_csv(pushed['trace']);b=pd.read_csv(control['trace'])
    a=a[a.dt_s>1e-12];b=b[b.dt_s>1e-12]
    onset=profile['push_onset'];end_force=profile['push_end'];horizon=profile['duration']
    r={k:pushed[k] for k in ID};r.update(force=profile['force'],direction=profile['direction'])
    r['both_survived_to_force']=min(pushed['duration_s'],control['duration_s'])>onset+1e-8
    r['both_full_duration']=pushed['full_duration'] and control['full_duration']
    assert pushed['initial_sha256']==control['initial_sha256'],r
    if r['both_survived_to_force']:
        assert pushed['prepush_sha256'] and pushed['prepush_sha256']==control['prepush_sha256'],r
    # Pair exact physics times, including a possible partial last decision on
    # failure; rounding a 1-ms final interval to a 10-ms tick duplicates keys.
    a=a.copy();b=b.copy();a['tick']=np.round(a.time_s/1e-4).astype(int);b['tick']=np.round(b.time_s/1e-4).astype(int)
    z=a.merge(b,on='tick',suffixes=('_push','_clean'),validate='one_to_one')
    regular=np.abs(z.time_s_push-z.time_s_clean)<1e-8
    z=z[regular];t=z.time_s_push.to_numpy();delta=(z.speed_mm_s_push-z.speed_mm_s_clean).to_numpy()
    before=(t<=onset+1e-8)&np.isfinite(delta)
    assert not before.any() or np.nanmax(abs(delta[before]))<1e-7,r
    post=(t>onset+1e-8)&np.isfinite(delta)
    r['observed_peak_delta_mm_s']=float(np.max(abs(delta[post]))) if post.any() else np.nan
    r['observed_delta_iae_mm']=float(np.sum(abs(delta[post])*z.dt_s_push.to_numpy()[post]))
    r['paired_observed_seconds']=float(z.dt_s_push.to_numpy()[post].sum())
    for band in [1.,2.]:
        s=metrics.settling(t,delta,end_force,horizon,r['both_full_duration'],band)
        for k,v in s.items():r[f'recovery_{band:g}_{k}']=v
    # Conditional recovery may be assessed only with a surviving matched control.
    # Failure-aware target MAE is reported for every pushed rollout separately.
    r['recovery_identifiable']=bool(control['full_duration'] and r['both_survived_to_force'])
    r['delta_iae_full_mm']=r['observed_delta_iae_mm'] if r['both_full_duration'] else np.nan
    for seconds in [.1,.2]:
        ca=metrics.cycle_mean(a,seconds);cb=metrics.cycle_mean(b,seconds)
        cz=ca[['tick','time_s','speed_mm_s']].merge(cb[['tick','time_s','speed_mm_s']],on='tick',suffixes=('_push','_clean'),validate='one_to_one')
        ct=cz.time_s_push.to_numpy();ce=(cz.speed_mm_s_push-cz.speed_mm_s_clean).to_numpy()
        prefix=f'cycle_mean_{round(seconds*1000)}ms_'
        for band in [1.,2.]:
            s=metrics.settling(ct,ce,end_force,horizon,r['both_full_duration'],band)
            for k,v in s.items():r[prefix+f'recovery_{band:g}_{k}']=v
    return r

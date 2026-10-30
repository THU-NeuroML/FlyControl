import numpy as np,pandas as pd,itertools
from pathlib import Path
from flycontrol.paths import sha256
def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out
def inference(a,l,rng):
    assert len(a)==len(l)==6 and np.isfinite(a).all() and np.isfinite(l).all()
    pool=np.r_[a,l];inds=np.array(list(itertools.combinations(range(12),6)));tot=pool[inds].sum(1);delta=l.mean()-a.mean();null=tot/6-(pool.sum()-tot)/6
    boot=l[rng.integers(0,6,(20000,6))].mean(1)-a[rng.integers(0,6,(20000,6))].mean(1);ci=np.quantile(boot,[.025,.975])
    return dict(adult_mean=a.mean(),adult_sd=a.std(ddof=1),larva_mean=l.mean(),larva_sd=l.std(ddof=1),difference=delta,ci_low=ci[0],ci_high=ci[1],p_raw=np.mean(abs(null)>=abs(delta)-1e-12),n_adult=6,n_larva=6)
def circ(x):return np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(x)))))
def summarize(epi,ramps,push,output):
    def save(name,d):d.to_csv(output/name,index=False)
    tails=[]
    for r in epi[epi.family.isin(['static','sine'])].to_dict('records'):
        path=Path(r['trace']);assert sha256(path)==r['trace_sha256'];x=pd.read_csv(path);use=x.time_s>1.+1e-9;err=np.abs(x.loc[use,'speed_mm_s']-x.loc[use,'target_mm_s'])
        tails.append(dict(model=r['model'],train_seed=r['train_seed'],eval_seed=r['eval_seed'],case_id=r['case_id'],error_p95_mm_s=np.quantile(err,.95) if len(err) else np.nan))
    tails=pd.DataFrame(tails);epi=epi.merge(tails,on=['model','train_seed','eval_seed','case_id'],how='left',validate='one_to_one')
    epi['absolute_final_heading_deg']=epi.heading_final_unwrapped_deg.abs();ramps['absolute_slope_error']=ramps.fitted_accel_error_mm_s2.abs()
    save('speed_episode_metrics.csv',epi);save('speed_ramp_segments.csv',ramps);save('speed_push_metrics.csv',push)
    units=[];definitions=[]
    def add(key,label,unit,frame,field,circular=False):
        q=frame.groupby(['model','train_seed'])[field].agg(circ if circular else lambda x: x.mean() if x.notna().all() else np.nan).reset_index(name='value');q=q.rename(columns={'model':'condition'});assert len(q)==12,(key,q)
        q['task']='speed';q['metric']=key;units.append(q);definitions.append(dict(task='speed',metric=key,label=label,unit=unit))
    static=epi[epi.family=='static']
    for k,l,field in [('static_mae','Fixed-target MAE','tracking_mae'),('static_rmse','Fixed-target RMSE','tracking_rmse'),('static_p95','Fixed-target 95th-percentile error','error_p95_mm_s')]:add(k,l,'mm/s',static,field)
    for case,label,key in [('legacy_step_up_+0','8→16 response t90','step_up_t90'),('legacy_step_down_+0','16→8 response t90','step_down_t90')]:
        q=epi[epi.case_id==case];assert len(q)==60;add(key,label,'s',q,'time_to_90_s')
    for direction in ['up','down']:
        q=ramps[(np.sign(ramps.requested_accel_mm_s2)==(1 if direction=='up' else -1))&(ramps.requested_accel_mm_s2.abs()>=5)]
        assert len(q)==240;add('ramp_'+direction+'_slope_error','Fast-ramp '+direction+' slope error','mm/s²',q,'absolute_slope_error')
    for f in [.25,.75]:
        q=epi[epi.case_id==f'sine_{f}'];assert len(q)==60
        for metric,label,unit,field,circular in [('gain','Amplitude gain','ratio','gain',False),('phase','Phase lag','degrees','phase_lag_deg',True),('p95','95th-percentile error','mm/s','error_p95_mm_s',False),('final_heading','Terminal absolute heading','degrees','absolute_final_heading_deg',False),('mean_heading','Mean absolute heading','degrees','heading_abs_mean_deg',False)]:
            add(f'sine_{f}_{metric}',f'{f} Hz: {label}',unit,q,field,circular)
    for force in [-15,15]:
        q=push[(push.direction=='down')&(push.force==force)];assert len(q)==60;add(f'push_down_{force}_recovery','Deceleration, force '+str(force)+': recovery','fraction',q,'recovery_2_reached')
    add('all_step_mae','All unperturbed steps: MAE','mm/s',epi[epi.family=='step'],'post_mae')
    assert len(definitions)==20
    u=pd.concat(units,ignore_index=True);save('speed_seed_values.csv',u);rng=np.random.default_rng(20260930);out=[]
    for d in definitions:
        q=u[u.metric==d['metric']];a=q[q.condition=='adult'].sort_values('train_seed').value.to_numpy();l=q[q.condition=='larva'].sort_values('train_seed').value.to_numpy();out.append(dict(d,**(inference(a,l,rng) if np.isfinite(a).all() and np.isfinite(l).all() else dict(adult_mean=float(np.nanmean(a)),larva_mean=float(np.nanmean(l)),p_raw=1.,non_estimable=True))))
    out=pd.DataFrame(out);out['p_holm']=holm(out.p_raw);out['family_size']=20;save('speed_inference.csv',out)
    # Keep every condition visible in source data even when the figure emphasizes Adult-favouring measurements.
    metrics=['tracking_mae','error_p95_mm_s','time_to_90_s','gain','phase_lag_deg','heading_abs_mean_deg','absolute_final_heading_deg']
    save('speed_all_conditions_seed_means.csv',epi.groupby(['model','train_seed','case_id'])[metrics].mean().reset_index())

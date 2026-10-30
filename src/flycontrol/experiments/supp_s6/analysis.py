"""S6 regional message substitution and recovery statistics."""
import numpy as np
import pandas as pd
from flycontrol.paths import read_json, write_json, sha256


def regional(root,spec,out):
    from flycontrol.experiments.supp_s6.metrics import load_raw, metrics, intervals
    rows=[];time=[]
    for p in sorted((root/'evaluation/regional/results').glob('*/episodes.json')):
        for row in read_json(p):
            r,curve=metrics(row,load_raw(row));rows.append(r)
            for i,t in enumerate(np.arange(1,51)/100):time.append(dict(condition=r['condition'],model_id=r['model_id'],signed_force=r['signed_force'],intervention=r['intervention'],time_s=t,**{k:float(v[i]) for k,v in curve.items()}))
    if not rows:raise ValueError('No usable regional branching states; inspect references/*/summary.json')
    d=pd.DataFrame(rows);d.to_csv(out/'regional_episodes.csv',index=False)
    key=['model_id','eval_seed','phase_bin','signed_force'];fields=['auc','success','flipped','survived_window','heading_only_failure','sustained_success_last100ms','tilt_auc','max_tilt_rad']
    baseline=d[d.intervention.eq('intact')][key+fields];paired=d.merge(baseline,on=key,suffixes=('','_intact'),validate='many_to_one')
    for f in fields:paired['delta_'+f]=paired[f].astype(float)-paired[f+'_intact'].astype(float)
    ds=['delta_'+f for f in fields];paired.to_csv(out/'regional_paired.csv',index=False)
    model=paired[paired.signed_force.ne(0)].groupby(['condition','model_id','intervention'])[ds].mean().reset_index();model.to_csv(out/'regional_seed_effects.csv',index=False)
    intervals(model,['condition','intervention'],ds).to_csv(out/'regional_summary.csv',index=False)
    pd.DataFrame(time).groupby(['condition','model_id','signed_force','intervention','time_s']).mean(numeric_only=True).to_csv(out/'regional_time_curves.csv')
    factorial=[]
    for (condition,model_id),q in model.groupby(['condition','model_id']):
        q=q.set_index('intervention')
        factorial.append(dict(condition=condition,model_id=model_id,**{m:float(q.loc['brain_vnc_reference',m]-q.loc['brain_reference',m]-q.loc['vnc_reference',m]) for m in ds}))
    f=pd.DataFrame(factorial);f.to_csv(out/'regional_factorial_seed_effects.csv',index=False)
    intervals(f,['condition'],ds).to_csv(out/'regional_factorial_interactions.csv',index=False)
    rng=np.random.default_rng(2026091718);contrasts=[]
    for mode in ['brain_reference','vnc_reference','brain_vnc_reference','replay_all_observations']:
        values=[model[model.condition.eq(c)&model.intervention.eq(mode)][ds].to_numpy() for c in ('banc_motor','banc_descending')]
        if not all(len(v) for v in values):continue
        a,b=values;ai=rng.integers(0,len(a),(20000,len(a)));bi=rng.integers(0,len(b),(20000,len(b)))
        boot=a[ai].mean(1)-b[bi].mean(1)
        for j,m in enumerate(ds):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);contrasts.append(dict(intervention=mode,metric=m,motor_minus_dn=a[:,j].mean()-b[:,j].mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(contrasts).to_csv(out/'regional_readout_interactions.csv',index=False)
    paired['both_survive']=paired.survived_window & paired.survived_window_intact
    paired['heading_auc_change_both_survive']=paired.delta_auc.where(paired.both_survive)
    paired[paired.signed_force.ne(0)].groupby(['condition','model_id','intervention']).agg(eligible_pairs=('both_survive','size'),both_surviving_pairs=('both_survive','sum'),conditional_heading_auc_change=('heading_auc_change_both_survive','mean')).to_csv(out/'regional_survivor_counts.csv')
    paths=sorted((root/'mechanisms/regional/neural_probe').glob('*.csv'))
    if paths:
        q=pd.concat([pd.read_csv(p) for p in paths]);q.to_csv(out/'regional_neural_probe.csv',index=False)

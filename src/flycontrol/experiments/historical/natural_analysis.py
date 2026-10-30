from pathlib import Path
import json,itertools,time,hashlib
import numpy as np,pandas as pd
from .common import W as BASE
from flycontrol.paths import read_json,run_root
W=BASE/'natural_response';S=W/'source_data'
SPEC=read_json(run_root()/'run.json')
pairs=pd.read_csv(S/'PAIRS.csv');groups=pd.read_csv(S/'GROUP_COUNTS.csv');names=groups.group.tolist();counts=groups.n_nodes.to_numpy();aff=groups.afferent.to_numpy(bool)
heights=[.2,.3,.45,.525];seeds=SPEC['seeds'];rel=np.arange(-10,21)
rng=np.random.default_rng(2026101607);boot=rng.integers(0,len(seeds),(20000,len(seeds)))
def interval(a):
 a=np.asarray(a,float);assert a.shape==(len(seeds),) and np.isfinite(a).all()
 return (float(a.mean()),np.nan,np.nan) if SPEC['smoke'] else (float(a.mean()),*np.quantile(a[boot].mean(1),[.025,.975]).tolist())
coverage=[];windows=[];aligned=[];clock=[];full=[]
for seed in seeds:
 z=np.load(W/'raw'/f'seed{seed}.npz');energy=z['energy'];phases=z['phases'];metric=z['metrics'].tolist();valid=z['valid'];assert energy.shape[0]==(1 if SPEC['smoke'] else 64)
 complete=np.zeros((len(phases),4),bool);a=np.full((len(phases),4,31,len(metric),len(names)),np.nan)
 for i,phase in enumerate(phases):
  for j,h in enumerate(heights):
   row=pairs.query('seed==@seed and phase_seed==@phase and height_mm==@h').iloc[0];event=int(row.encounter_frame)
   complete[i,j]=event>=10 and event+20<int(row.common_frames)
   for t,offset in enumerate(rel):
    ix=event+offset
    if ix>=0 and ix<121 and valid[i,j,ix]:a[i,j,t]=energy[i,j,ix]
   coverage.append(dict(seed=seed,height_mm=h,phase_seed=int(phase),encounter_ms=event*10,complete_window=complete[i,j],common_frames=int(row.common_frames)))
 for j,h in enumerate(heights):
  for selection in ['all_available','complete_window']:
   chosen=a[:,j] if selection=='all_available' else a[complete[:,j],j]
   e=np.nanmean(chosen,axis=0)
   for t,tm in enumerate(rel*10):
    n=int(np.isfinite(chosen[:,t,0,0]).sum())
    for k,group in enumerate(names):
     d=dict(seed=seed,height_mm=h,selection=selection,relative_time_ms=int(tm),group=group,n_pairs=n)
     for m,key in enumerate(metric):d[key]=float(e[t,m,k])
     aligned.append(d)
  # Absolute-clock contrast: all pairs valid at this time, separately report coverage.
  for t in range(121):
   n=int(valid[:,j,t].sum())
   if not n:continue
   e=np.nanmean(energy[:,j,t],axis=0)
   for k,group in enumerate(names):
    clock.append(dict(seed=seed,height_mm=h,time_ms=t*10,group=group,n_pairs=n,**{key:float(e[m,k]) for m,key in enumerate(metric)}))
  chosen=a[complete[:,j],j]
  for window,sel in [('pre',rel<0),('post',rel>=0)]:
   e=chosen[:,:, :, :][:,sel].mean((0,1))
   for k,group in enumerate(names):
    windows.append(dict(seed=seed,height_mm=h,window=window,group=group,n_pairs=int(complete[:,j].sum()),**{key:float(e[m,k]) for m,key in enumerate(metric)}))
 qa=json.loads((W/'qa'/f'replay_seed{seed}.json').read_text());assert qa['status']=='PASS'
for fname,rows in [('COVERAGE',coverage),('EVENT_PER_SEED',aligned),('CLOCK_PER_SEED',clock),('WINDOW_PER_SEED',windows)]:pd.DataFrame(rows).to_csv(S/(fname+'.csv'),index=False)
a=pd.DataFrame(aligned);win=pd.DataFrame(windows);cl=pd.DataFrame(clock)

def derive(df,keys):
 out=[]
 for labels,q in df.groupby(keys,sort=True):
  if not isinstance(labels,tuple):labels=(labels,)
  q=q.set_index('group').loc[names];e=q[metric].to_numpy().T
  d=dict(zip(keys,labels));globalflat=e[0].sum()
  for m,key in enumerate(metric):
   if m<2:continue
   globalrms=np.sqrt(e[m].sum()/globalflat)*100
   out.append(dict(**d,measurement='relative_global_rms_percent',contrast=key,group='whole_brain',value=globalrms))
   total=e[m,aff].sum()
   for k,group in enumerate(names):
    out.append(dict(**d,measurement='per_neuron_rms',contrast=key,group=group,value=np.sqrt(e[m,k]/counts[k])))
    if aff[k] and total>1e-30:
     out.append(dict(**d,measurement='log2_enrichment',contrast=key,group=group,value=np.log2(max(e[m,k]/counts[k],1e-30)/(total/counts[aff].sum()))))
  for k,group in enumerate(names):
   out.append(dict(**d,measurement='log2_activity_rms_ratio',contrast='obstacle_over_flat',group=group,value=.5*np.log2(max(e[1,k],1e-30)/max(e[0,k],1e-30))))
 return pd.DataFrame(out)
# Equal-weight height means are computed in energy space before ratios or roots.
aeq=a.query("selection=='complete_window'").groupby(['seed','relative_time_ms','group'])[metric].mean().reset_index()
weq=win.groupby(['seed','window','group'])[metric].mean().reset_index()
clockeq=cl.groupby(['seed','time_ms','group'])[metric].mean().reset_index()
ad=derive(aeq,['seed','relative_time_ms']);wd=derive(weq,['seed','window']);hd=derive(win,['seed','height_mm','window'])
for fname,df in [('EVENT_DERIVED_PER_SEED',ad),('WINDOW_DERIVED_PER_SEED',wd),('HEIGHT_DERIVED_PER_SEED',hd)]:df.to_csv(S/(fname+'.csv'),index=False)
summary=[]
for name,df,extra in [('event',ad,['relative_time_ms']),('window',wd,['window']),('height',hd,['height_mm','window'])]:
 keys=[*extra,'measurement','contrast','group']
 for labels,q in df.groupby(keys):
  q=q.sort_values('seed');assert q.seed.tolist()==seeds
  mean,lo,hi=interval(q.value.to_numpy())
  summary.append(dict(dataset=name,**dict(zip(keys,labels)),mean=mean,ci_low=lo,ci_high=hi,n_seeds=len(seeds)))
pd.DataFrame(summary).to_csv(S/'SUMMARY.csv',index=False)
report=dict(status='PASS',traces=len(pd.read_csv(S/'HISTORY_INDEX.csv')),paired_comparisons=len(pairs),n_training_seeds=len(seeds),smoke=SPEC['smoke'],complete_window_pairs=int(pd.DataFrame(coverage).complete_window.sum()),
 max_saved_action_error=max(json.loads((W/'qa'/f'replay_seed{s}.json').read_text())['max_saved_action_error'] for s in seeds),
 excluded_partial_terminal_frames=int(pd.read_csv(S/'HISTORY_INDEX.csv').partial_terminal_frame.sum()),
 summary_rows=len(summary),bootstrap_resamples=20000,seed=2026101607,
 no_training=True,no_new_physics=True,no_success_filtering=True,same_cue_initial_difference_exactly_zero=True)
(W/'qa/ANALYSIS_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
print(pd.DataFrame(summary).query("dataset=='window' and measurement=='relative_global_rms_percent'")[['window','contrast','mean','ci_low','ci_high']].to_string(index=False))
print(pd.DataFrame(summary).query("dataset=='window' and window=='post' and measurement=='log2_enrichment' and contrast=='same_high_cue_difference2'")[['group','mean','ci_low','ci_high']].to_string(index=False))

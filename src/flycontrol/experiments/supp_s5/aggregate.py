"""Training-model summaries; exhaustive cell annotations and paired statistics."""
from flycontrol.experiments.supp_s5.common import *
from scipy.stats import spearmanr
import itertools,gc
def interval(x):
    x=np.asarray(x,float);assert len(x)==len(read_json(ROOT/'run.json')['seeds']) and np.isfinite(x).all()
    if len(x)<2:return dict(mean=x.mean(),low=np.nan,high=np.nan,n=len(x),min=x.min(),max=x.max())
    r=np.random.default_rng(220922);b=r.choice(x,(20000,len(x))).mean(1)
    return dict(mean=x.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975),n=len(x),min=x.min(),max=x.max())
def signflip(x):
    x=np.asarray(x);stats=np.array([np.mean(x*np.array(s)) for s in itertools.product([-1,1],repeat=len(x))]);return float(np.mean(abs(stats)>=abs(x.mean())-1e-14))
def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty_like(p);out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out
def summarize(df,keys,metrics):
    rows=[]
    for kk,g in df.groupby(keys,dropna=False):
        if not isinstance(kk,tuple):kk=(kk,)
        for metric in metrics:
            rows.append(dict(zip(keys,kk),metric=metric,**interval(g[metric].to_numpy())))
    return pd.DataFrame(rows)
def main():
    state=[];inp=[];paths=[];cells=[];dns=[];corr=[];qa=[];native=[]
    for run in RUNS:
        folder=O/'models'/run['run_id'];meta=json.loads((folder/'COMPLETE.json').read_text());assert meta['status']=='PASS'
        s=pd.read_csv(folder/'state_sensitivity.csv.gz');ii=pd.read_csv(folder/'input_routes.csv.gz');pp=pd.read_csv(folder/'path_decomposition.csv.gz')
        common=['model','train_seed','E_stage','C_stage','channel','updates','task','phase','group']
        # Keep acceleration/deceleration separately, and register their equal-window post-switch pool.
        for frame in [s,ii,pp]:
            post=frame[(frame.task=='step')&(frame.phase!='before_switch')].copy();post['phase']='post_switch'
            frame_attrs=pd.concat([frame,post],ignore_index=True)
            if frame is s:s2=frame_attrs
            elif frame is ii:i2=frame_attrs
            else:p2=frame_attrs
        state.append(s2.groupby(common,dropna=False)[['energy_fraction','enrichment','mean_squared_derivative']].mean().reset_index())
        i2['component_gain_squared']=i2.gain_norm**2;i2['total_gain_squared']=i2.total_gain_norm**2;i2['dot_contribution']=i2.signed_share.fillna(0)*i2.total_gain_squared
        a=i2.groupby(common+['feature'],dropna=False)[['component_gain_squared','total_gain_squared','dot_contribution','signed_derivative','total_signed_derivative']].mean().reset_index()
        a['functional_share']=np.divide(a.dot_contribution,a.total_gain_squared,out=np.full(len(a),np.nan),where=a.total_gain_squared>1e-24);a['gain_norm']=np.sqrt(a.component_gain_squared);a['total_gain_norm']=np.sqrt(a.total_gain_squared);inp.append(a)
        paths.append(p2.groupby(common,dropna=False)[['one_edge_signed_share','multi_edge_signed_share','one_edge_squared_norm','total_squared_norm']].mean().reset_index())
        mp=mapping(run['model']);S=int((mp.partition=='sensory').sum());D=int((mp.partition=='output').sum());z=np.load(folder/'cell_energy.npz')
        for stage in ['initial','final']:
            for channel in ['drive','turn']:
                e=np.mean([z[f'{stage}__{channel}__step__{phase}'][:S].astype(float) for phase in ['acceleration','deceleration']],axis=0);e/=e.sum();n=mp.iloc[:S].copy();n['energy_fraction']=e
                for grouping in ['refined_group','sub_class','primary_type']:
                    keys=[grouping] if grouping=='refined_group' else ['refined_group',grouping]
                    for labels,g in n.groupby(keys):
                        labels=labels if isinstance(labels,tuple) else (labels,)
                        cells.append(dict(model=run['model'],train_seed=run['train_seed'],stage=stage,channel=channel,grouping=grouping,group=labels[0],label=labels[-1],n_cells=len(g),cell_fraction=len(g)/S,energy_fraction=g.energy_fraction.sum(),enrichment=g.energy_fraction.sum()/(len(g)/S)))
        ea=z['final__drive__step__acceleration'][:S].astype(float);ed=z['final__drive__step__deceleration'][:S].astype(float)
        corr.append(dict(model=run['model'],train_seed=run['train_seed'],comparison='acceleration_deceleration_sensory_energy',spearman=spearmanr(ea,ed).statistic))
        terms=np.load(folder/'output_pulse_terms.npz');windows=pd.read_csv(folder/'windows.csv');mask=(windows.task=='step')&(windows.phase!='before_switch');outmap=mp.iloc[-D:].reset_index(drop=True)
        for stage in ['initial','final']:
            for feature in ['target_speed','current_speed']:
                v=terms[f'{stage}__{feature}'][0,mask.to_numpy()].astype(float)
                av=np.mean(abs(v),axis=0);mass=av/av.sum();signed=v.mean(0);df=outmap[['state_index','root_id','primary_type','sub_class','refined_group','side']].copy()
                df['model']=run['model'];df['train_seed']=run['train_seed'];df['stage']=stage;df['feature']=feature;df['mean_abs_derivative']=av;df['mean_signed_derivative']=signed;df['absolute_contribution_fraction']=mass;dns.append(df)
        q=pd.read_csv(folder/'numerical_checks.csv');q['run_id']=run['run_id'];qa.append(q)
        del s,ii,pp,s2,i2,p2,z,terms;gc.collect();print('Aggregate',run['run_id'],flush=True)
    state=pd.concat(state,ignore_index=True);inp=pd.concat(inp,ignore_index=True);paths=pd.concat(paths,ignore_index=True);cells=pd.DataFrame(cells);dns=pd.concat(dns,ignore_index=True)
    for name,df in [('state_seed_metrics',state),('input_seed_metrics',inp),('path_seed_metrics',paths),('cell_type_seed_metrics',cells),('output_neuron_seed_metrics',dns),('phase_correspondence',pd.DataFrame(corr)),('numerical_checks',pd.concat(qa,ignore_index=True))]:df.to_csv(O/'source_data'/f'{name}.csv.gz',index=False)
    main=state[(state.task=='step')&(state.phase=='post_switch')&(state.updates==10)&(state.channel=='drive')]
    summary=summarize(main,['model','E_stage','C_stage','group'],['enrichment','energy_fraction','mean_squared_derivative']);summary.to_csv(O/'source_data/primary_neural_summary.csv',index=False)
    tests=[]
    for group,g in main[main.model.isin(['adult','er'])].groupby('group'):
        def vec(model,stage):return g[(g.model==model)&(g.E_stage==stage)&(g.C_stage==stage)].sort_values('train_seed').enrichment.to_numpy()
        af=vec('adult','final');ai=vec('adult','initial');ef=vec('er','final')
        for family,x in [('adult_final_vs_one',af-1),('adult_learning_change',af-ai),('adult_minus_er',af-ef)]:tests.append(dict(family=family,group=group,paired_difference=x.mean(),positive_models=int((x>0).sum()),p_two_sided=signflip(x),**interval(x)))
    tests=pd.DataFrame(tests)
    for family,idx in tests.groupby('family').groups.items():tests.loc[idx,'holm_q']=holm(tests.loc[idx,'p_two_sided'])
    tests.to_csv(O/'source_data/primary_neural_tests.csv',index=False)
    projection=pd.read_csv(O/'source_data/projection_group_metrics.csv');s=summarize(projection,['model','stage','feature','group'],['weight_fraction','weight_enrichment']);s.to_csv(O/'source_data/projection_summary.csv',index=False)
    # Input routing is a ratio of pooled dot products, with equal history/window weight within each model.
    a=inp[(inp.task=='step')&(inp.phase=='post_switch')&(inp.updates==10)&(inp.channel=='drive')&(inp.E_stage==inp.C_stage)]
    summarize(a,['model','E_stage','group','feature'],['functional_share','gain_norm','total_gain_norm','signed_derivative','total_signed_derivative'][:3]).to_csv(O/'source_data/input_primary_summary.csv',index=False)
    p=paths[(paths.task=='step')&(paths.phase=='post_switch')&(paths.channel=='drive')&(paths.E_stage==paths.C_stage)]
    summarize(p,['model','E_stage','group'],['one_edge_signed_share','multi_edge_signed_share']).to_csv(O/'source_data/path_primary_summary.csv',index=False)
    outputs=dns.groupby(['model','train_seed','stage','feature','primary_type']).agg(n_cells=('root_id','size'),absolute_contribution_fraction=('absolute_contribution_fraction','sum'),mean_signed_derivative=('mean_signed_derivative','sum')).reset_index();outputs.to_csv(O/'source_data/output_type_seed_metrics.csv',index=False)
    spec=read_json(ROOT/'run.json')
    write(O/'qa/AGGREGATE_COMPLETE.json',dict(status='PASS',smoke=spec['smoke'],models=len(RUNS),parameter_combinations=4*len(RUNS),windows_per_combination=len(windows),source='configured speed runs; initial/final and E/C hybrids',script_sha256=sha(__file__),formal_test_families=3,tests_per_family=9,small_n_exact_minimum_two_sided_p=2/2**len(spec['seeds'])))
    print(main[(main.E_stage=='final')&(main.C_stage=='final')].groupby(['model','group']).enrichment.mean().round(3).to_string(),flush=True)
if __name__=='__main__':main()

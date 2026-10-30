"""Supplemental statistics with model-level replication and retained failures."""
import numpy as np
import pandas as pd
from .paths import read_json,write_json,sha256,CONFIGS


def analyze(root,spec):
    if spec['family']=='development_terrain':
        return developmental_terrain(root,spec)
    if spec['family']=='supplemental':
        from .supplemental import worker
        worker(root,'analyze')
    else:
        out=root/'analysis';out.mkdir(exist_ok=True)
        paths=sorted((root/'mechanisms/historical/raw').glob('sensitivity_*/SENSITIVITY.csv.gz'))
        if not paths:raise FileNotFoundError('Historical state sensitivities have not been generated')
        d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        d.to_csv(out/'state_sensitivity.csv.gz',index=False)
        keys=['training_seed','pulse_time_s','horizon_ms','scale','category']
        means=d.groupby(keys)[['energy','rms','full_share','afferent_enrichment']].mean().reset_index();means.to_csv(out/'seed_sensitivity.csv',index=False)
        from .historical import worker
        worker(root,'natural_analysis')
        write_json(out/'manifest.json',dict(smoke=spec['smoke'],statistical_unit='training seed',sources={str(p.relative_to(root)):sha256(p) for p in paths}))


def analyze_local(root,spec):
    out=root/'analysis';out.mkdir(exist_ok=True)
    models=read_json(root/'training/models.json')
    for row in models:
        if sha256(root/row['checkpoint'])!=row['sha256']:raise ValueError('Checkpoint changed after evaluation')
    default=read_json(CONFIGS/(spec['experiment']+'.json'));formal=not spec['smoke'] and spec['methods']==default['methods'] and spec['seeds']==default['seeds']
    if spec['task']=='speed':
        from supplemental.speed_pairing import push_metrics,validate_trace
        profiles={p['case_id']:p for p in read_json(root/'configs/supplemental/speed_profiles.json')}
        episodes=[];segments=[];metas={};push=[]
        for model in models:
            folder=root/f"evaluation/speed/{model['method']}_seed{model['seed']}"
            assert read_json(folder/'complete.json')['checkpoint_sha256']==model['sha256']
            d=pd.read_csv(folder/'episodes.csv');episodes.append(d)
            p=folder/'segments.csv'
            if p.stat().st_size>1:segments.append(pd.read_csv(p))
            for p in folder.glob('eval_seed*/*.json'):
                if p.name=='HEARTBEAT.json':continue
                r=read_json(p);assert r['checkpoint_sha256']==model['sha256'] and sha256(r['trace'])==r['trace_sha256']
                validate_trace(pd.read_csv(r['trace']),r,profiles[r['case_id']]);metas[(r['model'],r['train_seed'],r['eval_seed'],r['case_id'])]=r
        for (m,s,e,c),r in metas.items():
            p=profiles[c]
            if p['family']=='push':
                control=metas[(m,s,e,f"legacy_step_{p['direction']}_+0")]
                push.append(push_metrics(r,control,p))
        epi=pd.concat(episodes,ignore_index=True);seg=pd.concat(segments,ignore_index=True) if segments else pd.DataFrame();push=pd.DataFrame(push)
        expected=len(spec['methods'])*len(spec['seeds'])*(8 if spec['smoke'] else 150)
        if len(epi)!=expected:raise ValueError(f'Incomplete speed grid: {len(epi)} / {expected}')
        epi.to_csv(out/'episodes.csv',index=False);seg.to_csv(out/'segments.csv',index=False);push.to_csv(out/'push_paired.csv',index=False)
        metrics=[c for c in epi.select_dtypes(include=['number','bool']) if c not in ('train_seed','eval_seed')]
        epi.groupby(['model','train_seed','case_id'])[metrics].mean().to_csv(out/'seed_means.csv')
        epi.groupby(['model','train_seed','case_id'])[metrics].count().to_csv(out/'observed_counts.csv')
        if formal and spec['experiment']=='development-speed':
            from supplemental.development_speed_summary import summarize
            summarize(epi,seg[seg.family.eq('ramp') & seg.protocol.eq('revised')].copy(),push,out)
        if spec['experiment']=='banc-sensitivity':
            from banc.aggregate import main
            main()
    elif spec['task']=='turning':
        rows=[]
        for task in ('direction','angle'):
            paths=sorted((root/f'evaluation/{task}').glob('*/episodes.csv'))
            d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);d['task']=task
            expected=len(spec['methods'])*len(spec['seeds'])*(1 if spec['smoke'] else 20)*(2 if task=='direction' else 6)
            if len(d)!=expected:raise ValueError('Incomplete developmental turning evaluation')
            rows.append(d);d.to_csv(out/f'{task}_episodes.csv',index=False)
        pd.concat(rows,ignore_index=True).to_csv(out/'episodes.csv',index=False)
        developmental_turning(root,spec,out,rows,formal)
    elif spec['task']=='heading':regional(root,spec,out)
    write_json(out/'manifest.json',dict(smoke=spec['smoke'],formal_cohort=formal,statistical_unit='training seed; evaluation repeats averaged within each model',checkpoints={r['checkpoint']:r['sha256'] for r in models},tables={p.name:sha256(p) for p in out.glob('*.csv*')}))


def developmental_turning(root,spec,out,frames,formal):
    from supplemental.development_speed_summary import inference
    from .analysis import holm
    definitions={'direction':[('turn_direction_accuracy',100),('target_aligned_heading_delta_rad',180/np.pi),('action_delta_l2_mean',1),('action_saturation_fraction',100)],'angle':[('heading_success',100),('final_heading_error_abs_rad',180/np.pi),('heading_error_abs_auc_rad_s',180/np.pi),('heading_gain_error_abs',1),('heading_overshoot',100),('heading_undershoot',100),('action_delta_l2_mean',1),('action_saturation_fraction',100)]}
    units=[];tests=[];rng=np.random.default_rng(20260930)
    for task,d in zip(('direction','angle'),frames):
        for field,scale in definitions[task]:
            q=d.groupby(['condition','train_seed'])[field].agg(lambda v:v.mean() if v.notna().all() else np.nan).reset_index(name='value');q.value*=scale;q['metric']=field;q['task']=task;units.append(q)
            if formal:
                a=q[q.condition=='adult'].sort_values('train_seed').value.to_numpy();l=q[q.condition=='larva'].sort_values('train_seed').value.to_numpy()
                info=inference(a,l,rng) if np.isfinite(a).all() and np.isfinite(l).all() else dict(p_raw=1.,non_estimable=True)
                tests.append(dict(task=task,metric=field,**info))
    pd.concat(units,ignore_index=True).to_csv(out/'seed_means.csv',index=False)
    if tests:
        # The manuscript's correction includes ten terrain endpoints. The joint
        # combine command replaces these conservative reserved slots with their p-values.
        q=pd.DataFrame(tests);q['p_holm22_reserved']=holm(np.r_[q.p_raw,np.ones(10)])[:len(q)];q['family_size']=22;q.to_csv(out/'turning_contrasts.csv',index=False)


def regional(root,spec,out):
    from supplemental.regional_metrics import load_raw,metrics,intervals
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


def developmental_terrain(root,spec):
    out=root/'analysis';out.mkdir(exist_ok=True)
    d=pd.read_csv(root/'evaluation/episodes.csv');expected=len(spec['methods'])*len(spec['seeds'])*(2 if spec['smoke'] else 800)
    if len(d)!=expected:raise ValueError('Incomplete developmental terrain grid')
    for row in read_json(root/'training/models.json'):
        assert sha256(root/row['checkpoint'])==row['sha256']
    key=['terrain','value_mm','evaluation_seed']
    for field in ('initial_qpos_sha256','geometry_sha256'):
        if not d.groupby(key)[field].nunique().eq(1).all():raise ValueError('Terrain physics pairing mismatch')
    d.to_csv(out/'episodes.csv',index=False)
    units=[]
    for terrain in ('gapped','blocks'):
        for field,scale in [('passed',100),('survived',100),('forward_distance_mm',1),('max_lateral_mm',1),('body_reaches_first_near_edge',100)]:
            q=d[d.terrain.eq(terrain)].groupby(['condition','train_seed'])[field].mean().reset_index(name='value');q.value*=scale;q['task']=terrain;q['metric']=field;units.append(q)
    pd.concat(units,ignore_index=True).to_csv(out/'seed_means.csv',index=False)
    write_json(out/'manifest.json',dict(smoke=spec['smoke'],statistical_unit='training seed',tables={p.name:sha256(p) for p in out.glob('*.csv')}))


def combine_development(terrain,turning,output):
    from .pipeline import load_run
    from .analysis import holm
    import itertools
    specs=[load_run(p) for p in (terrain,turning)]
    if [s['experiment'] for s in specs]!=['development-terrain','development-turning'] or any(s['smoke'] or s['seeds']!=list(range(6)) or s['methods']!=['adult','larva'] for s in specs):raise ValueError('The 22-test family requires both complete Adult/Larva six-seed cohorts')
    frames=[]
    for p in (terrain,turning):
        if not (p/'status/analyze.json').exists():raise ValueError('Complete both analyses before combining')
        manifest=read_json(p/'analysis/manifest.json')
        assert sha256(p/'analysis/seed_means.csv')==manifest['tables']['seed_means.csv']
        frames.append(pd.read_csv(p/'analysis/seed_means.csv'))
    d=pd.concat(frames,ignore_index=True);keys=d[['task','metric']].drop_duplicates();assert len(keys)==22
    indices=np.array(list(itertools.combinations(range(12),6)));rng=np.random.default_rng(20260930);rows=[]
    for task,metric in keys.itertuples(index=False,name=None):
        q=d[d.task.eq(task)&d.metric.eq(metric)]
        a=q[q.condition.eq('adult')].sort_values('train_seed').value.to_numpy();l=q[q.condition.eq('larva')].sort_values('train_seed').value.to_numpy();assert len(a)==len(l)==6
        finite=np.isfinite(a).all() and np.isfinite(l).all();diff=float(l.mean()-a.mean())
        if finite:
            pool=np.r_[a,l];sums=pool[indices].sum(1);null=sums/6-(pool.sum()-sums)/6
            p=float(np.mean(abs(null)>=abs(diff)-1e-12));b=l[rng.integers(0,6,(20000,6))].mean(1)-a[rng.integers(0,6,(20000,6))].mean(1);lo,hi=np.quantile(b,[.025,.975])
        else:p=1.;lo=hi=np.nan
        rows.append(dict(task=task,metric=metric,adult_mean=a.mean(),larva_mean=l.mean(),difference=diff,ci_low=lo,ci_high=hi,p_raw=p,non_estimable=not finite))
    out=pd.DataFrame(rows);out['p_holm22']=holm(out.p_raw);output.mkdir(parents=True,exist_ok=True);out.to_csv(output/'joint_22_contrasts.csv',index=False);d.to_csv(output/'seed_values.csv',index=False)
    write_json(output/'manifest.json',dict(family_size=22,exact_allocations=924,bootstrap_draws=20000,runs=[str(p) for p in (terrain,turning)]))

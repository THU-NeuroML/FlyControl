"""S3 Adult/larva terrain and turning statistics; joint 22-test family."""
import numpy as np
import pandas as pd
from flycontrol.paths import read_json, write_json, sha256


def developmental_turning(root,spec,out,frames,formal):
    from flycontrol.experiments.shared.statistics import inference
    from flycontrol.analysis import holm
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
    from flycontrol.pipeline import load_run
    from flycontrol.analysis import holm
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

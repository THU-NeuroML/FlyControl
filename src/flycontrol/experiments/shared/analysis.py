"""Supplemental statistics with model-level replication and retained failures."""
import pandas as pd
from flycontrol.paths import read_json, write_json, sha256, CONFIGS


def analyze(root, spec):
    if spec['family'] == 'development_terrain':
        from flycontrol.experiments.supp_s3.analysis import developmental_terrain
        return developmental_terrain(root, spec)
    if spec['family'] == 'supplemental':
        from flycontrol.experiments.shared.workflow import worker
        return worker(root, 'analyze')
    from flycontrol.experiments.supp_s1.analysis import analyze as sensitivity
    return sensitivity(root, spec)


def analyze_local(root,spec):
    out=root/'analysis';out.mkdir(exist_ok=True)
    models=read_json(root/'training/models.json')
    for row in models:
        if sha256(root/row['checkpoint'])!=row['sha256']:raise ValueError('Checkpoint changed after evaluation')
    default=read_json(CONFIGS/(spec['experiment']+'.json'));formal=not spec['smoke'] and spec['methods']==default['methods'] and spec['seeds']==default['seeds']
    if spec['task']=='speed':
        from flycontrol.experiments.shared.speed.pairing import push_metrics, validate_trace
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
            from flycontrol.experiments.supp_s4.analysis import summarize
            summarize(epi,seg[seg.family.eq('ramp') & seg.protocol.eq('revised')].copy(),push,out)
        if spec['experiment']=='banc-sensitivity':
            from flycontrol.experiments.supp_s5.aggregate import main
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
        from flycontrol.experiments.supp_s3.analysis import developmental_turning
        developmental_turning(root,spec,out,rows,formal)
    elif spec['task']=='heading':
        from flycontrol.experiments.supp_s6.analysis import regional
        regional(root,spec,out)
    write_json(out/'manifest.json',dict(smoke=spec['smoke'],formal_cohort=formal,statistical_unit='training seed; evaluation repeats averaged within each model',checkpoints={r['checkpoint']:r['sha256'] for r in models},tables={p.name:sha256(p) for p in out.glob('*.csv*')}))

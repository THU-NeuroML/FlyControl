"""Thirty waveform conditions, retaining failures and all raw histories."""
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import read_json,write_json,sha256


def evaluate(root,spec,method,seed):
    from supplemental import speed_policy as native, speed_evaluate as numerical, speed_metrics as metrics
    numerical.O=root/'configs/supplemental'
    cp=root/f'training/speed/{method}_seed{seed}/policy.pt'
    model=native.load(cp);check=native.replay_check(model)
    profiles=read_json(root/'configs/supplemental/speed_profiles.json')
    # Retain full waveform lengths in smoke; reduce only cohort and condition count.
    if spec['smoke']:profiles=[p for p in profiles if p['case_id'] in ('legacy_step_up_+0','legacy_step_down_+0','legacy_ramp_up','legacy_ramp_down')]
    if not profiles:raise ValueError('No supplemental speed profiles selected')
    seeds=[101003,101004] if spec['smoke'] else list(range(101000,101005))
    run=dict(run_id=f'{method}_seed{seed}',model=method,train_seed=seed)
    config=dict(model['cfg']['env']);config.update(terrains=['flat'],terrain_seeds=seeds,max_time=3.,perturbation={'type':'none'})
    env=numerical.EvalEnv(**config);env.axial_force=0.;env.impulse=0.;env.push_onset=1.5;env.push_end=1.54
    output=root/f'evaluation/speed/{method}_seed{seed}';output.mkdir(parents=True,exist_ok=True)
    rows,segments=[],[]
    try:
        for es in seeds:
            folder=output/f'eval_seed{es}';folder.mkdir(parents=True,exist_ok=True)
            for profile in profiles:
                meta_path=folder/(profile['case_id']+'.json')
                if meta_path.exists():
                    meta=read_json(meta_path)
                    if meta['checkpoint_sha256']!=sha256(cp) or sha256(meta['trace'])!=meta['trace_sha256']:raise ValueError('Supplemental evaluation changed')
                else:
                    meta=numerical.rollout(model,env,profile,es,folder,run)
                    write_json(meta_path,meta)
                trace=pd.read_csv(meta['trace'])
                row,details=metrics.score(trace,profile,meta);row.update(trace=meta['trace'],trace_sha256=meta['trace_sha256'])
                rows.append(row);segments.extend(details)
    finally:env.close()
    pd.DataFrame(rows).to_csv(output/'episodes.csv',index=False)
    pd.DataFrame(segments).to_csv(output/'segments.csv',index=False)
    write_json(output/'complete.json',dict(native_replay=check,episodes=len(rows),checkpoint_sha256=sha256(cp)))

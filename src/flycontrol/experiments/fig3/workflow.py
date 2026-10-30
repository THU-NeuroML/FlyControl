"""Registered turning controls and fresh shared-history neural assays."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
from flycontrol.paths import read_json, write_json, sha256
from flycontrol.mechanisms import run_job


def registered_train(root,spec,device):
    for seed in spec['seeds']:
        for method in spec['methods']:
            run_job(root,'turning','turning_push_train',seed,method,device)


def registered_evaluate(root,spec,workers):
    # The angle cohort is trained by the main workflow; push is a separate cohort.
    from flycontrol.pipeline import checkpoint
    steps=read_json(root/'protocol.json')['ppo']['total_timesteps']
    rows=[]
    for seed in spec['seeds']:
        for method in spec['methods']:
            for family in ('angle','push'):
                path=checkpoint(root,spec,method,seed) if family=='angle' else root/f'registered/runs/push_aligned_{method}_seed{seed}/policy_{steps:08d}.pt'
                rows.append(dict(family=family,method=method,seed=seed,path=str(path.relative_to(root)),sha256=sha256(path)))
    write_json(root/'data/figure3/registered_models.json',rows)
    for seed in spec['seeds']:
        for method in spec['methods']:
            run_job(root,'turning','turning_push_evaluate',seed,method,str(workers))
            if method in ('connectome','mlp'):run_job(root,'turning','turning_zero_evaluate',seed,method,str(workers))


def history(root,method,seed,phase,angle,output):
    import torch
    from turning_control.policy import build_policy
    from turning_control.task import PushTurnTask
    from flycontrol.experiments.fig3.paths import checkpoint
    cp=checkpoint(method,seed)
    policy=build_policy(None,64,2,method,seed,'cpu','normalized',-1.0)
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    policy.load_state_dict(payload['policy_state']);policy.eval()
    env=PushTurnTask(family='angle',objective='aligned',fixed_case=dict(angle_deg=float(angle),onset=.3))
    observations,actions,rates,micro=[],[],[],[]
    try:
        obs,initial=env.reset(seed=phase);h=policy.init_state(1,'cpu')
        with torch.no_grad():
            for t in range(50):
                observations.append(obs.copy())
                action,_,_,h=policy.act(torch.as_tensor(obs)[None],h,deterministic=True)
                actions.append(action[0].numpy().copy())
                if method=='connectome': rates.append(h[0,:18756].tanh().numpy().copy())
                obs,_,term,trunc,info=env.step(actions[-1]);micro.extend(env.last_micro_infos)
                if term or trunc: break
        output.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(observations=np.asarray(observations),actions=np.asarray(actions))
        if rates: arrays.update(rates=np.asarray(rates),selected_state_indices=np.arange(18756))
        np.savez_compressed(output,**arrays)
        write_json(output.with_suffix('.json'),dict(method=method,seed=seed,phase=phase,angle=angle,
            checkpoint_sha256=sha256(cp),sha256=sha256(output),initial=initial,frames=len(actions),physical_failure=bool(info['physical_failure'])))
        write_json(output.parent/'COMPLETE.json',dict(history=output.name,frames=len(actions)))
        return dict(seed=seed,phase=phase,angle=angle,path=str(output),frames=len(actions),physical_failure=bool(info['physical_failure']))
    finally: env.close()


def probe(root,spec,device):
    from flycontrol.mechanisms import link
    base=root/'data/figure3/mechanisms'
    bank=base/'landscape/flatness'
    protocol=read_json(root/'configs/figure3/input_gain.json')
    write_json(base/'input_gain/PROTOCOL.json',protocol)
    # Retain the published exploratory axes, and regenerate the full screen.
    write_json(base/'input_gain/AXIS_SELECTION.json',dict(axes=['Turn feedback','CPG'],
        origin='Manuscript exploratory choice; held fixed for this reproduction, not independent confirmation',
        selection_rule=protocol['selection']))
    run_job(root,'turning','turning_histories',spec['seeds'][0],device=device)
    for seed in spec['seeds']:
        for method in ('connectome','mlp'):
            run_job(root,'turning','turning_windows',seed,method,device)
        for assay in ('sensitivity','interface','activity','intervention'):
            run_job(root,'turning','turning_'+assay,seed,device=device)
        for method in ('connectome','mlp'):
            for assay in ('screen','surface'): run_job(root,'turning','turning_'+assay,seed,method,device)


def job(root,kind,method,seed,device):
    from flycontrol.experiments.fig3.paths import MECHANISMS as base, BANK as bank
    spec=read_json(root/'run.json')
    if kind=='graph':
        from flycontrol.experiments.fig3.registered.generate_graph import generate
        generate(seed,root/f'data/figure3/registered_graphs/er_seed{seed}.npz')
    elif kind=='push_train':
        steps=read_json(root/'protocol.json')['ppo']['total_timesteps']
        path=root/f'registered/runs/push_aligned_{method}_seed{seed}'
        if (path/'COMPLETE.json').exists(): return
        from flycontrol.experiments.fig3.registered.train import main
        sys.argv=['train','--method',method,'--family','push','--objective','aligned','--seed',str(seed),'--device',device]
        main()
    elif kind=='zero_evaluate':
        from turning_control.evaluate import evaluate
        from flycontrol.pipeline import checkpoint
        target=root/f'evaluation_zero/{method}_seed{seed}'
        if not (target/'COMPLETE.json').exists():
            evaluate(method,seed,target,checkpoint(root,spec,method,seed),workers=int(device),zero=True)
    elif kind=='push_evaluate':
        from flycontrol.experiments.fig3.registered.evaluate import main
        steps=read_json(root/'protocol.json')['ppo']['total_timesteps']
        sys.argv=['evaluate','--method',method,'--family','push','--objective','aligned','--seed',str(seed),'--steps',str(steps),'--phase-set','final','--workers',device]
        main()
    elif kind=='histories':
        # The original bank uses seed4301; the allowed smoke subset starts there.
        if seed!=4301: raise ValueError('The common turning bank requires training seed4301')
        for origin in ('connectome','mlp'):
            for angle in (-90,-60,-30,30,60,90):
                result=history(root,origin,seed,790000,angle,bank/f'input_{origin}_{angle:+d}.npz')
                if result['frames']!=50: raise RuntimeError('Prespecified turning history terminated early; no substitution or padding is allowed')
        rows=[]
        for training_seed in spec['seeds']:
            for phase in (790000,790101,790102):
                for angle in (-60,0,60):
                    output=base/f'activity/histories/seed{training_seed}_phase{phase}_angle{angle:+d}/trajectory.npz'
                    if angle==0: output=base/f'activity/controls/seed{training_seed}_phase{phase}/trajectory.npz'
                    rows.append(history(root,'connectome',training_seed,phase,angle,output))
        target=base/'activity/source_data';target.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(target/'turn_cohort.csv',index=False)
        for source,dest in [('group_inventory.csv','analysis_inventory.csv'),('group_membership.npz','analysis_membership.npz')]:
            shutil.copy2(root/'data/figure3/anatomy'/source,target/dest)
    elif kind=='windows':
        from flycontrol.experiments.fig3.mechanisms.prepare_windows import run
        run(method,seed,device,bank/f'{method}_{seed}')
    elif kind=='sensitivity':
        from flycontrol.experiments.fig3.mechanisms.sensitivity import run
        run(seed,device,base/f'sensitivity/models/connectome_{seed}')
    elif kind=='interface':
        from flycontrol.experiments.fig3.mechanisms.interface import run
        run(seed,device)
    elif kind=='activity':
        from flycontrol.experiments.fig3.mechanisms.activity import run
        run(seed,device,base/f'activity/models/seed{seed}')
    elif kind=='intervention':
        from flycontrol.experiments.fig3.mechanisms.intervention import worker
        for phase in ((790101,) if spec['smoke'] else (790101,790102)):
            for angle in ((60,) if spec['smoke'] else (-60,60)): worker((seed,phase,angle),base/'interventions/closed_loop')
    elif kind in ('screen','surface'):
        from flycontrol.experiments.fig3.mechanisms.input_gain import run
        run(method,seed,device,kind)
    else: raise ValueError(kind)

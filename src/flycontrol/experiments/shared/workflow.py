"""Independent supplemental cohorts with pinned data and fresh initialization."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import yaml
from flycontrol.paths import CONFIGS, read_json, write_json, sha256
from flycontrol.runtime import environment
from flycontrol.inputs import download, fetch


def tasks(spec):
    return ('direction','angle') if spec['task']=='turning' else (spec['task'],)


def worker(root,action,*args,device='cpu'):
    env=environment(root,'supplemental')
    env['FLYCONTROL_DEVICE']=device
    subprocess.run([sys.executable,'-m','flycontrol.experiments.shared.workflow',action,*map(str,args)],cwd=root,env=env,check=True)


def prepare(root,spec):
    for dataset,dest in [('flywire783','flywire783'),('banc626','banc626')]:
        if dataset=='banc626' and not any(m.startswith('banc') for m in spec['methods']): continue
        manifest=read_json(CONFIGS/f'inputs/{dataset}.json')
        download([dict(path=f'inputs/{dest}/{name}',url=r['source_url'],sha256=r['sha256'],bytes=r['bytes']) for name,r in manifest['files'].items()],root)
    if 'larva' in spec['methods']:
        record=read_json(CONFIGS/'inputs/larva.json');archive=fetch(record,root)
        with zipfile.ZipFile(archive) as z:
            for member,info in record['members'].items():
                data=z.read(member)
                if hashlib.sha256(data).hexdigest()!=info['sha256']: raise ValueError('Larval input checksum mismatch')
                (root/'inputs/larva'/info['path']).write_bytes(data)
    download(read_json(CONFIGS/'inputs/nmf_assets.json'),root)
    worker(root,'prepare')


def train(root,spec,device):
    for task in tasks(spec):
        for method in spec['methods']:
            for seed in spec['seeds']: worker(root,'train',task,method,seed,device=device)
    rows=[]
    for task in tasks(spec):
        for method in spec['methods']:
            for seed in spec['seeds']:
                path=root/f'training/{task}/{method}_seed{seed}/policy.pt'
                rows.append(dict(task=task,method=method,seed=seed,checkpoint=str(path.relative_to(root)),sha256=sha256(path)))
    write_json(root/'training/models.json',rows)


def evaluate(root,spec,workers):
    for row in read_json(root/'training/models.json'):
        if sha256(root/row['checkpoint'])!=row['sha256']: raise ValueError('Supplemental checkpoint changed')
        worker(root,'evaluate',row['task'],row['method'],row['seed'])


def prepare_local(root,spec):
    import pickle
    import numpy as np
    import torch
    from larva.locomotion.train import build_connectome
    from larva.locomotion.connectome import normalize_connectome
    from larva.locomotion.models import ValueHead
    contracts=read_json(root/'configs/supplemental/graphs.json')
    prepared={}
    for task in tasks(spec):
        for method in spec['methods']:
            for seed in spec['seeds']:
                config=yaml.safe_load((root/f'configs/supplemental/{task}/{method}_seed{seed}.yaml').read_text())
                c=config.get('connectome')
                key=(task,method,seed if method=='er' else None)
                if c:
                    for field in ('adult_dir','banc_dir'):
                        if field in c: c[field]=str(root/c[field])
                    if key not in prepared:
                        g=build_connectome(c)
                        if task=='speed' and method=='larva': g=normalize_connectome(g,'row_l1',1.0)
                        if method in contracts:
                            contract=contracts[method]
                            if [g.sensory_dim,g.internal_dim,g.output_dim]!=contract['dimensions']:raise ValueError('Supplemental graph partition mismatch')
                            actual={k:hashlib.sha256(getattr(g,k).tobytes()).hexdigest() for k in contract['arrays']}
                            if actual!=contract['arrays']:raise ValueError(f'Supplemental graph differs from manuscript contract: {method}; {actual}')
                        folder=root/'inputs/graphs';folder.mkdir(parents=True,exist_ok=True)
                        path=folder/f'{task}_{method}_{key[-1]}.pkl';path.write_bytes(pickle.dumps(g,protocol=5))
                        prepared[key]=(path,g)
                        arrays={k:getattr(g,k) for k in ('edge_pre','edge_post','edge_weight')}
                        if g.W is not None:
                            pre,post=np.nonzero(g.W);arrays=dict(edge_pre=pre,edge_post=post,edge_weight=g.W[pre,post])
                        np.savez_compressed(path.with_suffix('.npz'),**arrays,root_ids=np.asarray(g.sensory_ids+g.internal_ids+g.output_ids),sensory_dim=g.sensory_dim,internal_dim=g.internal_dim,output_dim=g.output_dim)
                    path,g=prepared[key];c.update(frozen_graph_path=str(path),frozen_graph_sha256=sha256(path))
                if config.get('initialization',{}).get('critic_state_from'):
                    path=root/f'initialization/critic_seed{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                    if not path.exists():
                        torch.manual_seed(40000+seed);critic=ValueHead(51,[128,128])
                        torch.save(dict(checkpoint_kind='initial',policy_state={'critic.'+k:v for k,v in critic.state_dict().items()},cfg={'seed':seed}),path)
                    config['initialization']['critic_state_from']=str(path)
                if spec['smoke']:
                    config['ppo'].update(total_timesteps=128,n_envs=2,n_steps=32,seq_len=8,n_epochs=1,n_minibatches=2)
                dest=root/f'training/configs/{task}/{method}_seed{seed}.yaml';dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(root/'inputs/supplemental.json',dict(configurations={str(p.relative_to(root)):sha256(p) for p in (root/'training/configs').rglob('*.yaml')},graphs={str(p.relative_to(root)):sha256(p) for p in (root/'inputs/graphs').glob('*')}))
    if spec['task']=='heading':
        from flycontrol.experiments.supp_s6.workflow import anatomy
        anatomy(root,spec)


def train_local(root,task,method,seed,device):
    from larva.locomotion.train import main
    folder=root/f'training/{task}/{method}_seed{seed}'
    receipt=folder/'training_receipt.json'
    if receipt.exists():
        for name,digest in read_json(receipt).items():
            if sha256(folder/name)!=digest:raise ValueError('Completed supplemental training output changed')
        return
    if (folder/'summary.json').exists():raise ValueError('Unsealed training directory; choose a new run directory')
    cfg=root/f'training/configs/{task}/{method}_seed{seed}.yaml'
    manifest=read_json(root/'inputs/supplemental.json')
    if sha256(cfg)!=manifest['configurations'][str(cfg.relative_to(root))]:raise ValueError('Resolved training configuration changed')
    sys.argv=['train','--config',str(cfg),'--output',str(folder),'--seed',str(seed),'--device',device]
    main()
    write_json(receipt,{p.name:sha256(p) for p in folder.glob('*.pt')})


def evaluate_local(root,spec,task,method,seed):
    if task=='speed':
        from flycontrol.experiments.shared.speed.workflow import evaluate
        evaluate(root,spec,method,seed)
    elif task in ('direction','angle'):
        from flycontrol.experiments.supp_s3.turning import evaluate
        evaluate(root, spec, task, method, seed)
    elif task=='heading':
        from flycontrol.experiments.supp_s6.workflow import evaluate
        evaluate(root,spec,method,seed)


def main():
    from flycontrol.paths import run_root
    root=run_root();spec=read_json(root/'run.json');action,*args=sys.argv[1:]
    if action=='prepare':prepare_local(root,spec)
    elif action=='train':train_local(root,args[0],args[1],int(args[2]),os.environ.get('FLYCONTROL_DEVICE','cpu'))
    elif action=='evaluate':evaluate_local(root,spec,args[0],args[1],int(args[2]))
    elif action=='regional_branch':
        from flycontrol.experiments.supp_s6 import evaluate as numerical
        from flycontrol.experiments.supp_s6.stable import StableReplay, stable_weights
        numerical.RegionalReplay=StableReplay;numerical.regional_weights=stable_weights
        job=read_json(args[0]);output=root/'evaluation/regional'
        numerical.evaluate(job,read_json(output/'protocol.json'),output/'results'/job['job_id'])
    elif action=='probe':
        if spec['experiment']=='banc-sensitivity':
            from flycontrol.experiments.supp_s5.prepare import prepare
            from flycontrol.experiments.supp_s5.analyze import run_one
            from flycontrol.experiments.supp_s5.common import RUNS, O
            import torch
            for name in ('source_data','models','qa'): (O/name).mkdir(parents=True,exist_ok=True)
            prepare()
            for run in RUNS: run_one(run,torch.device(os.environ.get('FLYCONTROL_DEVICE','cpu')))
        elif spec['task']=='heading':
            from flycontrol.experiments.supp_s6.workflow import probe
            probe(root,spec)
    elif action=='analyze':
        from flycontrol.experiments.shared.analysis import analyze_local
        analyze_local(root,spec)
    else:raise ValueError(action)

if __name__=='__main__':main()

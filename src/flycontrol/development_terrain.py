"""Six-seed Adult/Larva terrain controls for the joint S3 correction family."""
from pathlib import Path
import os,sys,subprocess,zipfile,hashlib
from .paths import read_json,write_json,sha256,CONFIGS
from .runtime import environment


def worker(root,action,*args,device='cpu'):
    env=environment(root,'supplemental' if action=='graph' else 'development');env['FLYCONTROL_DEVICE']=device
    subprocess.run([sys.executable,'-m','flycontrol.development_terrain',action,*map(str,args)],env=env,cwd=root,check=True)


def prepare(root,spec):
    from .inputs import prepare as inputs,download,fetch
    inputs(root,spec);download(read_json(CONFIGS/'inputs/nmf_assets.json'),root)
    record=read_json(CONFIGS/'inputs/larva.json');archive=fetch(record,root)
    with zipfile.ZipFile(archive) as z:
        for name,item in record['members'].items():
            data=z.read(name);assert hashlib.sha256(data).hexdigest()==item['sha256']
            (root/'inputs/larva'/item['path']).write_bytes(data)
    worker(root,'graph')


def train(root,spec,device):
    rows=[]
    for m in spec['methods']:
        for s in spec['seeds']:
            worker(root,'train',m,s,device=device)
            p=root/f'training/{m}_seed{s}/final_model.zip';normal=p.with_name('final_model_vecnormalize.pkl')
            rows.append(dict(method=m,seed=s,checkpoint=str(p.relative_to(root)),sha256=sha256(p),normalizer_sha256=sha256(normal)))
    write_json(root/'training/models.json',rows)


def main():
    from .paths import run_root
    import numpy as np
    root=run_root();spec=read_json(root/'run.json');action,*args=sys.argv[1:];protocol=read_json(root/'protocol.json')
    if action=='graph':
        import yaml
        from larva.locomotion.train import build_connectome
        from larva.locomotion.connectome import normalize_connectome
        conf=yaml.safe_load((root/'configs/supplemental/speed/larva_seed0.yaml').read_text())['connectome']
        g=normalize_connectome(build_connectome(conf),'row_l1',1.)
        if g.W is not None:
            pre,post=np.nonzero(g.W);weight=g.W[pre,post]
        else:pre,post,weight=g.edge_pre,g.edge_post,g.edge_weight
        np.savez_compressed(root/'inputs/larva_terrain.npz',edge_pre=pre,edge_post=post,edge_weight=weight,root_ids=np.asarray(g.sensory_ids+g.internal_ids+g.output_ids),sensory_dim=g.sensory_dim,internal_dim=g.internal_dim,output_dim=g.output_dim)
    elif action=='train':
        import torch
        import adult_experiment as adult,terrain_model as larval
        from stable_baselines3.common.logger import configure
        m,seed=args[0],int(args[1]);folder=root/f'training/{m}_seed{seed}'
        if (folder/'complete.json').exists():
            receipt=read_json(folder/'complete.json')
            if sha256(folder/'final_model.zip')!=receipt['checkpoint_sha256']:raise ValueError('Completed model changed')
            return
        if folder.exists():raise FileExistsError('Incomplete training exists; use a fresh run directory')
        (folder/'monitor').mkdir(parents=True);adult.configure_process();cfg=protocol['ppo']
        env=adult.build_vec(seed,folder,n_envs=cfg['n_envs'])
        try:
            if m=='adult':model=adult.build_model(seed,env,device=os.environ['FLYCONTROL_DEVICE'],n_steps=cfg['n_steps'])
            else:model=larval.build(m,seed,env,device=os.environ['FLYCONTROL_DEVICE'])
            if spec['smoke']:
                # Construct a correctly sized buffer, rather than mutating n_steps afterwards.
                from sb3_contrib import RecurrentPPO
                kwargs=model.policy_kwargs
                model=RecurrentPPO(adult.RecurrentCfCPolicy,env,n_steps=32,batch_size=32,n_epochs=1,seed=seed,device=os.environ['FLYCONTROL_DEVICE'],policy_kwargs=kwargs)
            model.set_logger(configure(str(folder),['csv']))
            before=larval.frozen_digest(model);model.learn(total_timesteps=cfg['total_timesteps'],progress_bar=False)
            assert before==larval.frozen_digest(model)
            model.save(folder/'final_model.zip');env.save(str(folder/'final_model_vecnormalize.pkl'))
            write_json(folder/'complete.json',dict(steps=model.num_timesteps,checkpoint_sha256=sha256(folder/'final_model.zip'),smoke=spec['smoke'],frozen_graph=before))
        finally:env.close()
    elif action=='evaluate':
        import pandas as pd
        import adult_experiment,terrain_model,ladder,evaluate_dt50us as ev
        ev.configure_process=lambda:None
        import torch
        torch.set_num_threads(1)
        rows=[]
        for model in read_json(root/'training/models.json'):
            source=root/model['checkpoint'];assert sha256(source)==model['sha256']
            assert sha256(source.with_name('final_model_vecnormalize.pkl'))==model['normalizer_sha256']
            for terrain,values in protocol['grid'].items():
                for level,value in enumerate(values[:1] if spec['smoke'] else values):
                    case=dict(terrain=terrain,level=level,parameter='gap_width_mm' if terrain=='gapped' else 'block_height_mm',value_mm=value,obstacle_start_x_mm=2.)
                    dest=root/f"evaluation/{model['method']}_seed{model['seed']}/{terrain}_{value:g}";dest.mkdir(parents=True,exist_ok=True)
                    for name in ('final_model.zip','final_model_vecnormalize.pkl'):
                        target=dest/name
                        if not target.exists():target.symlink_to(source.with_name(name))
                    ev.ManagedLegacyEnv=lambda **kw:ladder.DifficultyEnv(case,**kw)
                    seeds=protocol['evaluation_seeds'][:1] if spec['smoke'] else protocol['evaluation_seeds']
                    ev.evaluate_cases('connectome',model['seed'],[(terrain,s) for s in seeds],run_override=dest)
                    for s in seeds:
                        path=dest/f'evaluation/{terrain}_{s}/result.json';r=ladder.extend_result(path,case,'reproduction',model['method'],model['seed'])
                        rows.append(dict(r,condition=model['method'],train_seed=model['seed'],passed=int(r['body_beyond_first_far_edge_at_end_and_alive']),survived=int(r['survived_0_5_s']),terrain=terrain,value_mm=value,evaluation_seed=s,source=str(path),checkpoint_sha256=model['sha256']))
        pd.DataFrame(rows).to_csv(root/'evaluation/episodes.csv',index=False)
    else:raise ValueError(action)

if __name__=='__main__':main()

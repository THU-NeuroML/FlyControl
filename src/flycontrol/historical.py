"""Independent six-seed terrain protocol underlying Supplementary Fig. S1."""
from pathlib import Path
import os,sys,subprocess
from .paths import read_json,write_json,sha256
from .runtime import environment


def worker(root,action,*args,device='cpu'):
    env=environment(root,'historical');env['FLYCONTROL_DEVICE']=device
    subprocess.run([sys.executable,'-m','flycontrol.historical',action,*map(str,args)],env=env,cwd=root,check=True)


def train(root,spec,device):
    for m in spec['methods']:
        for s in spec['seeds']:worker(root,'train',m,s,device=device)
    steps=read_json(root/'protocol.json')['ppo']['total_timesteps']
    rows=[]
    for m in spec['methods']:
        for s in spec['seeds']:
            p=root/f'training/{m}_seed{s}/policy_{steps:08d}.pt'
            rows.append(dict(method=m,seed=s,checkpoint=str(p.relative_to(root)),sha256=sha256(p)))
    write_json(root/'training/models.json',rows)


def main():
    from .paths import run_root
    root=run_root();spec=read_json(root/'run.json');action,*args=sys.argv[1:]
    if action=='train':
        from historical_control.train import main as train_main
        sys.argv=['train','--method',args[0],'--seed',args[1],'--device',os.environ['FLYCONTROL_DEVICE']];train_main()
    elif action=='evaluate':
        from historical_control.evaluate import worker as evaluate_one
        import pandas as pd
        grid=read_json(root/'configs/historical/evaluation.json')
        if spec['smoke']:grid={**grid,'heights':grid['heights'][:1],'phases':grid['phases'][:1]}
        rows=[]
        for m in read_json(root/'training/models.json'):
            cp=root/m['checkpoint'];assert sha256(cp)==m['sha256']
            for h in grid['heights']:
                for phase in grid['phases']:
                    name=f"{m['method']}_seed{m['seed']}"
                    result=evaluate_one((name,str(cp),'blocks',h,phase,'final'))
                    trace=root/f'evaluation/final/{name}/blocks_{h:g}_seed{phase}/trace.npz'
                    rows.append(dict(training_seed=m['seed'],source_method=m['method'],height_mm=h,phase_seed=phase,trace=str(trace),trace_sha256=sha256(trace)))
        pd.DataFrame(rows).to_csv(root/'evaluation/history_index.csv',index=False)
        natural=[]
        phases=read_json(root/'configs/figure2/study.json')['phase_panels']['confirmation']
        if spec['smoke']:phases=phases[:1]
        for m in read_json(root/'training/models.json'):
            if m['method']!='connectome':continue
            cp=root/m['checkpoint'];name=f"connectome_seed{m['seed']}"
            for phase in phases:
                for height in [0,.2,.3,.45,.525]:
                    evaluate_one((name,str(cp),'blocks',height,phase,'final'))
                    path=root/f'evaluation/final/{name}/blocks_{height:g}_seed{phase}/trace.npz'
                    natural.append(dict(seed=m['seed'],height_mm=height,phase_seed=phase,trace=str(path),trace_sha256=sha256(path)))
        write_json(root/'evaluation/natural_histories.json',natural)
    elif action=='probe':
        from historical.sensitivity import run
        from historical.common import prepare
        prepare()
        for seed in spec['seeds']:run(seed)
        from historical import natural
        natural.prepare()
        for seed in spec['seeds']:natural.run(seed,spec['smoke'])
    elif action=='natural_prepare':
        from historical.natural import prepare
        prepare()
    elif action=='natural_analysis':
        import runpy
        runpy.run_module('historical.natural_analysis',run_name='__main__')
    else:raise ValueError(action)

if __name__=='__main__':main()

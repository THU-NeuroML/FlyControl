"""Seed-level analyses of generated neural assays, including complete test families."""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from .paths import read_json,write_json,sha256,CONFIGS
from .runtime import environment


def analyze(root,spec):
    if not spec.get('mechanisms'): return
    if not (root/'mechanisms/complete.json').exists():raise FileNotFoundError('Run the probe stage first')
    env=environment(root,spec['family'],'nmf','common')
    env['SPEED_ANALYSIS_ROOT']=str(root/'data/figure4')
    subprocess.run([sys.executable,'-m','flycontrol.mechanism_analysis'],env=env,cwd=root,check=True)


def collect(root,spec,source,pattern,output,**labels):
    paths=sorted(source.glob(pattern))
    if not paths:raise FileNotFoundError(f'No generated observations: {source/pattern}')
    frames=[]
    for p in paths:
        # Explicit metadata types avoid numeric inference on long hexadecimal hashes.
        frame=pd.read_csv(p,dtype={'initial_hash':str,'preintervention_hash':str,
                                  'physical_failure':str,'success':str,'restored':str})
        for k,func in labels.items():frame[k]=func(p)
        frames.append(frame)
    d=pd.concat(frames,ignore_index=True);d.to_csv(output,index=False)
    return d


def main():
    from .paths import run_root
    root=run_root();spec=read_json(root/'run.json');family=spec['family'];out=root/'analysis/neural';out.mkdir(parents=True,exist_ok=True)
    complete=read_json(root/'mechanisms/complete.json')
    if not complete.get('assays'):write_json(out/'complete.json',complete);return
    default=read_json(CONFIGS/(spec['experiment']+'.json'))
    formal=not spec['smoke'] and spec['seeds']==default['seeds'] and spec['methods']==default['methods']
    if family=='terrain':
        src=root/'data/figure2/mechanisms'
        for sub,file,name in [('coarse_neurons','sensitivity.csv','coarse'),('dense_neurons','sensitivity.csv','temporal'),('interventions','atlas_40ms.csv','atlas'),('interventions','interventions.csv','interventions'),('input_gain','surface.csv','input_gain'),('input_gain','axis_screen.csv','input_screen')]:
            collect(root,spec,src,f'{sub}/*/{file}',out/(name+'.csv'))
        if 'er' in spec['methods']:collect(root,spec,src,'er_atlas/*/atlas_40ms.csv',out/'er_atlas.csv')
        if formal:
            from fig2.current import temporal,circuit,er_atlas
            for m in (temporal,circuit,er_atlas):
                if hasattr(m,'S'):m.S=out
            temporal.Q=out;temporal.dense();circuit.R=out;(out/'qa').mkdir(exist_ok=True);circuit.main()
            # er_atlas writes a fixed source_data path; keep that explicit inside the run.
            (src/'source_data').mkdir(exist_ok=True);er_atlas.main()
            from .mechanism_statistics import terrain_comparisons,terrain_gain_comparisons
            terrain_comparisons(out);terrain_gain_comparisons(out)
    elif family=='speed':
        from fig4.analysis import mechanisms
        mechanisms(out)
        if formal:
            from fig4.analysis import tracking,ramp_increments
            from fig4.associations import analyze as associations
            tracking(out);ramp_increments(out);associations(out)
        d=pd.read_csv(out/'latest_neuron_class_seed_times.csv')
        # S2 uses mean(log2 enrichment); also expose log2(mean) so the convention is explicit.
        d['log2_enrichment']=np.log2(d.enrichment.where(d.enrichment>0))
        q=d.groupby(['phase','time_ms','group']).agg(mean_log2=('log2_enrichment','mean'),mean_enrichment=('enrichment','mean'),n=('seed','nunique')).reset_index();q['log2_mean']=np.log2(q.mean_enrichment);q.to_csv(out/'supplementary_S2.csv',index=False)
    elif family=='turning':
        src=root/'data/figure3/mechanisms'
        collect(root,spec,src,'interface/models/seed*/seed_metrics.csv',out/'interface.csv')
        collect(root,spec,src,'sensitivity/models/*/local.csv.gz',out/'sensitivity.csv.gz',seed=lambda p:int(p.parent.name.split('_')[-1]))
        raw=collect(root,spec,src,'interventions/closed_loop/*/episodes.csv',out/'interventions.csv')
        if formal:
            from fig3.behavior import analyze as behavior
            behavior(out/'registered_behavior')
            from fig3.activity_summary import extract,summarize
            extract(src/'activity',out/'activity.npz',out);summarize(out/'activity.npz',out)
            from fig3.mechanisms import summarize_interface
            summarize_interface.S=out/'interface';summarize_interface.S.mkdir(exist_ok=True);summarize_interface.main()
            import runpy
            os.environ['FIGURE3_ANALYSIS_OUTPUT']=str(out)
            runpy.run_module('fig3.mechanisms.summarize_input_gain',run_name='__main__')
            from .mechanism_statistics import turning_intervention_comparisons
            turning_intervention_comparisons(raw,out)
    tables={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and ('.csv' in p.suffixes or p.suffix=='.npz')}
    write_json(out/'complete.json',dict(formal_cohort=formal,smoke=spec['smoke'],assays=complete['assays'],tables=tables))

if __name__=='__main__':main()

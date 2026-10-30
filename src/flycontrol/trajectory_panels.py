"""Trace displays with prespecified selection and explicit post-termination handling."""
import numpy as np
import pandas as pd
from .paths import read_json,write_json,sha256


def plot(root,spec):
    import matplotlib.pyplot as plt
    out=root/'figures/trajectories';out.mkdir(parents=True,exist_ok=True);records=[]
    def save(d,x,y,group,name,xlabel,ylabel,*,equal=False):
        if d.empty:return
        fig,ax=plt.subplots(figsize=(6.5,3.7),layout='constrained')
        for key,q in d.groupby(group,sort=False):ax.plot(q[x],q[y],label=str(key).replace('_',' '),lw=1.2)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=name.replace('_',' ')+(' · SMOKE' if spec['smoke'] else ''))
        if equal:ax.set_aspect('equal',adjustable='datalim')
        ax.legend(frameon=False,fontsize=7);fig.supxlabel('Display only; see manifest for selection and termination handling.',fontsize=7)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=600)
        plt.close(fig);d.to_csv(out/f'{name}_source.csv',index=False);records.append(dict(name=name,source_sha256=sha256(out/f'{name}_source.csv')))
    family=spec['family'];seed=spec['seeds'][0]
    if family=='terrain':
        g=read_json(root/'evaluation/grid.json');height=max(g['heights']);phase=g['phases'][0];rows=[]
        for method in spec['methods']:
            if method=='er':continue
            p=root/f'evaluation/{method}_seed{seed}/h{height:g}_phase{phase}/trace.npz'
            with np.load(p) as z:
                rows.append(pd.DataFrame(dict(time_s=z['time_s'],forward_mm=z['position'][:,0]-z['position'][0,0],lateral_mm=z['position'][:,1]-z['position'][0,1],last_foot_mm=z['feet'][:,:,0].min(1),method=method)))
        d=pd.concat(rows);save(d,'time_s','last_foot_mm','method','F2_example_last_foot','Time (s)','Last foot world x (mm)');save(d,'forward_mm','lateral_mm','method','F2_example_path','Forward displacement (mm)','Lateral displacement (mm)',equal=True)
    elif family=='turning':
        phase=min(read_json(root/'protocol.json')['final_phase_seeds']);rows=[]
        for method in spec['methods']:
            if method=='er':continue
            p=next(p for p in sorted((root/f'evaluation/{method}_seed{seed}').glob('phase*.json')) if (lambda r:r['phase_seed']==phase and r['case']['angle_deg']==60)(read_json(p)))
            d=pd.DataFrame(read_json(p)['micro_trajectory']);d['method']=method;d['heading_error_deg']=np.rad2deg(d.heading_error_rad);rows.append(d[['elapsed_s','heading_error_deg','method']])
        save(pd.concat(rows),'elapsed_s','heading_error_deg','method','F3_plus60_example','Time (s)','Heading error (degrees)')
    elif family=='speed':
        cases=read_json(root/'configs/figure4/evaluation.json')['cases']
        for kind in ('ramps','sine'):
            case=next((c for c in cases if c['kind']==kind),None)
            if case is None:continue
            rows=[];target_added=False
            for method in spec['methods']:
                p=root/f"evaluation/{method}_seed{seed}/{kind}_c{case['command_seed']}_p{case['physics_seed']}.npz"
                with np.load(p) as z:
                    # Decimation only for display. All analysis uses every saved step.
                    for label,field in [(method,'filtered_body_forward_velocity_mm_s'),('target','target_applied_mm_s')]:
                        if label=='target' and target_added:continue
                        rows.append(pd.DataFrame(dict(time_s=z['elapsed_s'][::10],speed_mm_s=z[field][::10],series=label)))
                        if label=='target':target_added=True
            save(pd.concat(rows),'time_s','speed_mm_s','series','F4_'+kind+'_example','Time (s)','Forward speed (mm/s)')
    elif family=='body':
        rows=[]
        for method in spec['methods']:
            for training_seed in spec['seeds']:
                curves=[]
                for p in sorted((root/f'evaluation/{method}_seed{training_seed}').rglob('flat.npz')):
                    meta=read_json(p.with_suffix('.json'))
                    with np.load(p) as z:a=z['trajectory']
                    time=a[:,0];grid=np.arange(0,meta['horizon_s']+.001,.1)
                    xy=a[:,1:3]-a[0,1:3];curves.append(np.column_stack([np.interp(grid,time,xy[:,j]) for j in (0,1)]))
                if curves:
                    q=np.mean(curves,axis=0);rows.append(pd.DataFrame(dict(time_s=grid,forward_m=q[:,0],lateral_m=q[:,1],method=method,seed=training_seed)))
        if rows:
            d=pd.concat(rows);d.to_csv(out/'body_population_seed_paths.csv',index=False);mean=d.groupby(['method','time_s'])[['forward_m','lateral_m']].mean().reset_index()
            save(mean,'forward_m','lateral_m','method','F6_'+spec['experiment']+'_population_path','Forward displacement (m)','Lateral displacement (m)',equal=True)
            save(mean,'time_s','lateral_m','method','F6_'+spec['experiment']+'_lateral_trace','Time (s)','Lateral displacement (m)')
    elif family=='ant':
        for task in ('umaze','medium','large'):
            rows=[]
            for method in spec['methods']:
                p=root/f'evaluation/{method}_seed{seed}/{task}.json';r=read_json(p)[0]
                if 'trajectory_xy' not in r:continue
                xy=np.asarray(r['trajectory_xy']);rows.append(pd.DataFrame(dict(x_m=xy[:,0],y_m=xy[:,1],method=method)))
            if rows:save(pd.concat(rows),'x_m','y_m','method','F5_S8_'+task+'_paths','World x (m)','World y (m)',equal=True)
    write_json(out/'manifest.json',dict(panels=records,smoke=spec['smoke'],example_selection='First configured training seed and registered state; terrain uses highest evaluated height, turning +60°, speed first configured history of each kind, maze first reset.',body_population='Average all flat reset trajectories within each training seed, then average seeds equally. Final position held after termination for display only; no held pose enters metrics.',post_termination_examples='Stop at the recorded terminal frame.'))

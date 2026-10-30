"""Numerical paper panels from generated tables, with explicit seed aggregation."""
import re
import numpy as np
import pandas as pd
from .paths import read_json,write_json,sha256


def plot(root,spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=root/'figures/paper';out.mkdir(parents=True,exist_ok=True);records=[]
    def load(path):
        path=root/path
        if not path.exists():raise FileNotFoundError(f'Generate the analysis before plotting: {path}')
        return pd.read_csv(path,dtype={'initial_hash':str,'preintervention_hash':str,
                                       'physical_failure':str,'restored':str})
    def panel(d,x,y,series,unit,name,label,*,lines=False,outside_legend=False):
        required=[x,y,series,unit]
        if any(k not in d for k in required):raise ValueError(f'{name}: missing panel columns {set(required)-set(d)}')
        # Every mark is reduced within a training model before across-model means.
        finite=np.isfinite(pd.to_numeric(d[y],errors='coerce'));missing=int((~finite).sum())
        q=d.loc[finite].groupby(list(dict.fromkeys([series,unit,x])),dropna=False)[y].mean().reset_index()
        if q.empty:
            records.append(dict(name=name,status='no finite observations',omitted_rows=missing));return
        groups=list(q[series].unique());xs=sorted(q[x].unique()) if pd.api.types.is_numeric_dtype(q[x]) else list(q[x].unique())
        horizontal=not lines and (len(xs)>8 or max(map(lambda v:len(str(v)),xs))>23)
        fig,ax=plt.subplots(figsize=(9.0 if outside_legend else 7.1,max(3.9,len(xs)*.24+1.4) if horizontal else 3.9),layout='constrained')
        palette=plt.get_cmap('tab10')
        for j,key in enumerate(groups):
            a=q[q[series].eq(key)];color=palette(j%10)
            if lines:
                mean=a.groupby(x)[y].mean().reindex(xs);ax.plot(xs,mean,lw=1.25,ls=['-','--',':'][j//10%3],label=str(key).replace('_',' '),color=color)
            else:
                offset=(j-(len(groups)-1)/2)*.7/max(1,len(groups))
                for i,v in enumerate(xs):
                    vals=a[a[x].eq(v)][y].to_numpy();jitter=np.linspace(-.035,.035,len(vals))
                    pos=i+offset+jitter;label_key=str(key).replace('_',' ') if i==0 else None
                    ax.scatter(vals if horizontal else pos,pos if horizontal else vals,s=11,alpha=.55,color=color,linewidths=0,label=label_key)
                    if len(vals):
                        ticks=[i+offset-.065,i+offset+.065];means=[vals.mean()]*2
                        ax.plot(means if horizontal else ticks,ticks if horizontal else means,lw=1.5,color=color)
        if horizontal:
            ax.set_yticks(range(len(xs)),[str(v).replace('_',' ') for v in xs]);ax.invert_yaxis();ax.set_xlabel(label);ax.set_ylabel('')
            if name.startswith('S5_') and (q[y]>0).all():ax.set_xscale('log');ax.axvline(1,color='.75',lw=.7,zorder=0)
        else:
            if not lines:ax.set_xticks(range(len(xs)),[str(v).replace('_',' ').replace(' ','\n') for v in xs],rotation=0)
            ax.set_xlabel(x.replace('_',' '));ax.set_ylabel(label)
        n=q.groupby(series)[unit].nunique().to_dict();mode='SMOKE TEST: not paper results' if spec['smoke'] else 'Training-model aggregation'
        ax.set_title(name.replace('_',' ')+'\n'+mode,fontsize=9)
        if outside_legend:ax.legend(frameon=False,fontsize=6.5,loc='upper left',bbox_to_anchor=(1.01,1))
        else:ax.legend(frameon=False,fontsize=6.5,ncol=3 if len(groups)>4 else 2,loc='best')
        fig.supxlabel(('Lines: means of model means. ' if lines else 'Dots: model means; bars: across-model means. ')+f'Finite models per series: {min(n.values())}–{max(n.values())}; see source table for individual values.',fontsize=6)
        basename=re.sub(r'[^a-zA-Z0-9_\-]','_',name)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'{basename}.{ext}',dpi=600)
        plt.close(fig);q.to_csv(out/f'{basename}_source.csv',index=False)
        records.append(dict(name=name,status='exported',source=basename+'_source.csv',sha256=sha256(out/f'{basename}_source.csv'),omitted_nonfinite_rows=missing,model_counts=n))
    for folder,filename in [(root/'analysis','manifest.json'),(root/'analysis/neural','complete.json')]:
        if (folder/filename).exists():
            for name,digest in read_json(folder/filename).get('tables',{}).items():
                if sha256(folder/name)!=digest:raise ValueError('Analysis tables changed; rerun analysis before plotting')
    family=spec['family'];neural=root/'analysis/neural'
    if family=='terrain' and (neural/'complete.json').exists() and read_json(neural/'complete.json').get('assays'):
        d=load('analysis/neural/temporal.csv');panel(d,'horizon_ms','population_enrichment','group','seed','F2_temporal_sensitivity','Whole-network enrichment',lines=True)
        d=load('analysis/neural/atlas.csv');d['method']='Adult'
        if (neural/'er_atlas.csv').exists():e=load('analysis/neural/er_atlas.csv');e['method']='ER';d=pd.concat([d,e])
        panel(d,'group','enrichment','method','seed','F2_anatomical_atlas','40-ms enrichment')
        d=load('analysis/neural/interventions.csv');q=d.groupby(['seed','group','kind','replicate','dose'])[['delta_power_mean','baseline_power_mean']].mean().reset_index();q['relative_output_change']=100*np.sqrt(q.delta_power_mean/q.baseline_power_mean);q['series']=q['group']+' / '+q.kind
        panel(q,'dose','relative_output_change','series','seed','F2_reversible_interventions','Output change (%)',lines=True)
        gain_surfaces(root,spec,out,'terrain',records)
    elif family=='turning' and (neural/'interface.csv').exists():
        d=load('analysis/neural/interface.csv');d=d[d.C.eq(1)&d.horizon_ms.eq(100)&d.channel.eq('turn')]
        panel(d,'group','enrichment','variant','seed','F3_interface_exchange','100-ms turn enrichment')
        d=load('analysis/neural/sensitivity.csv.gz');inv=load('data/figure3/anatomy/group_inventory.csv');d=d.merge(inv[['group_id','level','label']],on='group_id');d=d[d.level.eq('input_class')]
        panel(d,'time_ms','enrichment','label','seed','F3_neural_sensitivity','Input-class enrichment',lines=True)
        d=load('analysis/neural/interventions.csv');d=d[d.force.eq(0)];base=d[d['mode'].eq('intact')][['seed','phase','angle','command_auc_rad_s']];d=d.merge(base,on=['seed','phase','angle'],suffixes=('','_intact'));d['delta_auc']=d.command_auc_rad_s-d.command_auc_rad_s_intact;d['cohort']='Adult'
        panel(d,'mode','delta_auc','cohort','seed','F3_AN_DN_interventions','Change in heading-error AUC (rad s)')
        gain_surfaces(root,spec,out,'turning',records)
        p=neural/'full_turn_activity_seed_values.csv'
        if p.exists():
            d=pd.read_csv(p);d['cohort']='Adult';panel(d,'group','enrichment','cohort','seed','F3_turn_activity','Matched-zero activity enrichment')
    elif family=='speed' and (neural/'latest_neuron_class_seed_times.csv').exists():
        d=load('analysis/neural/latest_neuron_class_seed_times.csv')
        for phase,q in d.groupby('phase'):panel(q,'time_ms','enrichment','group','seed','F4_'+phase+'_sensitivity','Input-class enrichment',lines=True)
        d=load('analysis/neural/latest_signed_class_seed_times.csv')
        for phase,q in d.groupby('phase'):panel(q,'time_ms','signed_response','group','seed','F4_'+phase+'_signed_response','Signed neural response',lines=True)
        d=load('analysis/neural/latest_event_class_ablation_seed.csv');panel(d,'source','fraction_of_full_target_response','phase','seed','F4_target_hold_interventions','Fraction of target response')
        if (neural/'ramp_increment_seeds.csv').exists():
            d=load('analysis/neural/ramp_increment_seeds.csv');panel(d,'window_start_s','delta_speed_mm_s','model','seed','F4_ramp_onset_response','Speed increment (mm/s)',lines=True)
        gain_surfaces(root,spec,out,'speed',records)
        d=load('analysis/neural/latest_neuron_class_seed_times.csv');d['log2_enrichment']=np.log2(d.enrichment.where(d.enrichment>0))
        for phase,q in d.groupby('phase'):panel(q,'time_ms','log2_enrichment','group','seed','S2_'+phase,'Mean log2 enrichment',lines=True)
    elif family=='historical':
        d=load('analysis/seed_sensitivity.csv')
        for (scale,pulse),q in d.groupby(['scale','pulse_time_s']):
            q=q.copy();q['log2_enrichment']=np.log2(q.afferent_enrichment.where(q.afferent_enrichment>0))
            panel(q,'horizon_ms','log2_enrichment','category','training_seed',f'S1_{scale}_pulse{pulse:g}','Log2 afferent enrichment',lines=True)
            panel(q[q.category.isin(['mechanosensory','AN','SA'])],'horizon_ms','rms','category','training_seed',f'S1_RMS_{scale}_pulse{pulse:g}','Per-neuron sensitivity (RMS)',lines=True)
        p=root/'mechanisms/historical/natural_response/source_data/EVENT_DERIVED_PER_SEED.csv'
        if p.exists():
            d=pd.read_csv(p)
            q=d[d.measurement.eq('relative_global_rms_percent')].copy()
            labels={'natural_difference2':'Obstacle vs flat: natural cues',
                    'same_high_cue_difference2':'Obstacle vs flat: common high cue',
                    'same_zero_cue_difference2':'Obstacle vs flat: common zero cue',
                    'flat_cue_effect2':'Cue effect: flat trajectory',
                    'obstacle_cue_effect2':'Cue effect: obstacle trajectory',
                    'interaction2':'Cue × trajectory interaction'}
            q['contrast']=q.contrast.map(labels)
            panel(q,'relative_time_ms','value','contrast','seed','Historical_obstacle_cue_exchange','Difference / flat activity RMS (%)',lines=True,outside_legend=True)

    elif family=='supplemental':
        if spec['task']=='turning':
            d=load('analysis/seed_means.csv')
            for (task,metric),q in d.groupby(['task','metric']):panel(q,'condition','value','condition','train_seed',f'S3_{task}_{metric}',metric.replace('_',' '))
        elif spec['task']=='speed':
            d=load('analysis/episodes.csv')
            for metric,label in [('tracking_mae','Tracking MAE (mm/s)'),('tracking_rmse','Tracking RMSE (mm/s)'),('time_to_90_s','Time to 90% (s)'),('gain','Sinusoid gain'),('phase_lag_deg','Phase lag (degrees)')]:
                if metric in d:panel(d,'case_id',metric,'model','train_seed',('S4_' if spec['experiment']=='development-speed' else 'BANC_behavior_')+metric,label)
            p=root/'analysis/speed_seed_values.csv'
            if p.exists():
                d=pd.read_csv(p)
                for metric,q in d.groupby('metric'):panel(q,'condition','value','condition','train_seed','S4_'+metric,metric.replace('_',' '))
            if spec['experiment']=='banc-sensitivity':
                d=load('mechanisms/banc/source_data/state_seed_metrics.csv.gz');d=d[d.task.eq('step')&d.phase.eq('post_switch')&d.updates.eq(10)&d.channel.eq('drive')&d.E_stage.eq(d.C_stage)]
                for m,q in d[d.model.str.startswith('banc')].groupby('model'):panel(q,'group','enrichment','E_stage','train_seed','S5_'+m,'100-ms drive enrichment')
        elif spec['task']=='heading':
            d=load('analysis/regional_seed_effects.csv')
            for metric in ['delta_auc','delta_success','delta_flipped']:panel(d,'intervention',metric,'condition','model_id','S6_'+metric,metric.replace('_',' '))
            d=load('analysis/regional_neural_probe.csv');d=d[d.force.ne(0)&d.start_s.eq(0)&d.stop_s.eq(.5)]
            for metric in ['action_change_rms','output_change_rms','action_response_change_norm_ratio']:panel(d,'mode',metric,'condition','model_id','S6_neural_'+metric,metric.replace('_',' '))
    elif family=='development_terrain':
        d=load('analysis/seed_means.csv')
        for metric,q in d.groupby('metric'):panel(q,'task','value','condition','train_seed','S3_family_terrain_'+metric,metric.replace('_',' '))
    elif family=='ant':
        d=load('analysis/seed_means.csv')
        for metric,q in d[d.task.eq('large')].groupby('metric'):panel(q,'method','value','method','seed','S8_Large_'+metric,metric.replace('_',' '))
        for task in ('umaze','medium','large'):
            episodes=load(f'analysis/{task}_episodes.csv')
            panel(episodes,'shortest_path_turns','success','method','training_seed',f'F5_S8_{task}_route_complexity','Success fraction',lines=True)
            rows=[]
            for deadline in np.arange(0,50.01,1):
                q=episodes[['method','training_seed']].copy();q['deadline_s']=deadline;q['success_by_deadline']=(episodes.success & episodes.time_to_goal_seconds.le(deadline)).astype(float);rows.append(q)
            panel(pd.concat(rows),'deadline_s','success_by_deadline','method','training_seed',f'F5_{task}_deadlines','Success fraction (all resets)',lines=True)
            rows=[]
            field='success_weighted_path_efficiency' if task=='umaze' else 'success_weighted_geodesic_efficiency'
            for threshold in np.linspace(0,1,21):
                q=episodes[['method','training_seed']].copy();q['efficiency_threshold']=threshold;q['success_above_threshold']=(episodes.success & episodes[field].ge(threshold)).astype(float);rows.append(q)
            panel(pd.concat(rows),'efficiency_threshold','success_above_threshold','method','training_seed',f'F5_{task}_efficiency_threshold','Success fraction (all resets)',lines=True)
        p=root/'analysis/supplementary_friction_episodes.csv'
        if p.exists():
            d=pd.read_csv(p)
            for metric in ['functional_failure','mean_x_velocity']:
                if metric in d:panel(d,'friction',metric,'method','training_seed','S7_'+metric,metric.replace('_',' '),lines=True)
    elif family=='body':
        d=load('analysis/episodes.csv');number={'tripod':'S9','hexapod':'S10'}.get(spec['experiment'],'F6_'+spec['experiment'])
        for metric in ['lateral_max_m','velocity_mps','corridor_025_time_s','corridor_050_time_s','corridor_075_time_s','corridor_100_time_s','success','horizon_speed_error_m_s']:
            if metric in d:panel(d,'condition',metric,'method','training_seed',number+'_'+metric,metric.replace('_',' '))
        if (root/'analysis/body_windows.csv').exists():
            d=load('analysis/body_windows.csv')
            for condition,q in d.groupby('condition'):
                for metric in ['lateral_drift_m_s','heading_rms_deg','speed_rmse_1s_m_s','speed_variability_1s_m_s','tilt_rms_deg']:
                    if metric in q:panel(q,'window',metric,'method','training_seed',number+'_'+condition+'_windows_'+metric,metric.replace('_',' '))
    if family in ('terrain','turning','speed','body','ant'):
        from .trajectory_panels import plot as trajectories
        trajectories(root,spec)
    write_json(out/'manifest.json',dict(paper=spec['paper'],smoke=spec['smoke'],panels=records,layout='One axis per file; no composite panel alignment required',inference='No new tests selected by these displays; see complete analysis families'))


def gain_surfaces(root,spec,out,family,records):
    import matplotlib.pyplot as plt
    surfaces=[]
    if family=='terrain':
        d=pd.read_csv(root/'analysis/neural/input_gain.csv');means=d.groupby(['method','x','y']).policy_kl.mean().reset_index()
        for m,q in means.groupby('method'):surfaces.append((m,q.pivot(index='y',columns='x',values='policy_kl').to_numpy(),[-.1,.1]))
    elif family=='turning':
        for m in ('connectome','mlp'):
            paths=[root/f'data/figure3/mechanisms/input_gain/models/{m}_{s}/surface.npz' for s in spec['seeds']]
            a=[np.load(p)['metrics'][...,0] for p in paths];surfaces.append((m,np.mean(a,axis=0),[-.1,.1]))
    else:
        z=np.load(root/'analysis/neural/latest_parameter_landscapes.npz')
        for i,m in enumerate(z['methods']):surfaces.append((str(m),z['metrics'][i,...,0].mean(0),[float(z['dose'][0]),float(z['dose'][-1])]))
    vmax=max(float(a.max()) for _,a,_ in surfaces)
    for m,a,limits in surfaces:
        fig,ax=plt.subplots(figsize=(4.8,4.1),layout='constrained');im=ax.imshow(a,origin='lower',extent=limits*2,vmin=0,vmax=vmax,cmap='viridis',interpolation='nearest',aspect='equal');fig.colorbar(im,ax=ax,label='Mean policy KL')
        labels={'terrain':('Velocity input gain','Joint-velocity input gain'),'turning':('Turn-feedback input gain','CPG input gain'),'speed':('Parameter direction 1','Parameter direction 2')}[family]
        ax.set(xlabel=labels[0],ylabel=labels[1],title=f'{family}: {m}'+(' · SMOKE' if spec['smoke'] else ''))
        name=f'{family}_{m}_gain_surface'
        for ext in ('pdf','svg','png'):fig.savefig(out/(name+'.'+ext),dpi=600)
        plt.close(fig);np.savez_compressed(out/(name+'_source.npz'),values=a,limits=limits,vmax=vmax);records.append(dict(name=name,status='exported',common_color_scale=[0,vmax],directly_evaluated_grid=list(a.shape)))

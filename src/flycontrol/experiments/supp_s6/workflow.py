"""BANC regional masks and complete within-checkpoint state branching."""
from pathlib import Path
import numpy as np
import pandas as pd
from flycontrol.paths import read_json, write_json, sha256


def anatomy(root,spec):
    raw=pd.read_csv(root/'inputs/banc626/connections_princeton.csv.gz',usecols=['pre_root_id','post_root_id','syn_count','neuropil'])
    ann=pd.read_csv(root/'inputs/banc626/neurons.csv.gz').rename(columns={'Root ID':'root_id','Super Class':'super_class','Class':'class'})
    region=np.where(raw.neuropil.str.startswith('VNC_'),'vnc',np.where(raw.neuropil.str.startswith('neck_'),'neck','brain'))
    counts=raw.syn_count.to_numpy(np.float64)
    out=root/'inputs/anatomy';out.mkdir(parents=True,exist_ok=True)
    for method in spec['methods']:
        z=np.load(root/f'inputs/graphs/heading_{method}_None.npz');ids=z['root_ids'];n=len(ids)
        nodes=pd.DataFrame(dict(root_id=ids)).merge(ann,on='root_id',how='left',validate='one_to_one').fillna('unassigned')
        mapping=pd.Series(np.arange(n),index=ids)
        mapped_pre=raw.pre_root_id.map(mapping);mapped_post=raw.post_root_id.map(mapping)
        valid=mapped_pre.notna() & mapped_post.notna()
        pre=mapped_pre[valid].to_numpy(np.int64);post=mapped_post[valid].to_numpy(np.int64)
        local_counts=counts[valid];local_region=region[valid]
        keys,inv=np.unique(pre*n+post,return_inverse=True);totals=np.bincount(inv,weights=local_counts)
        saved=z['edge_pre']*n+z['edge_post'];index=np.searchsorted(keys,saved)
        assert np.array_equal(keys[index],saved)
        fractions={r:(np.bincount(inv,weights=local_counts*(local_region==r),minlength=len(keys))/totals)[index].astype(np.float32) for r in ('brain','vnc','neck')}
        np.testing.assert_allclose(sum(fractions.values()),1,rtol=0,atol=2e-7)
        np.savez_compressed(out/f'{method}_regional_edge_fractions.npz',**{k:z[k] for k in ('edge_pre','edge_post','edge_weight','root_ids')},**{r+'_fraction':v for r,v in fractions.items()})
        cl=nodes.super_class.to_numpy()
        np.savez_compressed(out/f'{method}_annotated_neuron_masks.npz',vnc_intrinsic=cl=='ventral_nerve_cord_intrinsic',brain_intrinsic=np.isin(cl,['central_brain_intrinsic','optic_lobe_intrinsic']),descending=cl=='descending',ascending=cl=='ascending',leg_motor=nodes['class'].eq('leg_motor_neuron').to_numpy())


def evaluate(root,spec,method,seed):
    from larva.locomotion.scripts.eval_same_state_paired import evaluate_job
    protocol=read_json(root/'configs/supplemental/regional.json')
    protocol.update(source_sha256={},anatomy_directory=str(root/'inputs/anatomy'))
    if spec['smoke']: protocol.update(signed_forces=[0,-10,10],eval_seeds=[700],phase_bins=[0])
    import json,hashlib
    protocol['protocol_sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output=root/'evaluation/regional';write_json(output/'protocol.json',protocol)
    base=dict(protocol,first_snapshot_s=.75,last_snapshot_s=1.,max_time_s=1.7,interventions=['intact'])
    cp=root/f'training/heading/{method}_seed{seed}/policy.pt'
    for es in protocol['eval_seeds']:
        index=(spec['methods'].index(method)*len(spec['seeds'])+spec['seeds'].index(seed))*3+(es-700)
        model=f'{method}_seed{seed}';jobid=f'{model}_eval{es}'
        job=dict(model_id=model,condition=method,train_seed=seed,eval_seed=es,checkpoint=str(cp),checkpoint_sha256=sha256(cp),core_n=112885,job_id=jobid,job_index=index)
        folder=output/'references'/jobid
        evaluate_job(job,base,folder)
        for phase in protocol['phase_bins']:
            phase_dir=folder/f'phase_{phase}'
            if not phase_dir.exists():continue  # Unavailable states are retained by evaluate_job's QC.
            j=dict(job,job_id=f'{jobid}_phase{phase}',phase_bin=phase,source_phase_dir=str(phase_dir),snapshot_sha256=sha256(phase_dir/'snapshot.pkl.gz'))
            write_json(output/'jobs'/f"{j['job_id']}.json",j)
            from flycontrol.experiments.shared.workflow import worker
            worker(root,'regional_branch',output/'jobs'/f"{j['job_id']}.json")


def probe(root,spec):
    from flycontrol.experiments.supp_s6.probe import one
    for p in sorted((root/'evaluation/regional/jobs').glob('*eval700_phase0.json')):one(p)

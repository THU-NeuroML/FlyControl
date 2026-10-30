"""Same-state observation/message replay with intact recurrent dynamics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch

from larva.locomotion.paired_state import PairedSnapshot, _ENV_HANDLES, _FLY_RUNTIME, _plain_tree, state_digest
from larva.locomotion.scripts.eval_same_state_paired import (
    load_saved_policy, AuditedHeadingEnv, branch, atomic_json, sha256)
from larva.locomotion.scripts.eval_command2_policy import make_eval_env_config


OBS_GROUPS = {
    'replay_all_observations': list(range(51)),
    'replay_foot_position': list(range(18)),
    'replay_contact_force': list(range(18,24)),
    'replay_body_pose': list(range(24,28)),
    'replay_cpg_state': list(range(28,46)),
    'replay_heading_error': list(range(46,51)),
}
MESSAGE_MODES = ['replay_afferent_to_dn','replay_internal_to_dn','replay_dn_to_dn']


def model_hash(model):
    buffer = np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,None,buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def import_archive(env,path):
    with gzip.open(path,'rb') as f:
        a=pickle.load(f)
    assert model_hash(a['mujoco_data'].model)==model_hash(env._sim.physics.model.ptr)
    handles={k:env.__dict__[k] for k in _ENV_HANDLES if k in env.__dict__}
    env.__dict__.clear();env.__dict__.update(handles);env.__dict__.update(a['env'])
    snap=PairedSnapshot(a['mujoco_data'],env._sim.physics.model.ptr._address,a['env'],a['controller'],
        a['fly'],a['sim_time'],a['sim_rng'],a['observation'],torch.from_numpy(a['hidden'].copy()),
        a['numpy_rng'],a['python_rng'],a['torch_rng'],a['cuda_rng'],a['digest'])
    try:
        snap.restore(env)
    except RuntimeError as exc:
        if not str(exc).startswith('state restoration digest mismatch:'):
            raise
        for key in ['qpos','qvel','qacc','qacc_warmstart','sensordata','cfrc_ext','xpos','xquat',
                    'xfrc_applied','ctrl','qfrc_constraint']:
            assert np.array_equal(getattr(env._sim.physics.data.ptr,key),getattr(a['mujoco_data'],key)),key
        for saved,current in [(a['env'],{k:v for k,v in env.__dict__.items() if k not in _ENV_HANDLES}),
            (a['controller'],env._controller.__dict__),
            (a['fly'],_plain_tree({k:getattr(env._fly,k) for k in _FLY_RUNTIME if hasattr(env._fly,k)}))]:
            assert set(saved)==set(current)
            for key in saved:
                assert pickle.dumps(saved[key],protocol=5)==pickle.dumps(current[key],protocol=5),key
        snap.digest=state_digest(env,snap.observation,snap.hidden)
    return snap,a['digest']


class ReplayPolicy:
    def __init__(self,policy,mode='intact',baseline=None,edge_indices=None):
        self.policy=policy;self.actor=policy.actor;self.mode=mode;self.baseline=baseline
        self.index=0;self.inputs=[];self.hidden_before=[];self.capture=(baseline is None)
        self.edge_indices=edge_indices

    def act(self,obs,hidden,deterministic=True):
        observed=obs.clone()
        if self.capture:
            self.hidden_before.append(hidden[0].numpy().copy())
        if self.mode in OBS_GROUPS:
            ids=OBS_GROUPS[self.mode]
            obs=obs.clone();obs[:,ids]=torch.from_numpy(self.baseline['inputs'][self.index,ids])
        self.inputs.append(obs[0].numpy().copy())
        if self.index==0 and self.baseline is not None:
            assert torch.equal(obs,observed),'intervention changed onset input'
        result=self.policy.act(obs,hidden,deterministic=deterministic)
        self.index+=1
        return result

    @contextmanager
    def messages(self):
        core=self.actor.core
        if self.mode not in MESSAGE_MODES and self.edge_indices is None:
            yield;return
        pre,post=core.edge_pre,core.edge_post
        if self.edge_indices is not None:
            ids=torch.as_tensor(self.edge_indices,dtype=torch.long)
        else:
            target=post>=core.S+core.I
            source={'replay_afferent_to_dn':pre<core.S,
                    'replay_internal_to_dn':(pre>=core.S)&(pre<core.S+core.I),
                    'replay_dn_to_dn':pre>=core.S+core.I}[self.mode]
            ids=torch.nonzero(target&source,as_tuple=True)[0]
        ep,eo,ew=pre[ids],post[ids],core.edge_weight[ids]
        self.message_count=len(ids)
        original=core._recurrent_drive
        had_instance='_recurrent_drive' in core.__dict__
        previous_instance=core.__dict__.get('_recurrent_drive')
        def replay_drive(r):
            drive=original(r)
            ref=torch.tanh(torch.from_numpy(self.baseline['hidden_before'][self.index]))
            difference=(ref[ep].unsqueeze(0)-r[:,ep])*ew.unsqueeze(0)
            correction=torch.zeros_like(drive).index_add_(1,eo,difference)
            return drive+correction
        core._recurrent_drive=replay_drive
        try:
            yield
        finally:
            if had_instance:core._recurrent_drive=previous_instance
            else:del core._recurrent_drive


def extended_metrics(summary,arrays):
    times=arrays['relative_time_s'];dt=arrays['actual_step_duration_s']
    error=np.abs(arrays['heading_error_rad'])
    late=times>.4+1e-9
    summary['sustained_success_last100ms']=bool(summary['survived_window'] and late.any() and np.all(error[late]<=.25))
    summary['late100ms_max_error_rad']=float(error[late].max()) if late.any() else float(np.pi)
    summary['heading_bad_time_s']=float(np.sum(dt[error>.25])+summary['missing_duration_s'])
    return summary


def evaluate(job,protocol,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,h in protocol['source_sha256'].items():
        assert sha256(name)==h,('source changed',name)
    if (output/'complete.json').exists():
        assert json.loads((output/'complete.json').read_text())['protocol_sha256']==protocol['protocol_sha256']
        return
    torch.set_num_threads(1);torch.manual_seed(20260917)
    assert sha256(job['checkpoint'])==job['checkpoint_sha256']
    policy,cfg=load_saved_policy(Path(job['checkpoint']),int(job['core_n']))
    ec=make_eval_env_config(cfg,'heading_hold','flat',job['eval_seed'],1.7)
    ec['perturbation']={'type':'side_push','force':0.,'direction':'lateral','start_time_s':10.,'duration_s':.04}
    env=AuditedHeadingEnv(**ec)
    try:
        env.reset(seed=job['eval_seed'])
        phase_dir=Path(job['source_phase_dir'])
        snap,old_digest=import_archive(env,phase_dir/'snapshot.pkl.gz')
        reference_qc=json.loads((phase_dir/'replay_qc.json').read_text())
        collector=ReplayPolicy(policy)
        base,base_arrays=branch(env,collector,snap,0,'intact',.5,.04)
        assert base['trajectory_digest']==reference_qc['reference_digest'],'archived zero trajectory changed'
        baseline={'inputs':np.asarray(collector.inputs),'hidden_before':np.asarray(collector.hidden_before)}
        np.savez_compressed(output/'baseline_neural.npz',**baseline)
        old_rows=json.loads((phase_dir.parent/'episodes.json').read_text())
        rows=[];zero_exact=0;first_exact=0
        for f in protocol['signed_forces']:
            old=next(r for r in old_rows if r['phase_bin']==job['phase_bin'] and r['signed_force']==f and r['intervention']=='intact')
            with np.load(old['trace'],allow_pickle=False) as z:arr={k:z[k] for k in z.files}
            row=extended_metrics(dict(old),arr)
            row.update(protocol_sha256=protocol['protocol_sha256'],source_protocol_sha256=old['protocol_sha256'],
                       archive_digest=old_digest,prebranch_digest=snap.digest,condition='bio_original',
                       reused_intact_reference=True)
            rows.append(row)
        combinations=[(f,m) for f in protocol['signed_forces'] for m in protocol['modes']]
        order=np.random.default_rng(np.random.SeedSequence([2026091707,job['job_index']])).permutation(len(combinations))
        for i in order:
            f,mode=combinations[int(i)]
            proxy=ReplayPolicy(policy,mode,baseline)
            with proxy.messages():summary,arrays=branch(env,proxy,snap,f,'intact',.5,.04)
            assert np.array_equal(arrays['action'][0],base_arrays['action'][0]),(mode,'first action')
            first_exact+=1
            if f==0:
                assert summary['trajectory_digest']==base['trajectory_digest'],(mode,'non-noop zero replay')
                zero_exact+=1
            arrays['policy_input']=np.asarray(proxy.inputs)
            path=output/f'force_{f:+g}_{mode}.npz';np.savez_compressed(path,**arrays)
            summary=extended_metrics(summary,arrays)
            summary.update(**job,intervention=mode,trace=str(path.resolve()),
                protocol_sha256=protocol['protocol_sha256'],archive_digest=old_digest,reused_intact_reference=False,
                replayed_edge_count=getattr(proxy,'message_count',0))
            rows.append(summary);atomic_json(output/'episodes.json',rows)
            print(json.dumps({'job':job['job_id'],'force':f,'mode':mode,'wall_s':summary['wall_seconds']}),flush=True)
        repeated,_=branch(env,policy,snap,0,'intact',.5,.04)
        assert repeated['trajectory_digest']==base['trajectory_digest'],'intervention leaked'
        atomic_json(output/'complete.json',{'protocol_sha256':protocol['protocol_sha256'],'episodes':len(rows),
            'archived_reference_exact':True,'zero_replay_exact':zero_exact,'first_action_exact':first_exact,
            'final_reference_exact':True,'source_archive_digest':old_digest,'local_digest':snap.digest})
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(json.loads(a.job.read_text()),json.loads(a.protocol.read_text()),a.output)


if __name__=='__main__':main()

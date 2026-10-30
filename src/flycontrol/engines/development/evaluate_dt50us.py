"""Paired half-second evaluation; learned actions use the training environment."""
import argparse
import json
import time
import traceback
import hashlib
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from experiment import (ROOT, ManagedLegacyEnv, configure_process, write_json,
                        sha256, experiment_config)
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.rule_based_controller import RuleBasedController, construct_rules_graph
from dm_control.rl.control import PhysicsError

RB_WEIGHTS = {'rule1':-10, 'rule2_ipsi':2.5, 'rule2_contra':1,
              'rule3_ipsi':3., 'rule3_contra':2.}


def evaluate_cases(kind, train_seed, cases, run_override=None):
    configure_process()
    run=(Path(run_override) if run_override else
         ROOT/'runs'/f'{kind}_seed{train_seed:02d}' if kind in ['connectome','er']
         else ROOT/'baselines'/kind)
    env=None
    normalizer=None
    model=None
    if kind in ['connectome','er']:
        model=RecurrentPPO.load(run/'final_model.zip',device=os.environ.get('FLYCONTROL_DEVICE','cpu'))
        model.policy.set_training_mode(False)
        dummy=DummyVecEnv([lambda:ManagedLegacyEnv(randomize_terrain=False)])
        normalizer=VecNormalize.load(str(run/'final_model_vecnormalize.pkl'),dummy)
        normalizer.training=False
        normalizer.norm_reward=False
    results=[]
    try:
        for terrain,seed in cases:
            out=run/'evaluation'/f'{terrain}_{seed}'
            if (out/'result.json').exists():
                results.append(json.loads((out/'result.json').read_text()))
                continue
            out.mkdir(parents=True,exist_ok=True)
            env=ManagedLegacyEnv(terrain=terrain,terrain_seed=seed,randomize_terrain=False,
                                 max_time=.5,decision_interval=200,timestep=5e-5)
            obs,_=env.reset(seed=seed)
            torch.manual_seed(300000+seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(300000+seed)
            np.random.seed(seed)
            sim=env._sim
            initial_qpos=sim.physics.data.qpos.copy()
            geometry=hashlib.sha256(sim.physics.model.geom_pos.tobytes()+
                                    sim.physics.model.geom_size.tobytes()+
                                    sim.physics.model.geom_type.tobytes()).hexdigest()
            trace={k:[] for k in ['time_s','position','feet','orientation','up',
                                  'qpos','qvel','contact_forces','action']}
            def capture(action):
                raw=sim.fly.get_observation(sim)
                trace['time_s'].append(float(sim.curr_time))
                trace['position'].append(raw['fly'][0,:3].copy())
                trace['feet'].append(raw['end_effectors'].copy())
                trace['orientation'].append(raw['fly_orientation'].copy())
                trace['up'].append(raw['cardinal_vectors'][2].copy())
                trace['qpos'].append(sim.physics.data.qpos.copy())
                trace['qvel'].append(sim.physics.data.qvel.copy())
                trace['contact_forces'].append(raw['contact_forces'].copy())
                trace['action'].append(np.array(action,dtype=np.float32))
            capture(np.zeros(6))
            started=time.time();reason='time_limit';reward_sum=0.
            cfc_state=None
            episode_start=np.ones(1,dtype=bool)
            cfc_state_norms=[]
            if kind=='rule_based':
                steps=PreprogrammedSteps();steps.duration=1/12
                controller=RuleBasedController(timestep=5e-5,rules_graph=construct_rules_graph(),
                           weights=RB_WEIGHTS,preprogrammed_steps=steps,seed=seed)
                for tick in range(10000):
                    controller.step()
                    angles=[];adhesion=[]
                    for leg,phase in zip(controller.legs,controller.leg_phases):
                        angles.append(steps.get_joint_angles(leg,phase).flatten())
                        adhesion.append(steps.get_adhesion_onoff(leg,phase))
                    try:
                        raw,_,_,_,info=sim.step({'joints':np.concatenate(angles).astype(np.float32),
                                               'adhesion':np.array(adhesion,dtype=np.int32)})
                        if info.get('flip'): reason='flip'
                        elif not np.isfinite(raw['fly']).all(): reason='nan_detected'
                    except PhysicsError:
                        reason='physics_error'
                    if (tick+1)%200==0 or reason!='time_limit': capture(np.zeros(6))
                    if reason!='time_limit': break
            else:
                for _ in range(50):
                    if kind=='cpg': action=np.zeros(6,dtype=np.float32)
                    else:
                        normalized=normalizer.normalize_obs(obs[None,:])
                        action,cfc_state=model.predict(normalized,state=cfc_state,
                                      episode_start=episode_start,deterministic=True)
                        episode_start[:]=False
                        cfc_state_norms.append(float(np.linalg.norm(cfc_state[0])))
                        action=action[0]
                    obs,reward,terminated,truncated,info=env.step(action)
                    reward_sum+=reward
                    capture(action)
                    if terminated:
                        reason=('physics_error' if info.get('physics_error') else
                                'nan_detected' if info.get('nan_detected') else 'flip')
                        break
                    if truncated: break
            end_time=float(sim.curr_time)
            alive=reason=='time_limit' and end_time>=.5-1e-7
            arr={k:np.array(v) for k,v in trace.items()}
            if not all(np.isfinite(v).all() for v in arr.values()):
                alive=False;reason='nan_detected'
            p0,p1=arr['position'][0],arr['position'][-1]
            heading=np.array(arr['orientation'][0,:2],dtype=np.float64)
            heading/=max(np.linalg.norm(heading),1e-12)
            forward=float(np.dot(p1[:2]-p0[:2],heading))
            max_lateral=float(np.max(np.abs((arr['position'][:,:2]-p0[:2])@
                                          np.array([-heading[1],heading[0]]))))
            result={'condition':kind,'training_seed':train_seed if model else None,
                    'terrain':terrain,'evaluation_seed':seed,'survived_0_5_s':alive,
                    'reason':reason,'duration_s':end_time,'forward_distance_mm':forward,
                    'net_displacement_mm':float(np.linalg.norm(p1[:2]-p0[:2])),
                    'max_lateral_mm':max_lateral,'reward_sum':reward_sum if kind!='rule_based' else None,
                    'wall_s':time.time()-started,'geometry_sha256':geometry,
                    'initial_qpos_sha256':hashlib.sha256(initial_qpos.tobytes()).hexdigest(),
                    'evaluation_horizon_s':.5,'decision_interval':200,
                    'checkpoint':str(run/'final_model.zip') if model else None,
                    'normalization':'saved training statistics, frozen' if model else None,
                    'cfc_state_carried_across_decisions':True if model else None,
                    'cfc_state_norms':cfc_state_norms if model else None}
            # Diagnostic metadata only: distinguish numerical warning types without changing control.
            import mujoco
            warnings=sim.physics.data.warning
            result['physics_warning_counts']={name:{'count':int(warnings.number[int(index)]),
                'last_info':int(warnings.lastinfo[int(index)])}
                for name,index in mujoco.mjtWarning.__members__.items()
                if int(index)<len(warnings.number) and warnings.number[int(index)]>0}
            result['physics_dt_s']=float(sim.physics.model.opt.timestep)
            for key,value in list(result.items()):
                if isinstance(value,np.generic):
                    value=value.item()
                    result[key]=value
                if isinstance(value,float) and not np.isfinite(value): result[key]=None
            np.savez_compressed(out/'trace.npz',**arr)
            write_json(out/'result.json',result)
            results.append(result)
            env.close();env=None
    finally:
        if env is not None:env.close()
        if normalizer is not None:normalizer.close()
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['connectome','er','cpg','rule_based'],required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    run=(ROOT/'runs'/f'{args.kind}_seed{args.seed:02d}' if args.kind in ['connectome','er']
         else ROOT/'baselines'/args.kind)
    run.mkdir(parents=True,exist_ok=True)
    cfg=experiment_config()['evaluation']
    cases=[(terrain,seed) for terrain in cfg['terrains']
           for seed in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds_per_terrain'])]
    write_json(run/'EVALUATION_STARTED.json',{'pid':os.getpid(),'at_unix_s':time.time(),'episodes':len(cases)})
    try:
        rows=evaluate_cases(args.kind,args.seed,cases)
        if len(rows)!=160:raise RuntimeError('Incomplete evaluation')
        write_json(run/'EVALUATION_RESULTS.json',rows)
        write_json(run/'EVALUATION_FINISHED.json',{'episodes':len(rows),
                  'survived':sum(r['survived_0_5_s'] for r in rows),'at_unix_s':time.time(),
                  'results_sha256':sha256(run/'EVALUATION_RESULTS.json')})
    except BaseException:
        write_json(run/'EVALUATION_ERROR.json',{'at_unix_s':time.time(),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    raise SystemExit('Run worker.py for isolated difficulty outputs; direct execution is disabled.')

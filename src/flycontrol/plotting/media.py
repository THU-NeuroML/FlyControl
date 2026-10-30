"""Native simulator replays of newly generated evaluations (illustrations, not inference)."""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
from flycontrol.paths import CONFIGS, read_json, write_json, sha256
from flycontrol.runtime import environment


def render(root,number,output):
    from flycontrol.pipeline import load_run
    spec=load_run(root);recipe=next(r for r in read_json(CONFIGS/'videos.json') if r['number']==number)
    if recipe['config']!=spec['experiment']:raise ValueError(f"Video {number} requires the {recipe['config']} run")
    if not (root/'status/evaluate.json').exists():raise ValueError('Complete evaluation before rendering')
    family=spec['experiment'] if spec['family']=='body' else spec['family']
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','flycontrol.plotting.media',str(number),str(output.resolve())],cwd=root,env=environment(root,family,'nmf','common'),check=True)


class Movie:
    def __init__(self,path,label,fps):
        self.path=path;self.label=label;self.frames=0
        self.process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-threads','2','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
    def add(self,frame):
        import cv2
        frame=cv2.resize(np.asarray(frame),(640,480)).copy()
        cv2.rectangle(frame,(0,0),(640,28),(255,255,255),-1)
        cv2.putText(frame,self.label,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,40,55),1,cv2.LINE_AA)
        self.process.stdin.write(frame.astype(np.uint8).tobytes());self.frames+=1
    def close(self):
        self.process.stdin.close()
        if self.process.wait()!=0 or not self.frames:raise RuntimeError(f'Video export failed: {self.path}')


def fly(root,spec,recipe,out):
    import torch
    from dm_control.mujoco import engine
    from flycontrol.pipeline import checkpoint
    seed=spec['seeds'][0];family=spec['family'];records=[]
    if family=='turning':
        from turning_control.policy import build_policy
        from turning_control.task import PushTurnTask
        paths=sorted((root/f'evaluation/connectome_seed{seed}').glob('*.json'))
        cases=[read_json(p) for p in paths if p.name not in ('COMPLETE.json','SUMMARY.json')]
        cases=[r for r in cases if np.sign(r['case']['angle_deg'])==recipe['sign']]
        first=min(r['phase_seed'] for r in cases);cases=[r for r in cases if r['phase_seed']==first]
    else:
        from body_control.policy import build_policy
        from body_control.task import ContinuousSpeedTask
        paths=sorted((root/f'evaluation/connectome_seed{seed}').glob(recipe['kind']+'_*.json'))
        cases=[read_json(paths[0])] if paths else []
    if not cases:raise ValueError('Requested video conditions are absent from this evaluation subset')
    for method in ('connectome','mlp'):
        cp=checkpoint(root,spec,method,seed);payload=torch.load(cp,map_location='cpu',weights_only=False)
        policy=build_policy(None,64,2,method,seed,'cpu','normalized',-1.) if family=='turning' else build_policy(method,seed,'cpu',40)
        policy.load_state_dict(payload['policy_state']);policy.eval()
        for ci,case in enumerate(cases):
            env=PushTurnTask(family='angle',objective='aligned',fixed_case=case['case']) if family=='turning' else ContinuousSpeedTask(40,fixed_case=case['case'])
            movie=None;camera=None
            try:
                obs,initial=env.reset(seed=case['phase_seed']) if family=='turning' else env.reset()
                if initial['initial_state_sha256']!=case['initial']['initial_state_sha256']:raise ValueError('Replay initial state differs from evaluation')
                camera=engine.MovableCamera(env._sim.physics,height=480,width=640)
                movie=Movie(out/f'{method}_case{ci}.mp4',f"Video {recipe['number']} | {method} | seed {seed}"+(' | SMOKE' if spec['smoke'] else ''),25)
                h=policy.init_state(1,'cpu');actions=[]
                with torch.no_grad():
                    for _ in range(300):
                        a,_,_,h=policy.act(torch.as_tensor(obs)[None],h,deterministic=True)
                        obs,_,term,trunc,info=env.step(a[0].numpy());actions.append(a[0].numpy().tolist())
                        pos=env._sim.physics.bind(env._fly.thorax).xpos
                        camera.set_pose(distance=10,lookat=pos,azimuth=135,elevation=-30);movie.add(camera.render())
                        if term or trunc:break
                records.append(dict(method=method,seed=seed,case=case['case'],checkpoint_sha256=sha256(cp),frames=len(actions),physical_failure=bool(info['physical_failure']),initial=initial))
            finally:
                if movie:movie.close()
                del camera
                env.close()
    return records


def saved_states(root,spec,recipe,out):
    import mujoco
    seed=spec['seeds'][0];records=[]
    for method in ('connectome','mlp'):
        if spec['family']=='terrain':
            from flycontrol.experiments.fig2.current.steady_task import SteadyObstacleTask
            grid=read_json(root/'evaluation/grid.json');phase=grid['phases'][0]
            jobs=[(h,root/f'evaluation/{method}_seed{seed}/h{h:g}_phase{phase}/trace.npz') for h in grid['heights'] if h>0]
            if not jobs:raise ValueError('Obstacle videos need nonzero heights; use a full evaluation')
        else:
            jobs=[(recipe['condition'],p) for p in sorted((root/f'evaluation/{method}_seed{seed}').rglob(recipe['condition']+'.npz'))[:1]]
            if not jobs:raise ValueError('Requested body condition is absent from this evaluation subset')
        for name,p in jobs:
            movie=None;renderer=None
            if spec['family']=='terrain':
                env=SteadyObstacleTask(fixed_cell=('blocks',name),capture_micro_trace=True);env.reset(seed=phase);physics=env._sim.physics;model,data=physics.model.ptr,physics.data.ptr
                z=np.load(p);times=z['time_s'];qpos=z['qpos'];qvel=z['qvel'];distance=10.
            else:
                from body_control import env as module
                init=read_json(next((root/'training/runs').glob(f'*{method}*seed{seed}/initialization.json')))
                options={} if init.get('body_interface','native')=='native' else dict(interface=init['body_interface'])
                factory=getattr(module,'make_env',module.TaskEnv);env=factory(spec['experiment'],**options)
                meta=read_json(p.with_suffix('.json'));env.reset(seed=meta['evaluation_seed']);model,data=env.model,env.data
                z=np.load(p);a=z['trajectory'];times=a[:,0];qpos=a[:,1:1+meta['nq']];qvel=a[:,1+meta['nq']:1+meta['nq']+meta['nv']];distance=2.5
            try:
                renderer=mujoco.Renderer(model,height=480,width=640);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=distance;cam.azimuth=135;cam.elevation=-25
                movie=Movie(out/f'{method}_{name}.mp4',f"Video {recipe['number']} | {method} | {name}"+(' | SMOKE' if spec['smoke'] else ''),25)
                for t in np.arange(times[0],times[-1]+1e-8,.04):
                    i=int(np.argmin(abs(times-t)));data.qpos[:]=qpos[i];data.qvel[:]=qvel[i];mujoco.mj_forward(model,data)
                    cam.lookat[:]=qpos[i,:3];renderer.update_scene(data,camera=cam);movie.add(renderer.render())
                records.append(dict(method=method,seed=seed,condition=name,trace=str(p),trace_sha256=sha256(p),frames=movie.frames,simulation_steps=0))
            finally:
                if movie:movie.close()
                if renderer:renderer.close()
                env.close()
    return records


def ant(root,spec,recipe,out):
    from body_control import evaluate as flat, evaluate_maze as maze
    from flycontrol.experiments.fig5.workflow import stage_dir, ENVIRONMENTS
    task=recipe['task'];seed=spec['seeds'][0];protocol=read_json(root/'protocol.json');records=[]
    module=flat if task=='flat' else maze;attribute='ContinualAntEnv' if task=='flat' else 'AntMazeTransferEnv';original=getattr(module,attribute)
    for method in (spec['methods'] if recipe.get('seven') else ['connectome','mlp']):
        movie=Movie(out/f'{method}.mp4',f"Video {recipe['number']} | {task} | {method}"+(' | SMOKE' if spec['smoke'] else ''),20)
        def factory(*args,**kwargs):
            kwargs['render_mode']='rgb_array';env=original(*args,**kwargs);step=env.step
            def capture(action):
                result=step(action);movie.add(env.render());return result
            env.step=capture;return env
        setattr(module,attribute,factory)
        try:
            run=stage_dir(root,method,seed,task)
            if task=='flat':rows=flat.evaluate_run(run,[1.],episodes=1,seed=protocol['evaluation_seeds'][task],device='cpu',checkpoint='best',action_scale=.9)
            else:rows=maze.evaluate_antmaze_run(run,ENVIRONMENTS[task],episodes=1,seed=protocol['evaluation_seeds'][task],device='cpu',checkpoint='best',waypoint_planner='official_grid_waypoints' if task=='umaze' else 'none',continuing_task=False,action_smoothing=.8 if task=='umaze' else .95,action_scale=.8 if method=='connectome' and task!='umaze' else 1.)
            records.append(dict(method=method,seed=seed,checkpoint_sha256=sha256(run/'best_checkpoint.pt'),evaluation=rows))
        finally:setattr(module,attribute,original);movie.close()
    return records


def main():
    from flycontrol.paths import run_root
    root=run_root();spec=read_json(root/'run.json');number=int(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    recipe=next(r for r in read_json(CONFIGS/'videos.json') if r['number']==number)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    function=ant if spec['family']=='ant' else saved_states if spec['family'] in ('body','terrain') else fly
    records=function(root,spec,recipe,out)
    write_json(out/'manifest.json',dict(recipe=recipe,smoke=spec['smoke'],selection='First configured training seed and first available registered initial state; no outcome screening or seed replacement',role='Illustrative native replay; separate clips per method and condition; not cohort evidence or original video editing',records=records,files={p.name:sha256(p) for p in out.glob('*.mp4')}))
if __name__=='__main__':main()

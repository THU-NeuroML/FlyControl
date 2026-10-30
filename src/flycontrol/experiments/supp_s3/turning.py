"""S3 matched Adult/larva direction and angle evaluation."""
from flycontrol.paths import sha256


def evaluate(root, spec, task, method, seed):
    import torch
    from larva.locomotion.train import build_connectome
    from larva.locomotion.models import build_policy
    from larva.locomotion.scripts.eval_command2_policy import evaluate_policy_run,write_csv
    path=root/f'training/{task}/{method}_seed{seed}/policy.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False);cfg=payload['config'];pc=cfg['policy']
    graph=build_connectome(cfg['connectome'])
    policy=build_policy(pc['name'],payload['obs_dim'],payload['action_dim'],actor_kwargs=pc['actor_kwargs'],value_hidden=pc['value_hidden'],log_std_init=pc.get('log_std_init',-.5),connectome=graph)
    policy.load_state_dict(payload['policy_state']);policy.eval()
    seeds=[100] if spec['smoke'] else list(range(100,120))
    rows=evaluate_policy_run(model=method,run_dir=path.parent,policy=policy,cfg=cfg,task='steering_target' if task=='direction' else 'reorientation',terrains=['flat'],seeds=seeds,steering_targets=[-1.,1.],target_angles_rad=[-1.57079632679,-1.0471975512,-.5235987756,.5235987756,1.0471975512,1.57079632679],target_speeds_mm_s=[],device='cpu',max_time=None,hidden_init_mode='zero')
    for row in rows:row.update(condition=method,train_seed=seed,checkpoint_sha256=sha256(path))
    output=root/f'evaluation/{task}/{method}_seed{seed}';output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'episodes.csv',rows)

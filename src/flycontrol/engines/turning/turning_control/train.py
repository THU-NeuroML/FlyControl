"""Fresh 8,192-decision training; no selection or continuation changes."""

import json, time, traceback
from functools import partial
from .context import DATA, PROTOCOL, write_json, configure_process, sha
from .policy import build_policy, graph_path
from .task import make_env, OBS_DIM, ACTION_DIM
from .audited_ppo import AuditedPPO, tensor_hash
from fly_control.ppo import PPOConfig
from fly_control.vector_env import SubprocVecEnv


def train(method, seed, output, device="cuda:0"):
    configure_process()
    conf = json.loads(PROTOCOL.read_text())
    folder = output
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    config = dict(
        conf["ppo"],
        seed=seed,
        initialization_seed=seed,
        environment_seed=5_000_000 + seed * 100,
        policy_sampling_seed=100000 + seed,
        minibatch_seed=300000 + seed,
    )
    cfg = PPOConfig(**config)
    initial_log_std = float(conf.get("policy", {}).get("initial_log_std", -1.0))
    policy = build_policy(
        DATA / "graph.npz",
        OBS_DIM,
        ACTION_DIM,
        method,
        seed,
        device,
        "normalized",
        initial_log_std,
    )
    graph_before = None
    if method in ("connectome", "er"):
        graph_before = tensor_hash(
            [
                policy.actor.core.edge_pre,
                policy.actor.core.edge_post,
                policy.actor.core.edge_weight,
            ]
        )
    write_json(
        folder / "initialization.json",
        dict(
            method=method,
            seed=seed,
            config=vars(cfg),
            actor_parameter_count=sum(
                p.numel() for p in policy.actor.parameters() if p.requires_grad
            ),
            critic_parameter_count=sum(p.numel() for p in policy.critic.parameters()),
            critic_sha256=tensor_hash(policy.critic.parameters()),
            graph_sha256=sha(graph_path(method, seed)),
            graph_buffers_sha256=graph_before,
            action_storage="pre-tanh Gaussian coordinates",
            observation_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            family="angle",
            objective="aligned",
            teacher_used=False,
            pretrained_actor_loaded=False,
            readout="normalized",
            body="NeuromechFly",
            initial_log_std=initial_log_std,
            motor_interface="shared_CPG_command2_drive_and_turn",
        ),
    )
    env = SubprocVecEnv(
        [
            partial(make_env, i, cfg.n_envs, "angle", "aligned", str(folder))
            for i in range(cfg.n_envs)
        ]
    )
    trainer = AuditedPPO(
        env, policy, cfg, device, folder=folder, conf=conf, protocol_hash=sha(PROTOCOL)
    )
    try:
        result = trainer.learn(folder, resume_from=None)
        if graph_before:
            assert (
                tensor_hash(
                    [
                        policy.actor.core.edge_pre,
                        policy.actor.core.edge_post,
                        policy.actor.core.edge_weight,
                    ]
                )
                == graph_before
            )
        write_json(
            folder / "COMPLETE.json",
            dict(
                decisions=cfg.total_timesteps,
                elapsed_s=result["elapsed_wall"],
                graph_frozen=True,
                completed=time.time(),
            ),
        )
        write_json(
            folder / "STATUS.json",
            dict(state="complete", decisions=cfg.total_timesteps),
        )
    except BaseException:
        write_json(
            folder / "ERROR.json",
            dict(traceback=traceback.format_exc(), time=time.time()),
        )
        raise
    finally:
        env.close()

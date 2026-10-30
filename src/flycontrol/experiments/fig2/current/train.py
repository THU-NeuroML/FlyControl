"""Current 20-pair Adult/MLP/ER training; frozen numerical engine."""

from .paths import DATA
from fig6.paths import ROOT as REPO
from fig2.runtime import activate

activate()
from terrain_control.context import ROOT, CONFIG, write_json, configure_process, sha
from terrain_control.train import AuditedPPO, tensor_hash
from terrain_control.task import make_env
from terrain_control.policy import build_policy
from terrain_control.antithetic import install_antithetic
from fly_control.ppo import PPOConfig
from fly_control.vector_env import SubprocVecEnv
import argparse, json, time, traceback
from functools import partial
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["connectome", "mlp", "er"], required=True)
    p.add_argument("--seed", type=int, choices=range(2011, 2031), default=2011)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    configure_process()
    conf = json.loads((CONFIG / "protocol.json").read_text())
    run = ("smoke_" if a.smoke else "") + f"{a.method}_seed{a.seed}"
    folder = ROOT / "runs" / run
    # Operational guard only: exclude concurrent writers, including--resume.
    import fcntl, os

    locks = ROOT / "qa/run_locks"
    locks.mkdir(parents=True, exist_ok=True)
    run_lock = (locks / f"{run}.lock").open("a+")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Run already owned by another process:{run}")
    run_lock.seek(0)
    run_lock.truncate()
    run_lock.write(str(os.getpid()))
    run_lock.flush()
    if folder.exists() and not a.resume:
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(
        conf["ppo"],
        seed=a.seed,
        initialization_seed=a.seed,
        environment_seed=5_000_000 + a.seed * 100,
        policy_sampling_seed=100000 + a.seed,
        minibatch_seed=300000 + a.seed,
    )
    if a.smoke:
        config.update(
            total_timesteps=128,
            n_envs=2,
            n_steps=32,
            seq_len=8,
            n_epochs=2,
            n_minibatches=2,
        )
    cfg = PPOConfig(**config)
    initial_log_std = float(conf.get("policy", {}).get("initial_log_std", -1.0))
    graph_path = (
        DATA / "er_inputs" / f"er_seed{a.seed}.npz"
        if a.method == "er"
        else DATA / "graph.npz"
    )
    policy = build_policy(
        graph_path,
        688,
        6,
        "connectome" if a.method == "er" else a.method,
        a.seed,
        a.device,
        "normalized",
        initial_log_std,
    )

    install_antithetic(policy)
    graph_before = None
    if a.method in {"connectome", "er"}:
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
            method=a.method,
            seed=a.seed,
            config=vars(cfg),
            actor_parameter_count=sum(
                p.numel() for p in policy.actor.parameters() if p.requires_grad
            ),
            critic_parameter_count=sum(p.numel() for p in policy.critic.parameters()),
            critic_sha256=tensor_hash(policy.critic.parameters()),
            graph_sha256=sha(graph_path),
            graph_buffers_sha256=graph_before,
            action_storage="pre-tanh Gaussian coordinates",
            observation_dim=688,
            action_dim=6,
            teacher_used=False,
            pretrained_actor_loaded=False,
            readout="normalized",
            body="NeuromechFly",
            exploration_sampling="antithetic_environment_halves",
            initial_log_std=initial_log_std,
            leg_residual_l2_limit_rad=0.15 * np.sqrt(3),
            motor_interface="normalized_multijoint_leg_synergy",
        ),
    )
    if a.method == "er":
        reference = build_policy(DATA / "graph.npz", 688, 6, "connectome", a.seed,
                                 a.device, "normalized", initial_log_std)
        for (name, value), (other, expected) in zip(policy.named_parameters(), reference.named_parameters()):
            assert name == other and torch.equal(value, expected), name
        del reference
        write_json(folder / "MATCHED_INITIALIZATION.json", dict(
            all_trainable_tensors_identical=True, graph_is_only_model_difference=True,
            reference="fresh same-seed Adult initialization"))
    env = SubprocVecEnv(
        [
            partial(make_env, i, cfg.n_envs, conf["cells"], run)
            for i in range(cfg.n_envs)
        ]
    )
    trainer = AuditedPPO(
        env,
        policy,
        cfg,
        a.device,
        folder=folder,
        conf=conf,
        protocol_hash=sha(CONFIG / "protocol.json"),
    )
    try:
        result = trainer.learn(
            folder, resume_from=folder / "checkpoint.pt" if a.resume else None
        )
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


if __name__ == "__main__":
    main()

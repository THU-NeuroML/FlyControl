"""Fresh paired PPO, using the successful hexapod neural implementation.

Task-specific orchestration only; optimizer/replay semantics are inherited
from its frozen PPO, with the same full-rollout KL adaptive actor schedule.
"""

from .common import DATA, PROTOCOL, REPO, ROOT, write_json, configure_process, sha
import argparse, json, time, hashlib, traceback
from functools import partial
import numpy as np
import torch
from .task import make_env, OBS_DIM, ACTION_DIM
from fly_control.ppo import PPOConfig, RecurrentPPO
from fly_control.vector_env import SubprocVecEnv
from .policy_adapter import build_policy, graph_path


def tensor_hash(items):
    h = hashlib.sha256()
    for x in items:
        h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class AuditedPPO(RecurrentPPO):
    def __init__(self, *args, folder, conf, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder, self.conf = folder, conf
        self.rule = conf["adaptive_policy_kl"]
        self._policy_sampling_generator = torch.Generator("cpu").manual_seed(
            self.cfg.policy_sampling_seed
        )

    def _collect(self, *args, **kwargs):
        self.current_update = kwargs.get("update", 0)
        result = super()._collect(*args, **kwargs)
        if self.current_update == 1:
            b = result[0]
            write_json(
                self.folder / "FIRST_ROLLOUT.json",
                dict(
                    observations_sha256=tensor_hash([b.obs]),
                    gaussian_actions_sha256=tensor_hash([b.actions]),
                    rewards_sha256=tensor_hash([b.rewards]),
                ),
            )
        return result

    def _update(self, buf):
        before = tensor_hash(self.policy.parameters())

        # Read-only diagnostics for whether the independent critic fits its
        # actual rollout targets; no extra optimizer steps or RNG draws.
        def critic_fit():
            with torch.no_grad():
                target = buf.returns.reshape(-1)
                prediction = self.policy.critic(buf.obs.reshape(-1, buf.obs.shape[-1]))
                error = target - prediction
                variance = float(target.var(unbiased=False))
                return dict(
                    target_mean=float(target.mean()),
                    target_std=variance**0.5,
                    prediction_mean=float(prediction.mean()),
                    rmse=float(error.square().mean().sqrt()),
                    variance_explained=1.0 - float(error.var(unbiased=False)) / variance
                    if variance > 1e-12
                    else None,
                )

        critic_before = critic_fit()
        with torch.no_grad():
            error = 0.0
            for batch in buf.iter_chunks(
                self.cfg.seq_len,
                self.cfg.n_minibatches,
                rng=np.random.default_rng(993000 + self.current_update),
            ):
                logp, _, _ = self._evaluate_chunk(batch)
                error = max(error, float((logp - batch["old_lp"]).abs().max()))
            assert error < 0.003, ("Probability replay error", error)
        out = super()._update(buf)
        critic_after = critic_fit()
        assert out["n_minibatches"] > 0
        with torch.no_grad():
            total = 0.0
            count = 0
            for batch in buf.iter_chunks(
                self.cfg.seq_len,
                self.cfg.n_minibatches,
                rng=np.random.default_rng(994000 + self.current_update),
            ):
                logp, _, _ = self._evaluate_chunk(batch)
                delta = logp - batch["old_lp"]
                guard = delta.exp() - 1 - delta
                total += float(guard.sum())
                count += guard.numel()
        end_kl = total / count
        factor = (
            self.rule["factor"]
            if end_kl < self.rule["target"] / 2
            else 1 / self.rule["factor"]
            if end_kl > self.rule["target"] * 2
            else 1.0
        )
        for group in self.optim.param_groups:
            if group.get("group_name") == "actor":
                old = group["lr"]
                group["lr"] = float(
                    np.clip(old * factor, self.rule["min_lr"], self.rule["max_lr"])
                )
                self._optimizer_learning_rates["actor"] = group["lr"]
                out.update(actor_lr_used=old, actor_lr_next=group["lr"])
        changed = tensor_hash(self.policy.parameters()) != before
        assert changed
        out.update(
            replay_log_probability_max_error=error,
            end_of_update_policy_kl=end_kl,
            parameters_changed=changed,
            physical_action_rms=float(buf.actions.tanh().square().mean().sqrt()),
            critic_fit_before=critic_before,
            critic_fit_after=critic_after,
        )
        write_json(self.folder / f"update_{self.current_update:04d}.json", out)
        return out

    def _save_checkpoint_atomic(self, ckpt, path):
        # Always evaluate fixed-budget policy files, never best-by-return.
        super()._save_checkpoint_atomic(ckpt, path)
        if path.name not in ("checkpoint.pt", "initial_checkpoint.pt"):
            return
        steps = int(ckpt["timesteps"])
        write_json(
            self.folder / "STATUS.json",
            dict(
                state="training",
                decisions=steps,
                target=self.cfg.total_timesteps,
                elapsed_s=ckpt["elapsed_wall"],
                update=int(ckpt["update"]),
                pid=__import__("os").getpid(),
            ),
        )
        if (
            steps in self.conf["evaluation_budgets"]
            or steps == self.cfg.total_timesteps
        ):
            slim = {
                k: ckpt[k]
                for k in (
                    "policy_state",
                    "timesteps",
                    "obs_rms_state",
                    "ret_rms_state",
                    "protocol_hash",
                )
            }
            target = self.folder / f"policy_{steps:08d}.pt"
            super()._save_checkpoint_atomic(slim, target)
            write_json(
                self.folder / f"READY_{steps:08d}.json",
                dict(decisions=steps, checkpoint=str(target), sha256=sha(target)),
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["connectome", "mlp", "er"], required=True)
    p.add_argument("--family", choices=["push", "angle"], required=True)
    p.add_argument("--objective", choices=["legacy", "aligned"], required=True)
    p.add_argument("--seed", type=int, choices=range(4301, 4325), default=4301)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    configure_process()
    conf = json.loads((PROTOCOL).read_text())
    run = (
        "smoke_" if a.smoke else ""
    ) + f"{a.family}_{a.objective}_{a.method}_seed{a.seed}"
    folder = ROOT / "runs" / run
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
            total_timesteps=512,
            n_envs=2,
            n_steps=128,
            seq_len=32,
            n_epochs=2,
            n_minibatches=2,
        )
    cfg = PPOConfig(**config)
    initial_log_std = float(conf.get("policy", {}).get("initial_log_std", -1.0))
    policy = build_policy(
        DATA / "graph.npz",
        OBS_DIM,
        ACTION_DIM,
        a.method,
        a.seed,
        a.device,
        "normalized",
        initial_log_std,
    )
    graph_before = None
    if a.method in ("connectome", "er"):
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
            graph_sha256=sha(graph_path(a.method, a.seed)),
            graph_buffers_sha256=graph_before,
            action_storage="pre-tanh Gaussian coordinates",
            observation_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            family=a.family,
            objective=a.objective,
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
            partial(make_env, i, cfg.n_envs, a.family, a.objective, run)
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
        protocol_hash=sha(PROTOCOL),
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

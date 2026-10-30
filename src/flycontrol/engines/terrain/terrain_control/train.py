"""Fresh paired PPO, using the successful hexapod neural implementation.

Task-specific orchestration only; optimizer/replay semantics are inherited
from its frozen PPO, with the same full-rollout KL adaptive actor schedule.
"""

from .context import ROOT, CONFIG, GRAPH, write_json, configure_process, sha
import argparse, json, time, hashlib, traceback
from functools import partial
import numpy as np
import torch
from .task import make_env
from fly_control.ppo import PPOConfig
from .ppo import RecurrentPPO
from fly_control.vector_env import SubprocVecEnv
from .policy import build_policy
from .antithetic import install_antithetic, reset_audit, finish_audit


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
        self.interaction_clock = dict(physics_steps=0, micro_steps=0, macro_decisions=0)
        self.height_interaction_clock = {}
        self.actual_height_interaction_clock = {}
        native_step = self.env.step

        def clocked_step(actions):
            result = native_step(actions)
            for info in result[3]:
                self.interaction_clock["physics_steps"] += int(
                    info["action_hold_physics_steps"]
                )
                self.interaction_clock["micro_steps"] += int(
                    info["action_hold_micro_steps"]
                )
                self.interaction_clock["macro_decisions"] += 1
                cell = info.get("training_band_id", info["terrain_spec_id"])
                h = self.height_interaction_clock.setdefault(
                    cell, dict(macro_decisions=0, physics_steps=0)
                )
                h["macro_decisions"] += 1
                h["physics_steps"] += int(info["action_hold_physics_steps"])
                key = f"{info['training_nominal_value_mm']:.17g}:{info['terrain_value_mm']:.17g}"
                actual = self.actual_height_interaction_clock.setdefault(
                    key,
                    dict(
                        nominal_value_mm=info["training_nominal_value_mm"],
                        value_mm=info["terrain_value_mm"],
                        macro_decisions=0,
                        physics_steps=0,
                    ),
                )
                actual["macro_decisions"] += 1
                actual["physics_steps"] += int(info["action_hold_physics_steps"])
            return result

        self.env.step = clocked_step

    def _collect(self, *args, **kwargs):
        self.current_update = kwargs.get("update", 0)
        reset_audit(self.policy, self._policy_sampling_generator)
        result = super()._collect(*args, **kwargs)
        audit = finish_audit(
            self.policy,
            self._policy_sampling_generator,
            self.cfg.n_steps,
            self.cfg.n_envs,
        )
        write_json(
            self.folder / f"SAMPLING_AUDIT_{self.current_update:04d}.json", audit
        )
        if self.current_update == 1:
            b = result[0]
            record = dict(
                observations_sha256=tensor_hash([b.obs]),
                gaussian_actions_sha256=tensor_hash([b.actions]),
                rewards_sha256=tensor_hash([b.rewards]),
            )
            write_json(self.folder / "FIRST_ROLLOUT.json", record)
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
            requested_tanh_action_rms=float(buf.actions.tanh().square().mean().sqrt()),
            critic_fit_before=critic_before,
            critic_fit_after=critic_after,
        )
        out["interaction_clock"] = dict(
            self.interaction_clock,
            actual_physical_seconds=self.interaction_clock["physics_steps"] * 5e-5,
            nominal_physical_seconds=self.interaction_clock["macro_decisions"]
            * 0.01
            * self.conf["action_hold"]["repeat"],
        )
        out["height_interaction_clock"] = dict(self.height_interaction_clock)
        out["actual_height_interaction_clock"] = {
            k: dict(v) for k, v in self.actual_height_interaction_clock.items()
        }
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
        if True:
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
    p.add_argument("--method", choices=["connectome", "mlp"], required=True)
    p.add_argument("--seed", type=int, default=1601)
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
    policy = build_policy(
        GRAPH, 688, 6, a.method, a.seed, a.device, "normalized", initial_log_std
    )

    install_antithetic(policy)
    graph_before = None
    if a.method == "connectome":
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
            graph_sha256=sha(GRAPH),
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

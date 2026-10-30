"""PPO training with stored Gaussian actions and body-specific checkpoint evaluation."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path
import time
import numpy as np
import torch

from fly_control.ppo import PPOConfig, RecurrentPPO
from fly_control.vector_env import SubprocVecEnv, SyncVecEnv
from fly_control.io import NoTeacherImports, write_json, CheckpointYield
from fly_control.io import digest
from body_control.env import TaskEnv, make_env, BODY_CONFIG
from body_control import build_policy


@torch.no_grad()
def evaluate(policy, body, device, seeds, command, interface="native"):
    from flycontrol.paths import run_root, read_json
    smoke = read_json(run_root() / "run.json")["smoke"]
    horizon = 0.2 if smoke else 20.0
    if smoke:
        seeds = list(seeds)[:1]
    envs = [make_env(body, eval_speed=command, interface=interface) for _ in seeds]
    try:
        obs = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, seeds)])
        state = policy.init_state(len(envs), device)
        done = np.zeros(len(envs), bool)
        failed = done.copy()
        length = np.zeros(len(envs), int)
        rewards = np.zeros(len(envs))
        discounted = rewards.copy()
        lat = rewards.copy()
        tilt = rewards.copy()
        late_x = np.zeros(len(envs))
        late_time = np.zeros(len(envs))
        elapsed = np.zeros(len(envs))
        action_hash = [hashlib.sha256() for _ in seeds]
        for step in range(round(horizon / envs[0].dt)):
            action, _, _, state = policy.act(
                torch.as_tensor(obs, device=device), state, deterministic=True
            )
            actions = action.cpu().numpy()
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                old_x = float(e.data.qpos[0])
                obs[i], reward, term, trunc, info = e.step(actions[i])
                action_hash[i].update(actions[i].tobytes())
                rewards[i] += reward
                discounted[i] += np.exp(-elapsed[i] / 5) * reward
                elapsed[i] += info["control_duration_s"]
                length[i] += 1
                if elapsed[i] > 10 + 1e-8:
                    late_x[i] += e.data.qpos[0] - old_x
                    late_time[i] += info["control_duration_s"]
                lat[i] = max(lat[i], abs(e.data.qpos[1] - e.physics.origin[1]))
                tilt[i] = max(tilt[i], info["torso_tilt_degrees"])
                failed[i] = term
                done[i] = term or trunc
            if done.all():
                break
        result = []
        for i, (e, seed) in enumerate(zip(envs, seeds)):
            speed = float((e.data.qpos[0] - e.physics.origin[0]) / horizon)
            survived = bool(not failed[i] and elapsed[i] >= horizon - 1e-6)
            result.append(
                dict(
                    body=body,
                    seed=seed,
                    command=command,
                    duration_s=float(elapsed[i]),
                    survived=survived,
                    velocity_mps=speed,
                    late_velocity_mps=(
                        float(late_x[i] / late_time[i]) if late_time[i] > 0 else None
                    ),
                    lateral_max_m=float(lat[i]),
                    max_tilt_deg=float(tilt[i]),
                    return_raw=float(rewards[i]),
                    discounted_return=float(discounted[i]),
                    success=bool(
                        survived
                        and (
                            abs(speed) <= 0.05
                            if command < 0.02
                            else speed >= 0.7 * command
                        )
                        and lat[i] <= 0.75
                    ),
                    action_sha256=action_hash[i].hexdigest(),
                    teacher_used=False,
                    body_interface=interface,
                )
            )
        return result
    finally:
        for env in envs:
            env.close()


class TrainingPPO(RecurrentPPO):
    def __init__(self, *args, root, folder, body, interface="native", **kwargs):
        super().__init__(*args, **kwargs)
        self.root, self.folder, self.body = root, folder, body
        self.interface = interface
        self.adaptive_kl = json.loads((root / "protocol.json").read_text()).get(
            "adaptive_policy_kl"
        )
        if self.adaptive_kl:
            assert {g.get("group_name") for g in self.optim.param_groups} == {
                "actor",
                "critic",
                "log_std",
            }
        self._policy_sampling_generator = torch.Generator("cpu").manual_seed(
            self.cfg.policy_sampling_seed
        )

    def _collect(self, *args, **kwargs):
        self.current_update = kwargs.get("update", 0)
        result = super()._collect(*args, **kwargs)
        if self.current_update == 1:
            buf = result[0]
            write_json(
                self.folder / "first_rollout.json",
                dict(
                    observations_sha256=hashlib.sha256(
                        buf.obs.cpu().numpy().tobytes()
                    ).hexdigest(),
                    gaussian_actions_sha256=hashlib.sha256(
                        buf.actions.cpu().numpy().tobytes()
                    ).hexdigest(),
                    rewards_sha256=hashlib.sha256(
                        buf.rewards.cpu().numpy().tobytes()
                    ).hexdigest(),
                ),
            )
        return result

    def _update(self, buf):
        # Diagnose the full stored rollout before ANY optimizer mutation.
        with torch.no_grad():
            max_error = 0.0
            guard_values = []
            audit_rng = np.random.default_rng(993000 + self.current_update)
            for batch in buf.iter_chunks(
                self.cfg.seq_len, self.cfg.n_minibatches, rng=audit_rng
            ):
                logp, _, _ = self._evaluate_chunk(batch)
                delta = logp - batch["old_lp"]
                max_error = max(max_error, float(delta.abs().max()))
                guard_values.append(float((delta.exp() - 1 - delta).mean()))
            assert max_error < 0.003, ("Unchanged-policy replay mismatch", max_error)
        out = super()._update(buf)
        assert out["n_minibatches"] > 0, ("No accepted update", out)
        if self.adaptive_kl:
            # Measure the complete rollout after the update, rather than using
            # the average of intermediate minibatch KL values as its endpoint.
            with torch.no_grad():
                total_kl = 0.0
                count = 0
                rng = np.random.default_rng(994000 + self.current_update)
                for batch in buf.iter_chunks(
                    self.cfg.seq_len, self.cfg.n_minibatches, rng=rng
                ):
                    logp, _, _ = self._evaluate_chunk(batch)
                    delta = logp - batch["old_lp"]
                    guard = delta.exp() - 1 - delta
                    total_kl += float(guard.sum())
                    count += guard.numel()
                end_kl = total_kl / count
            rule = self.adaptive_kl
            factor = 1.0
            if end_kl < rule["target"] / 2:
                factor = rule["factor"]
            elif end_kl > rule["target"] * 2:
                factor = 1 / rule["factor"]
            for group in self.optim.param_groups:
                if group["group_name"] == "actor":
                    old_lr = group["lr"]
                    group["lr"] = float(
                        np.clip(old_lr * factor, rule["min_lr"], rule["max_lr"])
                    )
                    self._optimizer_learning_rates["actor"] = group["lr"]
                    out.update(actor_lr_used=old_lr, actor_lr_next=group["lr"])
            out["end_of_update_policy_kl"] = end_kl
        out.update(
            replay_log_probability_max_error=max_error,
            replay_guard_max=max(guard_values),
            physical_action_abs_mean=float(buf.actions.tanh().abs().mean()),
            physical_action_saturation_fraction=float(
                (buf.actions.tanh().abs() >= 0.95).float().mean()
            ),
        )
        if self.current_update <= 2:
            actor = self.policy.actor
            head = actor.core.output_layer if hasattr(actor, "core") else actor.net[-1]
            if isinstance(head, torch.nn.Sequential):
                head = head[-1]
            out["nonzero_joint_head_rows"] = int(
                (head.weight.detach().abs().sum(1) > 0).sum()
            )
            write_json(self.folder / f"update_audit_{self.current_update}.json", out)
        return out


def main():
    import sys
    from body_control.training import CorrectPPO

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--body", choices=list(BODY_CONFIG), required=True)
    p.add_argument("--method", choices=["connectome", "mlp"], required=True)
    p.add_argument(
        "--readout", choices=["linear", "normalized", "reflex"], default="reflex"
    )
    p.add_argument("--seed", type=int, default=820)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    sys.meta_path.insert(0, NoTeacherImports())
    os.nice(10)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.3, torch.device(a.device))
    root = a.root.resolve()
    conf = json.loads((root / "protocol.json").read_text())
    run_id = f"{a.body}_{a.method}_{a.readout}_seed{a.seed}"
    folder = root / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    interface = conf.get("body_interfaces", {}).get(a.body, "native")
    e = make_env(a.body, interface=interface)
    obs = e.observation_space.shape[0]
    actions = e.nu
    dt = e.dt
    bodyconf = e.config.copy()
    e.close()
    policy = build_policy(
        root / "inputs/graph.npz",
        obs,
        actions,
        a.method,
        a.seed,
        a.device,
        a.readout,
        bodyconf["log_std"],
    )
    config = dict(
        conf["ppo"],
        gamma=float(np.exp(-dt / 5)),
        lam=float(np.exp(-dt / 0.8)),
        seed=a.seed,
        initialization_seed=a.seed,
        environment_seed=920000 + a.seed * 100,
        policy_sampling_seed=100000 + a.seed,
        minibatch_seed=300000 + a.seed,
    )
    if a.smoke:
        config.update(
            total_timesteps=256,
            n_envs=2,
            n_steps=64,
            seq_len=16,
            n_epochs=2,
            n_minibatches=4,
        )
    cfg = PPOConfig(**config)
    write_json(
        folder / "initialization.json",
        dict(
            body=a.body,
            method=a.method,
            readout=a.readout,
            seed=a.seed,
            config=vars(cfg),
            action_storage="pre-tanh Gaussian coordinates",
            teacher_used=False,
            pretrained_actor_loaded=False,
            body_interface=interface,
            body_config=bodyconf,
            actor_parameter_count=sum(
                x.numel() for x in policy.actor.parameters() if x.requires_grad
            ),
            critic_sha256=hashlib.sha256(
                b"".join(
                    x.detach().cpu().numpy().tobytes()
                    for x in policy.critic.parameters()
                )
            ).hexdigest(),
        ),
    )
    vec = (SyncVecEnv if a.smoke else SubprocVecEnv)(
        [
            partial(make_env, a.body, cfg.n_envs, interface=interface)
            for _ in range(cfg.n_envs)
        ]
    )
    trainer = CorrectPPO(
        vec,
        policy,
        cfg,
        a.device,
        root=root,
        folder=folder,
        body=a.body,
        interface=interface,
        protocol_hash=digest(root / "protocol.json"),
    )
    try:
        checkpoint = folder / "checkpoint.pt"
        result = trainer.learn(
            folder, resume_from=checkpoint if checkpoint.exists() else None
        )
        write_json(
            folder / "complete.json",
            dict(
                timesteps=cfg.total_timesteps,
                elapsed_s=result["elapsed_wall"],
                at=time.time(),
                teacher_used=False,
            ),
        )
        write_json(
            folder / "status.json",
            dict(state="complete", timesteps=cfg.total_timesteps),
        )
    except CheckpointYield:
        print("Checkpointed; yielding resources", flush=True)
        raise SystemExit(75)
    finally:
        vec.close()

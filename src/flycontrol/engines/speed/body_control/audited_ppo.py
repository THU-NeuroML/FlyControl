"""Fresh paired PPO, using the successful hexapod neural implementation.

Task-specific orchestration only; optimizer/replay semantics are inherited
from its frozen PPO, with the same full-rollout KL adaptive actor schedule.
"""

from .context import write_json, sha
import hashlib
import numpy as np
import torch
from fly_control.ppo import RecurrentPPO


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
                    variance_explained=(
                        1.0 - float(error.var(unbiased=False)) / variance
                        if variance > 1e-12
                        else None
                    ),
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
            else 1 / self.rule["factor"] if end_kl > self.rule["target"] * 2 else 1.0
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

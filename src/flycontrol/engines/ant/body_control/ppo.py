"""Original Ant PPO protocol; only methods differing from the later robot PPO.

Changing timeout GAE, checkpoint scoring or the tanh action likelihood would
change the published experiment. Shared normalization, buffers and unchanged
trainer methods come from fly_control.ppo.
"""

from __future__ import annotations
import os, random, sys, time
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import nn, optim
from fly_control.ppo import (
    PPOConfig,
    RunningMeanStd,
    ObservationRunningMeanStd,
    RolloutBuffer as SharedRolloutBuffer,
    RecurrentPPO as SharedRecurrentPPO,
    _to_tensor,
    _log,
)
from fly_control.vector_env import SyncVecEnv, SubprocVecEnv
from .policy import TanhPolicy as Policy

EPISODE_INFO_METRICS = (
    "x_velocity",
    "y_velocity",
    "upright_factor",
    "torso_tilt_degrees",
    "torso_height",
    "goal_velocity",
    "goal_distance",
    "reward_official_goal",
    "reward_forward",
    "reward_ctrl",
    "reward_contact",
    "reward_survive",
    "wall_contact_count",
    "wall_contact_normal_force",
    "wall_contact_max_normal_force",
)


class RolloutBuffer(SharedRolloutBuffer):

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_terminated: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
                next_nonterm = 1.0 - last_terminated.float()
            else:
                next_val = self.values[t + 1]
                next_nonterm = 1.0 - self.terminated[t]
            next_val = torch.where(
                self.timeouts[t].bool(), self.timeout_values[t], next_val
            )
            delta = self.rewards[t] + gamma * next_val * next_nonterm - self.values[t]
            gae = delta + gamma * lam * next_nonterm * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def iter_chunks(self, seq_len: int, n_minibatches: int):
        """Yield contiguous time-chunks per env, grouped into minibatches.

        Each yielded chunk is ``(seq_len, batch)`` shaped; batch is the
        number of (env, chunk) pairs in this minibatch.
        """
        assert (
            self.n_steps % seq_len == 0
        ), f"n_steps ({self.n_steps}) must be divisible by seq_len ({seq_len})"
        n_chunks_per_env = self.n_steps // seq_len
        total_chunks = n_chunks_per_env * self.n_envs
        chunk_size = max(1, total_chunks // n_minibatches)
        (env_idx, chunk_idx) = np.meshgrid(
            np.arange(self.n_envs), np.arange(n_chunks_per_env), indexing="ij"
        )
        env_idx = env_idx.ravel()
        chunk_idx = chunk_idx.ravel()
        order = np.random.permutation(total_chunks)
        for start in range(0, total_chunks, chunk_size):
            sel = order[start : start + chunk_size]
            envs = env_idx[sel]
            chunks = chunk_idx[sel]
            t0 = chunks * seq_len
            obs = torch.stack(
                [self.obs[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            actions = torch.stack(
                [self.actions[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            old_lp = torch.stack(
                [self.log_probs[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            old_val = torch.stack(
                [self.values[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            adv = torch.stack(
                [self.advantages[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            ret = torch.stack(
                [self.returns[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            dones = torch.stack(
                [self.dones[t : t + seq_len, e] for (t, e) in zip(t0, envs)], dim=1
            )
            h0 = torch.stack([self.h_init[t, e] for (t, e) in zip(t0, envs)], dim=0)
            yield dict(
                obs=obs,
                actions=actions,
                old_lp=old_lp,
                old_val=old_val,
                adv=adv,
                ret=ret,
                dones=dones,
                h0=h0,
            )


class RecurrentPPO(SharedRecurrentPPO):

    def __init__(
        self,
        vec_env,
        policy: Policy,
        config: PPOConfig,
        device: str = "cpu",
        protocol_hash: str | None = None,
    ) -> None:
        self.env = vec_env
        self.policy = policy.to(device)
        self.cfg = config
        self.device = torch.device(device)
        self.protocol_hash = protocol_hash
        self.optim = optim.Adam(policy.parameters(), lr=config.lr)
        self.ret_rms: Optional[RunningMeanStd] = (
            RunningMeanStd() if config.normalize_reward else None
        )
        observation_shape = tuple(
            (int(value) for value in self.env.observation_space.shape)
        )
        self.obs_rms: Optional[ObservationRunningMeanStd] = (
            ObservationRunningMeanStd(observation_shape)
            if config.normalize_observation
            else None
        )
        self._running_returns = np.zeros(self.env.num_envs, dtype=np.float64)
        self._ep_return_raw = np.zeros(self.env.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.env.num_envs, dtype=np.int64)
        self._ep_info_sums = {
            key: np.zeros(self.env.num_envs, dtype=np.float64)
            for key in EPISODE_INFO_METRICS
        }
        self._ep_info_counts = {
            key: np.zeros(self.env.num_envs, dtype=np.int64)
            for key in EPISODE_INFO_METRICS
        }
        self._best_score: float = -float("inf")
        self._best_update: int = 0
        self._terrain_returns: dict[str, deque] = {}
        self._terrain_flips: dict[str, deque] = {}
        self._completed_episode_meta: list[dict] = []

    def _checkpoint_dict(
        self,
        update: int,
        timesteps: int,
        history: list,
        completed_returns: list,
        elapsed_wall: float,
    ) -> dict:
        """Everything needed to bit-resume training from end of `update`.

        Note: per-worker env RNG state is NOT captured. That's the only
        non-deterministic piece on resume — terrain randomization will
        restart from a fresh sequence. Empirically irrelevant for
        convergence, but flagged here so no one is surprised when a
        resumed run isn't bitwise-identical to an uninterrupted one.
        """
        get_states = getattr(self.env, "get_states", None)
        env_states = get_states() if get_states is not None else None
        return {
            "format_version": 4,
            "protocol_hash": self.protocol_hash,
            "update": update,
            "timesteps": timesteps,
            "elapsed_wall": elapsed_wall,
            "policy_state": self.policy.state_dict(),
            "optim_state": self.optim.state_dict(),
            "history": history,
            "completed_returns": completed_returns,
            "best_score": self._best_score,
            "best_update": self._best_update,
            "ret_rms_state": (
                self.ret_rms.state_dict() if self.ret_rms is not None else None
            ),
            "obs_rms_state": (
                self.obs_rms.state_dict() if self.obs_rms is not None else None
            ),
            "terrain_returns": {k: list(v) for (k, v) in self._terrain_returns.items()},
            "terrain_flips": {k: list(v) for (k, v) in self._terrain_flips.items()},
            "completed_episode_meta": self._completed_episode_meta,
            "ep_return_raw": self._ep_return_raw,
            "ep_len": self._ep_len,
            "env_states": env_states,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "cfg": {
                "total_timesteps": self.cfg.total_timesteps,
                "n_envs": self.cfg.n_envs,
                "n_steps": self.cfg.n_steps,
                "lr": self.cfg.lr,
                "ent_coef": self.cfg.ent_coef,
                "normalize_reward": self.cfg.normalize_reward,
                "normalize_observation": self.cfg.normalize_observation,
                "observation_clip": self.cfg.observation_clip,
                "target_kl": self.cfg.target_kl,
                "best_checkpoint_mode": self.cfg.best_checkpoint_mode,
                "best_checkpoint_window": self.cfg.best_checkpoint_window,
                "best_checkpoint_min_episodes": self.cfg.best_checkpoint_min_episodes,
                "best_checkpoint_velocity_key": self.cfg.best_checkpoint_velocity_key,
                "best_checkpoint_min_forward_velocity": self.cfg.best_checkpoint_min_forward_velocity,
                "best_checkpoint_max_high_z_termination_rate": self.cfg.best_checkpoint_max_high_z_termination_rate,
                "best_checkpoint_max_inverted_completion_rate": self.cfg.best_checkpoint_max_inverted_completion_rate,
                "best_checkpoint_max_functional_failure_rate": self.cfg.best_checkpoint_max_functional_failure_rate,
                "seed": self.cfg.seed,
            },
        }

    def load_checkpoint(self, path: Path) -> dict:
        """Restore policy/optim/RNG/history/normalizer; return metadata."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        checkpoint_protocol_hash = ckpt.get("protocol_hash")
        if (
            self.protocol_hash is not None
            and checkpoint_protocol_hash is not None
            and (checkpoint_protocol_hash != self.protocol_hash)
        ):
            raise ValueError(
                f"Checkpoint protocol hash does not match the current config: {checkpoint_protocol_hash} != {self.protocol_hash}"
            )
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        rng = ckpt.get("rng_state", {})
        if "torch" in rng:
            torch_rng = rng["torch"]
            if isinstance(torch_rng, torch.Tensor):
                torch_rng = torch_rng.detach().cpu().to(torch.uint8)
            else:
                torch_rng = torch.as_tensor(torch_rng, dtype=torch.uint8)
            torch.set_rng_state(torch_rng)
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        if self.ret_rms is not None and ckpt.get("ret_rms_state") is not None:
            self.ret_rms.load_state_dict(ckpt["ret_rms_state"])
        if self.obs_rms is not None:
            if ckpt.get("obs_rms_state") is None:
                raise ValueError("normalized-observation run lacks obs_rms_state")
            self.obs_rms.load_state_dict(ckpt["obs_rms_state"])
        self._best_score = float(ckpt.get("best_score", -float("inf")))
        self._best_update = int(ckpt.get("best_update", 0))
        for k, v in ckpt.get("terrain_returns", {}).items():
            self._terrain_returns[k] = deque(v, maxlen=100)
        for k, v in ckpt.get("terrain_flips", {}).items():
            self._terrain_flips[k] = deque(v, maxlen=100)
        self._completed_episode_meta = list(ckpt.get("completed_episode_meta", []))
        if "ep_return_raw" in ckpt:
            self._ep_return_raw = np.asarray(ckpt["ep_return_raw"], dtype=np.float64)
        if "ep_len" in ckpt:
            self._ep_len = np.asarray(ckpt["ep_len"], dtype=np.int64)
        env_states = ckpt.get("env_states")
        if env_states is not None:
            set_states = getattr(self.env, "set_states", None)
            if set_states is None:
                raise RuntimeError(
                    "Checkpoint contains environment protocol state but the vector environment cannot restore it"
                )
            set_states(env_states)
        _log(
            f"[train] resumed from {path}: update={ckpt['update']} timesteps={ckpt['timesteps']} history_rows={len(ckpt.get('history', []))} best_score={self._best_score:.2f}@upd{self._best_update} elapsed_so_far={ckpt.get('elapsed_wall', 0.0) / 60:.1f}min"
        )
        return ckpt

    def _collect(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        update: Optional[int] = None,
        n_updates: Optional[int] = None,
    ) -> tuple:
        cfg = self.cfg
        buf = RolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=self.env.num_envs,
            obs_dim=obs.shape[-1],
            action_dim=self.env.action_space.shape[0],
            state_dim=self.policy.state_dim,
            device=self.device,
        )
        roll_t0 = time.time()
        eps_done_roll = 0
        ep_lens_roll: list[int] = []
        completed_meta_roll: list[dict] = []
        last_terminated_np = np.zeros(self.env.num_envs, dtype=np.bool_)
        tag = (
            f"upd {update:>4d}" + (f"/{n_updates}" if n_updates else "")
            if update is not None
            else "rollout"
        )
        for t in range(cfg.n_steps):
            (action, log_prob, value, h_next) = self.policy.act(obs, h)
            actions_np = action.detach().cpu().numpy()
            (next_obs_raw, rewards_np, dones_np, infos) = self.env.step(actions_np)
            terminated_np = np.asarray(
                [bool(info.get("terminated", d)) for (info, d) in zip(infos, dones_np)],
                dtype=np.bool_,
            )
            timeout_np = np.asarray(
                [bool(info.get("truncated", False)) for info in infos], dtype=np.bool_
            )
            last_terminated_np = terminated_np
            timeout_values_np = np.zeros(self.env.num_envs, dtype=np.float32)
            timeout_indices = [
                i
                for (i, is_timeout) in enumerate(timeout_np)
                if is_timeout
                and (not terminated_np[i])
                and ("terminal_observation" in infos[i])
            ]
            if timeout_indices:
                final_obs_raw = np.stack(
                    [
                        np.asarray(infos[i]["terminal_observation"], dtype=np.float32)
                        for i in timeout_indices
                    ]
                )
                final_obs = self._normalize_observation(final_obs_raw, update=False)
                with torch.no_grad():
                    timeout_values = self.policy.critic(
                        _to_tensor(final_obs, self.device)
                    )
                timeout_values_np[timeout_indices] = (
                    timeout_values.detach().cpu().numpy().astype(np.float32)
                )
            if self.ret_rms is not None:
                self._running_returns = cfg.gamma * self._running_returns + rewards_np
                self.ret_rms.update(self._running_returns)
                self._running_returns *= 1.0 - dones_np.astype(np.float64)
                rewards_for_buf = rewards_np / np.sqrt(self.ret_rms.var + 1e-08)
            else:
                rewards_for_buf = rewards_np
            next_obs_np = self._normalize_observation(next_obs_raw, update=True)
            buf.insert(
                t,
                obs=obs,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=_to_tensor(rewards_for_buf.astype(np.float32), self.device),
                done=_to_tensor(dones_np.astype(np.float32), self.device),
                terminated=_to_tensor(terminated_np.astype(np.float32), self.device),
                timeout=_to_tensor(timeout_np.astype(np.float32), self.device),
                timeout_value=_to_tensor(timeout_values_np, self.device),
                h_in=h,
            )
            done_mask = _to_tensor(dones_np.astype(np.float32), self.device).unsqueeze(
                -1
            )
            h = h_next * (1.0 - done_mask)
            obs = _to_tensor(next_obs_np, self.device)
            self._ep_return_raw += rewards_np
            self._ep_len += 1
            for key in EPISODE_INFO_METRICS:
                values = np.asarray(
                    [float(info.get(key, np.nan)) for info in infos], dtype=np.float64
                )
                finite = np.isfinite(values)
                self._ep_info_sums[key][finite] += values[finite]
                self._ep_info_counts[key][finite] += 1
            for i, d in enumerate(dones_np):
                if d:
                    info_i = infos[i] if infos and i < len(infos) else {}
                    terrain_i = str(info_i.get("terrain", "unknown"))
                    flipped_i = bool(info_i.get("flip", False))
                    episode_meta = {
                        "context": terrain_i,
                        "context_id": int(info_i.get("context_id", -1)),
                        "friction": float(info_i.get("friction", np.nan)),
                        "return": float(self._ep_return_raw[i]),
                        "length": int(self._ep_len[i]),
                        "flip": flipped_i,
                    }
                    for key in (
                        "native_terminated",
                        "high_z_termination",
                        "low_z_termination",
                        "nonfinite_termination",
                        "inverted",
                        "inverted_termination",
                        "physical_termination",
                        "functional_failure",
                        "goal_terminated",
                        "success",
                        "success_ever",
                    ):
                        if key in info_i:
                            episode_meta[key] = bool(info_i[key])
                    if "goals_reached" in info_i:
                        episode_meta["goals_reached"] = int(info_i["goals_reached"])
                    inverted_completion_i = bool(
                        info_i.get("truncated", False)
                        and (not info_i.get("terminated", False))
                        and info_i.get("inverted", False)
                    )
                    episode_meta["inverted_completion"] = inverted_completion_i
                    episode_meta["functional_failure"] = bool(
                        episode_meta.get("functional_failure", flipped_i)
                        or inverted_completion_i
                    )
                    episode_meta["flip"] = episode_meta["functional_failure"]
                    for key in EPISODE_INFO_METRICS:
                        count = int(self._ep_info_counts[key][i])
                        if count:
                            total = float(self._ep_info_sums[key][i])
                            episode_meta[f"mean_{key}"] = total / count
                            episode_meta[f"sum_{key}"] = total
                        self._ep_info_sums[key][i] = 0.0
                        self._ep_info_counts[key][i] = 0
                    if "mean_reward_ctrl" in episode_meta:
                        episode_meta["mean_action_cost"] = -episode_meta[
                            "mean_reward_ctrl"
                        ]
                        episode_meta["episode_action_cost"] = -episode_meta[
                            "sum_reward_ctrl"
                        ]
                    completed_meta_roll.append(episode_meta)
                    self._ep_return_raw[i] = 0.0
                    ep_lens_roll.append(int(self._ep_len[i]))
                    self._ep_len[i] = 0
                    eps_done_roll += 1
            if (
                cfg.rollout_heartbeat > 0
                and (t + 1) % cfg.rollout_heartbeat == 0
                and (t + 1 < cfg.n_steps)
            ):
                elapsed = time.time() - roll_t0
                env_steps_done = (t + 1) * self.env.num_envs
                sps = env_steps_done / max(0.001, elapsed)
                remaining_outer = cfg.n_steps - (t + 1)
                eta = remaining_outer * self.env.num_envs / max(0.001, sps)
                mean_len = (
                    float(np.mean(ep_lens_roll)) if ep_lens_roll else float("nan")
                )
                _log(
                    f"  [{tag} {t + 1:>3d}/{cfg.n_steps}] env_steps={env_steps_done:>5d} sps={sps:5.1f} eps={eps_done_roll:>3d} mean_ep_len={mean_len:5.1f} eta_roll={eta:5.0f}s"
                )
        with torch.no_grad():
            (_, _, last_value) = self.policy.forward(obs, h)
        last_terminated = _to_tensor(last_terminated_np.astype(np.float32), self.device)
        buf.compute_gae(last_value, last_terminated, cfg.gamma, cfg.lam)
        roll_elapsed = time.time() - roll_t0
        total_env_steps = cfg.n_steps * self.env.num_envs
        _log(
            f"  [{tag}] rollout done: {total_env_steps} env_steps in {roll_elapsed:.1f}s ({total_env_steps / max(0.001, roll_elapsed):.1f} sps), {eps_done_roll} episodes completed"
        )
        return (buf, obs, h, completed_meta_roll)

    def _evaluate_chunk(self, batch) -> tuple:
        (seq_len, B) = batch["obs"].shape[:2]
        if hasattr(self.policy.actor, "forward_sequence"):
            means, _ = self.policy.actor.forward_sequence(batch["obs"], batch["h0"].to(self.device), batch["dones"])
            dist = self.policy.distribution(means)
            a = batch["actions"].clamp(-1 + 1e-6, 1 - 1e-6)
            raw = .5 * (torch.log1p(a) - torch.log1p(-a))
            lp = dist.log_prob(raw).sum(-1) - torch.log1p(-a.square() + 1e-6).sum(-1)
            value = self.policy.critic(batch["obs"].reshape(seq_len * B, -1)).reshape(seq_len, B)
            return lp, value, dist.entropy().sum(-1)
        if not self.policy.actor.has_state:
            flat_obs = batch["obs"].reshape(seq_len * B, -1)
            flat_actions = batch["actions"].reshape(seq_len * B, -1)
            flat_hidden = torch.zeros(
                seq_len * B, 0, dtype=flat_obs.dtype, device=self.device
            )
            (new_lp, entropy, new_val, _) = self.policy.evaluate(
                flat_obs, flat_hidden, flat_actions
            )
            return (
                new_lp.reshape(seq_len, B),
                new_val.reshape(seq_len, B),
                entropy.reshape(seq_len, B),
            )
        h = batch["h0"].to(self.device)
        new_lp = torch.zeros(seq_len, B, device=self.device)
        new_val = torch.zeros(seq_len, B, device=self.device)
        entropy = torch.zeros(seq_len, B, device=self.device)
        for t in range(seq_len):
            (lp_t, ent_t, val_t, h_next) = self.policy.evaluate(
                batch["obs"][t], h, batch["actions"][t]
            )
            new_lp[t] = lp_t
            new_val[t] = val_t
            entropy[t] = ent_t
            done_mask = batch["dones"][t].unsqueeze(-1)
            h = h_next * (1.0 - done_mask)
        return (new_lp, new_val, entropy)

    def _update(self, buf: RolloutBuffer) -> dict:
        cfg = self.cfg
        if cfg.target_kl is not None and (
            not np.isfinite(cfg.target_kl) or cfg.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive")
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "guard_kl": 0.0,
            "n_minibatches": 0,
            "epochs_completed": 0,
            "kl_early_stop": False,
        }
        stop_early = False
        for _ in range(cfg.n_epochs):
            for batch in buf.iter_chunks(cfg.seq_len, cfg.n_minibatches):
                adv = batch["adv"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-08)
                (new_lp, new_val, ent) = self._evaluate_chunk(batch)
                ratio = torch.exp(new_lp - batch["old_lp"])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                v_clipped = batch["old_val"] + torch.clamp(
                    new_val - batch["old_val"], -cfg.vf_clip_range, cfg.vf_clip_range
                )
                vl_unclipped = (new_val - batch["ret"]).pow(2)
                vl_clipped = (v_clipped - batch["ret"]).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                entropy_loss = -ent.mean()
                loss = (
                    policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss
                )
                with torch.no_grad():
                    kl = (batch["old_lp"] - new_lp).mean().item()
                    log_ratio = new_lp - batch["old_lp"]
                    guard_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean().item()
                if cfg.target_kl is not None and guard_kl > 1.5 * cfg.target_kl:
                    stats["kl_early_stop"] = True
                    stop_early = True
                    break
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optim.step()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += -entropy_loss.item()
                stats["approx_kl"] += kl
                stats["guard_kl"] += guard_kl
                stats["n_minibatches"] += 1
            stats["epochs_completed"] += 1
            if stop_early:
                break
        n = max(1, stats["n_minibatches"])
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "guard_kl"):
            stats[k] /= n
        return stats

    def learn(
        self, log_dir: Optional[Path] = None, resume_from: Optional[Path] = None
    ) -> dict:
        cfg = self.cfg
        history: list[dict] = []
        completed_returns: list[float] = []
        start_update = 1
        prev_elapsed = 0.0
        if resume_from is not None:
            ckpt = self.load_checkpoint(Path(resume_from))
            history = list(ckpt.get("history", []))
            completed_returns = list(ckpt.get("completed_returns", []))
            start_update = int(ckpt["update"]) + 1
            prev_elapsed = float(ckpt.get("elapsed_wall", 0.0))
        rng_seeds = [cfg.seed + i for i in range(self.env.num_envs)]
        _log(
            f"[train] resetting {self.env.num_envs} envs (this builds the first sim per env; later resets are cached)..."
        )
        reset_t0 = time.time()
        initial_obs = self._normalize_observation(
            self.env.reset(rng_seeds), update=True
        )
        obs = _to_tensor(initial_obs, self.device)
        self._ep_return_raw[:] = 0.0
        self._ep_len[:] = 0
        for key in EPISODE_INFO_METRICS:
            self._ep_info_sums[key][:] = 0.0
            self._ep_info_counts[key][:] = 0
        self._running_returns[:] = 0.0
        _log(f"[train] initial reset done in {time.time() - reset_t0:.1f}s")
        h = self.policy.init_state(self.env.num_envs, self.device)
        steps_per_iter = cfg.n_steps * self.env.num_envs
        n_updates = cfg.total_timesteps // steps_per_iter
        _log(
            f"[train] plan: {n_updates} updates x {steps_per_iter} env_steps = {n_updates * steps_per_iter} total env_steps (target {cfg.total_timesteps}); start_update={start_update}"
        )
        if start_update > n_updates:
            _log(
                f"[train] checkpoint already past total_timesteps; nothing to do. Bump total_timesteps in config to continue further."
            )
            return {
                "history": history,
                "completed_returns": completed_returns,
                "elapsed_wall": prev_elapsed,
            }
        ckpt_path = Path(log_dir) / "checkpoint.pt" if log_dir is not None else None
        best_ckpt_path = (
            Path(log_dir) / "best_checkpoint.pt" if log_dir is not None else None
        )
        initial_ckpt_path = (
            Path(log_dir) / "initial_checkpoint.pt" if log_dir is not None else None
        )
        t0 = time.time()
        if (
            initial_ckpt_path is not None
            and start_update == 1
            and (not initial_ckpt_path.exists())
        ):
            initial_ckpt = self._checkpoint_dict(
                update=0,
                timesteps=0,
                history=history,
                completed_returns=completed_returns,
                elapsed_wall=prev_elapsed,
            )
            initial_ckpt["checkpoint_kind"] = "initial"
            self._save_checkpoint_atomic(initial_ckpt, initial_ckpt_path)
        for update in range(start_update, n_updates + 1):
            _log(
                f"[upd {update:>4d}/{n_updates}] start rollout: collecting {steps_per_iter} env_steps ({cfg.n_steps} outer x {self.env.num_envs} envs)..."
            )
            (buf, obs, h, new_meta) = self._collect(
                obs, h, update=update, n_updates=n_updates
            )
            new_returns_only = [float(meta["return"]) for meta in new_meta]
            completed_returns.extend(new_returns_only)
            for meta in new_meta:
                terrain_i = str(meta["context"])
                ret_i = float(meta["return"])
                flipped_i = bool(meta["flip"])
                if terrain_i not in self._terrain_returns:
                    self._terrain_returns[terrain_i] = deque(maxlen=100)
                    self._terrain_flips[terrain_i] = deque(maxlen=100)
                self._terrain_returns[terrain_i].append(float(ret_i))
                self._terrain_flips[terrain_i].append(1 if flipped_i else 0)
                self._completed_episode_meta.append(
                    {
                        "update": int(update),
                        "timesteps": int(update * steps_per_iter),
                        **meta,
                    }
                )
            recent = completed_returns[-100:] if completed_returns else [0.0]
            ep_rew_mean = float(np.mean(recent))
            (checkpoint_score, checkpoint_diagnostics) = self._checkpoint_candidate(
                ep_rew_mean
            )
            new_best = False
            best_policy_state = None
            if (
                best_ckpt_path is not None
                and len(completed_returns) >= self.cfg.best_checkpoint_min_episodes
                and (checkpoint_score > self._best_score)
            ):
                self._best_score = checkpoint_score
                self._best_update = update
                new_best = True
                best_policy_state = self._clone_policy_state()
            upd_t0 = time.time()
            _log(
                f"  [upd {update:>4d}/{n_updates}] starting PPO update ({cfg.n_epochs} epochs x {cfg.n_minibatches} minibatches)..."
            )
            stats = self._update(buf)
            steps = update * steps_per_iter
            run_elapsed = time.time() - t0
            total_elapsed = prev_elapsed + run_elapsed
            row = dict(
                update=update,
                timesteps=steps,
                fps=int(steps / max(0.001, total_elapsed)),
                ep_rew_mean=ep_rew_mean,
                best_score=self._best_score,
                best_update=self._best_update,
                checkpoint_score=checkpoint_score,
                checkpoint_gate_pass=bool(checkpoint_diagnostics["gate_pass"]),
                checkpoint_window_forward_velocity=float(
                    checkpoint_diagnostics.get("mean_forward_velocity", float("nan"))
                ),
                checkpoint_window_success_rate=float(
                    checkpoint_diagnostics.get("success_rate", float("nan"))
                ),
                checkpoint_window_mean_goals_reached=float(
                    checkpoint_diagnostics.get("mean_goals_reached", float("nan"))
                ),
                checkpoint_window_functional_failure_rate=float(
                    checkpoint_diagnostics.get("functional_failure_rate", float("nan"))
                ),
                checkpoint_window_high_z_termination_rate=float(
                    checkpoint_diagnostics.get("high_z_termination_rate", float("nan"))
                ),
                checkpoint_window_inverted_completion_rate=float(
                    checkpoint_diagnostics.get("inverted_completion_rate", float("nan"))
                ),
                **stats,
            )
            terrain_summary: list[str] = []
            for terrain_name in sorted(self._terrain_returns.keys()):
                rets = self._terrain_returns[terrain_name]
                flips = self._terrain_flips[terrain_name]
                if len(rets) == 0:
                    rew_t = float("nan")
                    flip_t = float("nan")
                else:
                    rew_t = float(np.mean(rets))
                    flip_t = float(np.mean(flips))
                row[f"rew_{terrain_name}"] = rew_t
                row[f"flip_{terrain_name}"] = flip_t
                row[f"n_{terrain_name}"] = len(rets)
                terrain_summary.append(
                    f"{terrain_name}:rew={rew_t:.0f}/flip={flip_t:.2f}(n={len(rets)})"
                )
            history.append(row)
            if update % cfg.log_interval == 0:
                updates_done_this_run = update - start_update + 1
                eta_total = (n_updates - update) * (
                    run_elapsed / max(1, updates_done_this_run)
                )
                best_tag = " *NEW BEST*" if new_best else ""
                _log(
                    f"[upd {update:>4d}/{n_updates}] t={steps:>8d} fps={row['fps']:>5d} rew={row['ep_rew_mean']:.2f} best={self._best_score:.2f}@{self._best_update} pl={row['policy_loss']:.3f} vl={row['value_loss']:.3f} ent={row['entropy']:.3f} kl={row['approx_kl']:.4f} upd_t={time.time() - upd_t0:.1f}s eta_total={eta_total / 60:.1f}min{best_tag}"
                )
                if terrain_summary:
                    _log(f"  per-terrain: {' | '.join(terrain_summary)}")
            if log_dir is not None:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(log_dir) / "history.npy", np.array(history))
                np.save(
                    Path(log_dir) / "episode_history.npy",
                    np.array(self._completed_episode_meta, dtype=object),
                )
            if ckpt_path is not None:
                ckpt = self._checkpoint_dict(
                    update=update,
                    timesteps=steps,
                    history=history,
                    completed_returns=completed_returns,
                    elapsed_wall=total_elapsed,
                )
                self._save_checkpoint_atomic(ckpt, ckpt_path)
                if new_best and best_ckpt_path is not None:
                    best_ckpt = dict(ckpt)
                    best_ckpt["policy_state"] = best_policy_state
                    best_ckpt["checkpoint_kind"] = "best_rollout_policy"
                    best_ckpt["score_update"] = update
                    best_ckpt["policy_source"] = "pre_ppo_update"
                    best_ckpt["selection_diagnostics"] = checkpoint_diagnostics
                    self._save_checkpoint_atomic(best_ckpt, best_ckpt_path)
        return {
            "history": history,
            "completed_returns": completed_returns,
            "best_score": self._best_score,
            "best_update": self._best_update,
            "elapsed_wall": total_elapsed,
        }

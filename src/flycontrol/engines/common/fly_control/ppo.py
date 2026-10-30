"""Recurrent PPO for the locomotion env.

Why we don't use SB3's PPO + custom feature extractor:

  SB3 PPO calls ``features_extractor.forward(obs)`` independently per
  step, with no notion of persistent hidden state across env steps. The
  recurrent actor's state is the time integral over the episode, so any
  per-call zero-state initialization invalidates the experiment.

This implementation is the minimum that handles a recurrent actor
correctly:

  * Vectorized envs run in lockstep, hidden state is owned externally and
    reset according to the declared zero/legacy/policy episode semantics.
  * The rollout stores ``h_init[t]`` (the hidden state fed into step t)
    so PPO can replay the actor through contiguous chunks.
  * Mini-batches are sliced as contiguous (env, time) chunks of length
    ``seq_len``; within a chunk, ``done`` flags reset the replayed h using the
    same semantics as rollout collection.
  * The critic is state-less (MLP on obs) so we recompute values
    timestep-wise without replay headache.
"""

from __future__ import annotations
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .actor_critic import Policy
from .vector_env import SyncVecEnv, SubprocVecEnv

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
    "forward_velocity_mm_s",
    "filtered_forward_velocity_mm_s",
    "body_z_mm",
)


def _log(msg: str) -> None:
    """Single point where progress lines are emitted.

    Writes to stderr so it stays visible even if stdout is piped, and
    forces a flush so the user sees output the instant it is produced
    (the default block buffering on non-tty stdout is why training looks
    "dead" for tens of minutes).
    """
    print(msg, file=sys.stderr, flush=True)


@dataclass
class PPOConfig:
    total_timesteps: int = 200000
    n_envs: int = 8
    n_steps: int = 256
    seq_len: int = 64
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    vf_clip_range: float = 0.2
    lr: float = 0.0003
    actor_lr: float | None = None
    critic_lr: float | None = None
    log_std_lr: float | None = None
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    separate_actor_critic_grad_clipping: bool = False
    target_kl: float | None = None
    normalize_reward: bool = True
    normalize_observation: bool = False
    observation_clip: float = 10.0
    log_interval: int = 1
    seed: int = 0
    separate_rng_streams: bool = False
    initialization_seed: int | None = None
    environment_seed: int | None = None
    policy_sampling_seed: int | None = None
    minibatch_seed: int | None = None
    initial_hidden_mode: str = "legacy_random"
    rollout_heartbeat: int = 16
    best_checkpoint_mode: str = "return"
    best_checkpoint_window: int = 100
    best_checkpoint_min_episodes: int = 10
    best_checkpoint_velocity_key: str = "mean_x_velocity"
    best_checkpoint_min_forward_velocity: float = 0.0
    best_checkpoint_max_high_z_termination_rate: float = 1.0
    best_checkpoint_max_inverted_completion_rate: float = 1.0
    best_checkpoint_max_functional_failure_rate: float = 1.0


class RunningMeanStd:
    """Welford's online mean/variance.

    Used to track the running variance of discounted returns so we can
    rescale rewards into O(1) range before they hit the value head.
    Mean is tracked but NOT subtracted from rewards (would change the
    sign of advantages and break learning).
    """

    def __init__(self, eps: float = 0.0001) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / new_count
        self.mean = new_mean
        self.var = m2 / new_count
        self.count = new_count

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


class ObservationRunningMeanStd:
    """Per-feature Welford statistics for vector observations."""

    def __init__(self, shape: tuple[int, ...], eps: float = 0.0001) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == self.mean.ndim:
            values = values[np.newaxis, ...]
        if values.shape[1:] != self.mean.shape:
            raise ValueError(
                f"observation shape {values.shape[1:]} != {self.mean.shape}"
            )
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = float(values.shape[0])
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / new_count
        self.mean = new_mean
        self.var = m2 / new_count
        self.count = new_count

    def normalize(self, x: np.ndarray, *, clip: float) -> np.ndarray:
        if clip <= 0:
            raise ValueError("observation_clip must be positive")
        values = np.asarray(x, dtype=np.float64)
        normalized = (values - self.mean) / np.sqrt(self.var + 1e-08)
        return np.clip(normalized, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("observation normalizer shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(var).all():
            raise ValueError("observation normalizer contains non-finite values")
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = float(state["count"])


def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _hidden_after_done(
    policy: Policy,
    hidden_next: torch.Tensor,
    done_mask: torch.Tensor,
    initial_hidden_mode: str,
) -> torch.Tensor:
    """Apply the declared episode-boundary recurrent-state semantics.

    ``legacy_random`` retains the historical behavior: use the actor's state
    initializer only at the first vector reset and use zeros after later
    episode boundaries.  ``zero`` uses zeros everywhere.  ``policy`` uses the
    actor-defined initializer at every boundary, which is required by cores
    whose state has a physical baseline (for example nonnegative tonic rates).
    """
    if initial_hidden_mode == "policy":
        reset_hidden = policy.init_state(hidden_next.shape[0], hidden_next.device)
        return torch.where(done_mask.bool(), reset_hidden, hidden_next)
    return hidden_next * (1.0 - done_mask)


class RolloutBuffer:
    """Holds one rollout for ``n_envs`` envs of length ``n_steps``."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = torch.zeros(n_steps, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, n_envs, action_dim, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.terminated = torch.zeros(n_steps, n_envs, device=device)
        self.timeouts = torch.zeros(n_steps, n_envs, device=device)
        self.timeout_values = torch.zeros(n_steps, n_envs, device=device)
        self.h_init = torch.zeros(n_steps, n_envs, state_dim, device=device)
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)

    def insert(
        self,
        t,
        *,
        obs,
        action,
        log_prob,
        value,
        reward,
        done,
        terminated,
        timeout,
        timeout_value,
        h_in,
    ):
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.terminated[t] = terminated
        self.timeouts[t] = timeout
        self.timeout_values[t] = timeout_value
        self.h_init[t] = h_in

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_terminated: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        """Compute GAE without leaking advantages across auto-reset episodes.

        A time-limit truncation is bootstrapped from the value of its terminal
        observation, but it is still an episode boundary.  Therefore the TD
        delta uses ``terminated`` (timeouts may bootstrap), whereas the GAE
        recursion uses ``done`` (timeouts must not inherit the reset episode's
        later advantages).

        ``last_terminated`` is retained in the public signature for checkpoint
        and test compatibility; the current transition masks stored in the
        buffer are the authoritative boundary flags.
        """
        del last_terminated
        gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            next_val = torch.where(
                self.timeouts[t].bool(), self.timeout_values[t], next_val
            )
            bootstrap_nonterminal = 1.0 - self.terminated[t]
            trace_continues = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_val * bootstrap_nonterminal
                - self.values[t]
            )
            gae = delta + gamma * lam * trace_continues * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def iter_chunks(
        self, seq_len: int, n_minibatches: int, rng: np.random.Generator | None = None
    ):
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
        order = (
            np.random.permutation(total_chunks)
            if rng is None
            else rng.permutation(total_chunks)
        )
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


class RecurrentPPO:

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
        explicit_group_lrs = any(
            (
                value is not None
                for value in (config.actor_lr, config.critic_lr, config.log_std_lr)
            )
        )
        self._optimizer_learning_rates = {
            "actor": float(config.lr if config.actor_lr is None else config.actor_lr),
            "critic": float(
                config.lr if config.critic_lr is None else config.critic_lr
            ),
            "log_std": float(
                config.lr if config.log_std_lr is None else config.log_std_lr
            ),
        }
        if config.lr <= 0 or any(
            (value <= 0 for value in self._optimizer_learning_rates.values())
        ):
            raise ValueError("all PPO learning rates must be positive")
        self._optimizer_group_mode = (
            "actor_critic_log_std" if explicit_group_lrs else "single_legacy"
        )
        if explicit_group_lrs:
            self.optim = optim.Adam(
                [
                    {
                        "params": list(policy.actor.parameters()),
                        "lr": self._optimizer_learning_rates["actor"],
                        "group_name": "actor",
                    },
                    {
                        "params": list(policy.critic.parameters()),
                        "lr": self._optimizer_learning_rates["critic"],
                        "group_name": "critic",
                    },
                    {
                        "params": [policy.log_std],
                        "lr": self._optimizer_learning_rates["log_std"],
                        "group_name": "log_std",
                    },
                ],
                lr=config.lr,
            )
        else:
            self.optim = optim.Adam(policy.parameters(), lr=config.lr)
        self._policy_sampling_generator: torch.Generator | None = None
        self._minibatch_rng: np.random.Generator | None = None
        if config.separate_rng_streams:
            policy_seed = (
                config.seed
                if config.policy_sampling_seed is None
                else int(config.policy_sampling_seed)
            )
            minibatch_seed = (
                config.seed
                if config.minibatch_seed is None
                else int(config.minibatch_seed)
            )
            self._policy_sampling_generator = torch.Generator(device=self.device)
            self._policy_sampling_generator.manual_seed(policy_seed)
            self._minibatch_rng = np.random.default_rng(minibatch_seed)
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
            "algorithm_semantics": {
                "gae_timeout_bootstrap": "terminal_observation_once",
                "gae_trace_boundary": "done_including_truncation",
                "actor_critic_grad_clipping": (
                    "separate"
                    if self.cfg.separate_actor_critic_grad_clipping
                    else "global_legacy"
                ),
                "optimizer_parameter_groups": {
                    "mode": self._optimizer_group_mode,
                    "learning_rates": dict(self._optimizer_learning_rates),
                },
                "recurrent_episode_reset": self.cfg.initial_hidden_mode,
            },
            "protocol_hash": self.protocol_hash,
            "update": update,
            "timesteps": timesteps,
            "elapsed_wall": elapsed_wall,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else str(self.device)
            ),
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
                "policy_sampling": (
                    self._policy_sampling_generator.get_state()
                    if self._policy_sampling_generator is not None
                    else None
                ),
                "minibatch": (
                    self._minibatch_rng.bit_generator.state
                    if self._minibatch_rng is not None
                    else None
                ),
            },
            "cfg": {
                "total_timesteps": self.cfg.total_timesteps,
                "n_envs": self.cfg.n_envs,
                "n_steps": self.cfg.n_steps,
                "lr": self.cfg.lr,
                "actor_lr": self.cfg.actor_lr,
                "critic_lr": self.cfg.critic_lr,
                "log_std_lr": self.cfg.log_std_lr,
                "ent_coef": self.cfg.ent_coef,
                "normalize_reward": self.cfg.normalize_reward,
                "normalize_observation": self.cfg.normalize_observation,
                "observation_clip": self.cfg.observation_clip,
                "target_kl": self.cfg.target_kl,
                "separate_actor_critic_grad_clipping": self.cfg.separate_actor_critic_grad_clipping,
                "best_checkpoint_mode": self.cfg.best_checkpoint_mode,
                "best_checkpoint_window": self.cfg.best_checkpoint_window,
                "best_checkpoint_min_episodes": self.cfg.best_checkpoint_min_episodes,
                "best_checkpoint_velocity_key": self.cfg.best_checkpoint_velocity_key,
                "best_checkpoint_min_forward_velocity": self.cfg.best_checkpoint_min_forward_velocity,
                "best_checkpoint_max_high_z_termination_rate": self.cfg.best_checkpoint_max_high_z_termination_rate,
                "best_checkpoint_max_inverted_completion_rate": self.cfg.best_checkpoint_max_inverted_completion_rate,
                "best_checkpoint_max_functional_failure_rate": self.cfg.best_checkpoint_max_functional_failure_rate,
                "seed": self.cfg.seed,
                "separate_rng_streams": self.cfg.separate_rng_streams,
                "initialization_seed": self.cfg.initialization_seed,
                "environment_seed": self.cfg.environment_seed,
                "policy_sampling_seed": self.cfg.policy_sampling_seed,
                "minibatch_seed": self.cfg.minibatch_seed,
                "initial_hidden_mode": self.cfg.initial_hidden_mode,
            },
        }

    def _save_checkpoint_atomic(self, ckpt: dict, path: Path) -> None:
        """Save via tmpfile + os.replace so we never leave a half-written
        file on disk. A SIGKILL mid-torch.save would otherwise produce a
        truncated .pt that fails to load.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    def _clone_policy_state(self) -> dict:
        """Snapshot policy weights before an optimizer step mutates them."""
        return {
            k: v.detach().cpu().clone() for (k, v) in self.policy.state_dict().items()
        }

    def _checkpoint_candidate(self, return_score: float) -> tuple[float, dict]:
        mode = str(self.cfg.best_checkpoint_mode)
        if mode == "return":
            return (
                return_score,
                {"mode": mode, "gate_pass": True, "return_score": return_score},
            )
        if mode == "goal_success":
            window = max(1, int(self.cfg.best_checkpoint_window))
            recent = self._completed_episode_meta[-window:]
            minimum = max(1, int(self.cfg.best_checkpoint_min_episodes))
            successes = np.asarray(
                [
                    float(row.get("success_ever", row.get("success", False)))
                    for row in recent
                ],
                dtype=np.float64,
            )
            failures = np.asarray(
                [
                    float(row.get("functional_failure", row.get("flip", False)))
                    for row in recent
                ],
                dtype=np.float64,
            )
            goals = np.asarray(
                [
                    float(
                        row.get(
                            "goals_reached",
                            row.get("success_ever", row.get("success", False)),
                        )
                    )
                    for row in recent
                ],
                dtype=np.float64,
            )
            success_rate = float(successes.mean()) if successes.size else 0.0
            failure_rate = float(failures.mean()) if failures.size else 1.0
            mean_goals = float(goals.mean()) if goals.size else 0.0
            gate_pass = bool(
                len(recent) >= minimum
                and failure_rate <= self.cfg.best_checkpoint_max_functional_failure_rate
            )
            bounded_return = float(np.arctan(return_score / 100.0))
            score = (
                (0.0 if gate_pass else -1000000.0)
                + 100000.0 * success_rate
                - 10000.0 * failure_rate
                + 100.0 * mean_goals
                + bounded_return
            )
            return (
                score,
                {
                    "mode": mode,
                    "gate_pass": gate_pass,
                    "window_episodes": len(recent),
                    "success_rate": success_rate,
                    "functional_failure_rate": failure_rate,
                    "mean_goals_reached": mean_goals,
                    "maximum_functional_failure_rate": self.cfg.best_checkpoint_max_functional_failure_rate,
                    "return_score": return_score,
                },
            )
        if mode != "stability_velocity":
            raise ValueError(
                f"best_checkpoint_mode must be 'return', 'goal_success', or 'stability_velocity', got {mode!r}"
            )
        window = max(1, int(self.cfg.best_checkpoint_window))
        recent = self._completed_episode_meta[-window:]
        minimum = max(1, int(self.cfg.best_checkpoint_min_episodes))
        velocities = np.asarray(
            [
                float(row.get(self.cfg.best_checkpoint_velocity_key, np.nan))
                for row in recent
            ],
            dtype=np.float64,
        )
        failures = np.asarray(
            [
                float(row.get("functional_failure", row.get("flip", False)))
                for row in recent
            ],
            dtype=np.float64,
        )
        high_z_terminations = np.asarray(
            [float(row.get("high_z_termination", False)) for row in recent],
            dtype=np.float64,
        )
        inverted_completions = np.asarray(
            [float(row.get("inverted_completion", False)) for row in recent],
            dtype=np.float64,
        )
        finite_velocity = velocities[np.isfinite(velocities)]
        mean_velocity = (
            float(finite_velocity.mean()) if finite_velocity.size else -float("inf")
        )
        failure_rate = float(failures.mean()) if failures.size else 1.0
        high_z_termination_rate = (
            float(high_z_terminations.mean()) if high_z_terminations.size else 1.0
        )
        inverted_completion_rate = (
            float(inverted_completions.mean()) if inverted_completions.size else 1.0
        )
        gate_pass = bool(
            len(recent) >= minimum
            and finite_velocity.size == len(recent)
            and (mean_velocity >= self.cfg.best_checkpoint_min_forward_velocity)
            and (
                high_z_termination_rate
                <= self.cfg.best_checkpoint_max_high_z_termination_rate
            )
            and (
                inverted_completion_rate
                <= self.cfg.best_checkpoint_max_inverted_completion_rate
            )
            and (failure_rate <= self.cfg.best_checkpoint_max_functional_failure_rate)
        )
        score = (
            mean_velocity if gate_pass else -1000000.0 + mean_velocity - failure_rate
        )
        return (
            score,
            {
                "mode": mode,
                "gate_pass": gate_pass,
                "window_episodes": len(recent),
                "mean_forward_velocity": mean_velocity,
                "velocity_key": self.cfg.best_checkpoint_velocity_key,
                "functional_failure_rate": failure_rate,
                "high_z_termination_rate": high_z_termination_rate,
                "inverted_completion_rate": inverted_completion_rate,
                "minimum_forward_velocity": self.cfg.best_checkpoint_min_forward_velocity,
                "maximum_functional_failure_rate": self.cfg.best_checkpoint_max_functional_failure_rate,
                "maximum_high_z_termination_rate": self.cfg.best_checkpoint_max_high_z_termination_rate,
                "maximum_inverted_completion_rate": self.cfg.best_checkpoint_max_inverted_completion_rate,
                "return_score": return_score,
            },
        )

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
        if self._policy_sampling_generator is not None:
            state = rng.get("policy_sampling")
            if state is None:
                raise ValueError(
                    "separate-RNG checkpoint lacks policy_sampling RNG state"
                )
            self._policy_sampling_generator.set_state(state.detach().cpu())
        if self._minibatch_rng is not None:
            state = rng.get("minibatch")
            if state is None:
                raise ValueError("separate-RNG checkpoint lacks minibatch RNG state")
            self._minibatch_rng.bit_generator.state = state
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

    def _normalize_observation(
        self, observation: np.ndarray, *, update: bool
    ) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if self.obs_rms is None:
            return values
        if update:
            self.obs_rms.update(values)
        return self.obs_rms.normalize(values, clip=float(self.cfg.observation_clip))

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
            (action, log_prob, value, h_next) = self.policy.act(
                obs, h, generator=self._policy_sampling_generator
            )
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
            h = _hidden_after_done(
                self.policy, h_next, done_mask, cfg.initial_hidden_mode
            )
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
                        "terrain_spec_id": str(info_i.get("terrain_spec_id", "")),
                        "layout_seed": int(info_i.get("layout_seed", -1)),
                        "perturbation_type": str(
                            info_i.get("perturbation_type", "none")
                        ),
                        "terminated": bool(info_i.get("terminated", False)),
                        "truncated": bool(info_i.get("truncated", False)),
                        "friction": float(info_i.get("friction", np.nan)),
                        "return": float(self._ep_return_raw[i]),
                        "length": int(self._ep_len[i]),
                        "flip": flipped_i,
                        "episode_metric_schema": "physical_functional_v2",
                        "physical_flip": flipped_i,
                        "physics_error": bool(info_i.get("physics_error", False)),
                        "nan_detected": bool(info_i.get("nan_detected", False)),
                        "training_traversal_failure": str(
                            info_i.get("training_traversal_failure", "")
                        ),
                        "elapsed_s": float(info_i.get("elapsed_s", np.nan)),
                    }
                    if episode_meta["physics_error"]:
                        for key in ("physics_error_type", "physics_error_message"):
                            if key in info_i:
                                episode_meta[key] = str(info_i[key])
                        if "physics_error_after_physics_steps" in info_i:
                            episode_meta["physics_error_after_physics_steps"] = int(
                                info_i["physics_error_after_physics_steps"]
                            )
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
                    numerical_failure = (
                        episode_meta["physics_error"] or episode_meta["nan_detected"]
                    )
                    if numerical_failure and isinstance(
                        info_i.get("numerical_failure_replay"), dict
                    ):
                        episode_meta["numerical_failure_replay"] = info_i[
                            "numerical_failure_replay"
                        ]
                    terrain_info = info_i.get("terrain_spec")
                    course_episode = (
                        isinstance(terrain_info, dict)
                        and terrain_info.get("layout_mode") == "course_v1"
                    )
                    course_failure = course_episode and (
                        not bool(info_i.get("success", False))
                    )
                    episode_meta["functional_failure"] = bool(
                        episode_meta.get("functional_failure", flipped_i)
                        or inverted_completion_i
                        or numerical_failure
                        or course_failure
                    )
                    episode_meta["physical_failure"] = bool(
                        flipped_i
                        or numerical_failure
                        or episode_meta["training_traversal_failure"] == "fall_into_gap"
                        or episode_meta.get("physical_termination", False)
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
            h = _hidden_after_done(
                self.policy, h_next, done_mask, self.cfg.initial_hidden_mode
            )
        return (new_lp, new_val, entropy)

    def _update(self, buf: RolloutBuffer) -> dict:
        cfg = self.cfg
        if cfg.target_kl is not None and (
            not np.isfinite(cfg.target_kl) or cfg.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive")
        actor_parameters = [
            parameter
            for parameter in self.policy.actor.parameters()
            if parameter.requires_grad
        ]
        critic_parameters = [
            parameter
            for parameter in self.policy.critic.parameters()
            if parameter.requires_grad
        ]
        log_std_parameters = [self.policy.log_std]
        actor_before = [parameter.detach().clone() for parameter in actor_parameters]

        def gradient_l2(parameters: list[torch.Tensor]) -> float:
            squared = torch.zeros((), device=self.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared = squared + parameter.grad.detach().pow(2).sum()
            return float(torch.sqrt(squared).item())

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "guard_kl": 0.0,
            "actor_grad_l2_preclip": 0.0,
            "critic_grad_l2_preclip": 0.0,
            "log_std_grad_l2_preclip": 0.0,
            "total_grad_l2_preclip": 0.0,
            "n_minibatches": 0,
            "epochs_completed": 0,
            "kl_early_stop": False,
        }
        stop_early = False
        for _ in range(cfg.n_epochs):
            for batch in buf.iter_chunks(
                cfg.seq_len, cfg.n_minibatches, rng=self._minibatch_rng
            ):
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
                actor_grad = gradient_l2(actor_parameters)
                critic_grad = gradient_l2(critic_parameters)
                log_std_grad = gradient_l2(log_std_parameters)
                stats["actor_grad_l2_preclip"] += actor_grad
                stats["critic_grad_l2_preclip"] += critic_grad
                stats["log_std_grad_l2_preclip"] += log_std_grad
                if cfg.separate_actor_critic_grad_clipping:
                    nn.utils.clip_grad_norm_(
                        actor_parameters + log_std_parameters, cfg.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(critic_parameters, cfg.max_grad_norm)
                    total_grad = np.sqrt(
                        actor_grad**2 + critic_grad**2 + log_std_grad**2
                    )
                else:
                    total_grad_tensor = nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                    total_grad = float(total_grad_tensor.item())
                stats["total_grad_l2_preclip"] += float(total_grad)
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
        for k in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "guard_kl",
            "actor_grad_l2_preclip",
            "critic_grad_l2_preclip",
            "log_std_grad_l2_preclip",
            "total_grad_l2_preclip",
        ):
            stats[k] /= n
        stats["actor_parameter_delta_l2"] = float(
            np.sqrt(
                sum(
                    (
                        float((after.detach() - before).pow(2).sum().item())
                        for (before, after) in zip(actor_before, actor_parameters)
                    )
                )
            )
        )
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
        environment_seed = (
            cfg.seed if cfg.environment_seed is None else int(cfg.environment_seed)
        )
        rng_seeds = [environment_seed + i for i in range(self.env.num_envs)]
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
        if cfg.initial_hidden_mode not in {"legacy_random", "zero", "policy"}:
            raise ValueError(
                "initial_hidden_mode must be 'legacy_random', 'zero', or 'policy'"
            )
        h = self.policy.init_state(self.env.num_envs, self.device)
        if cfg.initial_hidden_mode == "zero":
            h.zero_()
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
                sampled_action_abs_mean=float(buf.actions.abs().mean().item()),
                sampled_action_saturation_fraction=float(
                    (buf.actions.abs() >= 0.95).float().mean().item()
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

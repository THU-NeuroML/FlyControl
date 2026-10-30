"""Shared PPO optimizer with the original terrain episode diagnostics retained."""

from __future__ import annotations
from typing import Optional
import time
import numpy as np
import torch
from fly_control.ppo import (
    RecurrentPPO as SharedPPO,
    RolloutBuffer,
    EPISODE_INFO_METRICS,
    _hidden_after_done,
    _to_tensor,
    _log,
)


class RecurrentPPO(SharedPPO):
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
        # Per-episode metadata: (terrain, return, flipped). The list is
        # what learn() needs to feed per-terrain rolling windows.
        completed_meta_roll: list[dict] = []
        last_terminated_np = np.zeros(self.env.num_envs, dtype=np.bool_)
        tag = (
            f"upd {update:>4d}" + (f"/{n_updates}" if n_updates else "")
            if update is not None
            else "rollout"
        )

        for t in range(cfg.n_steps):
            action, log_prob, value, h_next = self.policy.act(
                obs,
                h,
                generator=self._policy_sampling_generator,
            )
            actions_np = action.detach().cpu().numpy()
            next_obs_raw, rewards_np, dones_np, infos = self.env.step(actions_np)
            terminated_np = np.asarray(
                [bool(info.get("terminated", d)) for info, d in zip(infos, dones_np)],
                dtype=np.bool_,
            )
            timeout_np = np.asarray(
                [bool(info.get("truncated", False)) for info in infos], dtype=np.bool_
            )
            last_terminated_np = terminated_np

            timeout_values_np = np.zeros(self.env.num_envs, dtype=np.float32)
            timeout_indices = [
                i
                for i, is_timeout in enumerate(timeout_np)
                if is_timeout
                and not terminated_np[i]
                and "terminal_observation" in infos[i]
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

            # ---- Reward normalization (gym NormalizeReward style) ----
            # Track per-env discounted return online; update RMS on it;
            # divide raw rewards by sqrt(var). Reset the per-env return
            # accumulator on done. Keeps value loss in O(1).
            if self.ret_rms is not None:
                self._running_returns = cfg.gamma * self._running_returns + rewards_np
                self.ret_rms.update(self._running_returns)
                self._running_returns *= 1.0 - dones_np.astype(np.float64)
                rewards_for_buf = rewards_np / np.sqrt(self.ret_rms.var + 1e-8)
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
            # Reset hidden state for envs that just terminated
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
                    if info_i.get("reward_schema") == "traversal_success_potential_v1":
                        episode_meta["reward_schema"] = str(info_i["reward_schema"])
                        episode_meta["reward_terminal_base"] = float(
                            info_i["reward_terminal_base"]
                        )
                        episode_meta["task_deadline_reached"] = bool(
                            info_i["task_deadline_reached"]
                        )
                        episode_meta["objective_return_residual"] = (
                            episode_meta["return"]
                            - episode_meta["reward_terminal_base"]
                        )
                    if isinstance(info_i.get("terrain_spec"), dict):
                        episode_meta["terrain_kinematics_schema"] = (
                            "flygym_terrain_kinematics_v1"
                        )
                        for key in (
                            "training_progress_fraction",
                            "training_checkpoints_passed",
                            "inverted_duration_s",
                            "max_contiguous_inverted_duration_s",
                        ):
                            if key in info_i and np.isfinite(float(info_i[key])):
                                episode_meta[f"final_{key}"] = float(info_i[key])
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
                        and not info_i.get("terminated", False)
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
                    course_failure = course_episode and not bool(
                        info_i.get("success", False)
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
                    # Legacy aggregate 'flip' includes functional failure.
                    # Scientific physical-flip reporting uses physical_flip.
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
                and (t + 1) < cfg.n_steps
            ):
                elapsed = time.time() - roll_t0
                env_steps_done = (t + 1) * self.env.num_envs
                sps = env_steps_done / max(1e-3, elapsed)
                remaining_outer = cfg.n_steps - (t + 1)
                eta = remaining_outer * self.env.num_envs / max(1e-3, sps)
                mean_len = (
                    float(np.mean(ep_lens_roll)) if ep_lens_roll else float("nan")
                )
                _log(
                    f"  [{tag} {t + 1:>3d}/{cfg.n_steps}] "
                    f"env_steps={env_steps_done:>5d} "
                    f"sps={sps:5.1f} "
                    f"eps={eps_done_roll:>3d} "
                    f"mean_ep_len={mean_len:5.1f} "
                    f"eta_roll={eta:5.0f}s"
                )

        with torch.no_grad():
            _, _, last_value = self.policy.forward(obs, h)
        last_terminated = _to_tensor(last_terminated_np.astype(np.float32), self.device)
        buf.compute_gae(last_value, last_terminated, cfg.gamma, cfg.lam)

        roll_elapsed = time.time() - roll_t0
        total_env_steps = cfg.n_steps * self.env.num_envs
        _log(
            f"  [{tag}] rollout done: {total_env_steps} env_steps in "
            f"{roll_elapsed:.1f}s ({total_env_steps / max(1e-3, roll_elapsed):.1f} sps), "
            f"{eps_done_roll} episodes completed"
        )
        return buf, obs, h, completed_meta_roll

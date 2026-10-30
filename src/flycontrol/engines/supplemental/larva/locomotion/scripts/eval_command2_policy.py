"""Evaluate trained command2 policies on steering/path tasks."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from larva.locomotion.env import TerrainLocomotionEnv
from larva.locomotion.models import build_policy
from larva.locomotion.train import build_connectome, policy_uses_connectome


DEFAULT_TARGET_SPEEDS_MM_S = [8.0, 12.0, 16.0]
ACTION_SATURATION_ABS_THRESHOLD = 0.95
PRIMARY_POST_PUSH_WINDOW_S = 0.5
HIDDEN_INIT_MODES = ("zero", "seeded_random", "legacy_random")
RECURRENT_STATE_MODES = ("intact", "hidden_reset")


def initialize_policy_state(
    policy,
    batch: int,
    device: torch.device,
    *,
    mode: str = "zero",
    seed: int | None = None,
) -> torch.Tensor:
    """Create an evaluation state under an explicit reproducibility protocol.

    ``ConnectomeCfC.init_state`` historically sampled an unseeded random
    state.  Stateless MLP policies did not have this additional source of
    evaluation variation, so a nominally paired replay was not actually
    paired across architectures.  Canonical evaluations therefore use a
    deterministic zero state.  ``seeded_random`` is retained for a separate
    initial-condition robustness analysis, while ``legacy_random`` exists
    only to reproduce already frozen historical artifacts.
    """
    if mode not in HIDDEN_INIT_MODES:
        raise ValueError(
            f"hidden-init mode must be one of {HIDDEN_INIT_MODES}, got {mode!r}"
        )
    if mode == "zero":
        state_dim = getattr(policy, "state_dim", None)
        if state_dim is not None:
            return torch.zeros(batch, int(state_dim), device=device)
        # Small test doubles and legacy policy wrappers may not expose
        # ``state_dim``.  Preserve their shape contract without leaking a
        # random initial state into the rollout.
        return torch.zeros_like(policy.init_state(batch, device))
    if mode == "legacy_random":
        return policy.init_state(batch, device)
    if seed is None:
        raise ValueError("seeded_random hidden initialization requires a seed")

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    # fork_rng prevents evaluation state creation from changing the caller's
    # global RNG stream.
    with torch.random.fork_rng(devices=cuda_devices):
        if device.type == "cuda":
            # ``torch.manual_seed`` would seed every CUDA device.  Restrict the
            # temporary seed to the device that allocates this state so paired
            # evaluation does not perturb unrelated device RNG streams.
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))
        return policy.init_state(batch, device)


def turn_direction_accuracy(target_turn_sign: float, heading_delta_rad: float) -> float:
    return float(target_turn_sign * heading_delta_rad > 0.0)


def direction_accuracy_for_targets(
    target_turn_sign: float | None,
    target_angle_rad: float | None,
    heading_delta_rad: float,
) -> float | str:
    """Compute direction accuracy for steering-sign or reorientation targets."""
    target = target_turn_sign
    if target is None:
        target = target_angle_rad
    if target is None:
        return ""
    return turn_direction_accuracy(float(target), heading_delta_rad)


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("run spec must have form label=path")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise ValueError("run spec must have form label=path")
    return label, path


def training_seed_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", name)
    return match.group(1) if match else ""


def checkpoint_selection_from_name(policy_name: str) -> str:
    """Return an explicit snapshot role without treating initial as best."""
    return {
        "initial_policy.pt": "initial_untrained",
        "policy.pt": "fixed_budget_final",
        "best_policy.pt": "validation_best",
    }.get(policy_name, "explicit_checkpoint")


def summarize_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["task"]))].append(row)

    summary = {}
    for key, items in grouped.items():
        rewards = np.array([float(r["total_reward"]) for r in items], dtype=np.float64)
        flips = np.array([bool(r["flip"]) for r in items], dtype=np.float64)
        acc_values = [
            float(r["turn_direction_accuracy"])
            for r in items
            if r.get("turn_direction_accuracy") not in ("", None)
        ]
        out = {
            "n": len(items),
            "total_reward_mean": float(rewards.mean()),
            "flip_rate": float(flips.mean()),
        }
        if acc_values:
            out["turn_direction_accuracy_mean"] = float(np.mean(acc_values))
        summary[key] = out
    return summary


def _count_params(policy) -> tuple[int, int]:
    parameters = list(policy.parameters()) if hasattr(policy, "parameters") else []
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    return int(total), int(trainable)


def budget_metadata(
    policy,
    cfg: dict,
    *,
    input_dim: int | None = None,
    action_dim: int | None = None,
) -> dict:
    """Return cheap, per-row metadata needed for fair comparisons."""
    total, trainable = _count_params(policy)
    actor = getattr(policy, "actor", None)
    core = getattr(actor, "core", None)
    pol_cfg = cfg.get("policy", {})
    actor_kwargs = pol_cfg.get("actor_kwargs", {})
    conn_cfg = cfg.get("connectome", {})
    ppo_cfg = cfg.get("ppo", {})

    if action_dim is None and hasattr(policy, "log_std"):
        action_dim = int(policy.log_std.numel())
    if input_dim is None and core is not None and hasattr(core, "input_dim"):
        input_dim = int(core.input_dim)

    if core is None:
        core_n = core_s = core_i = core_o = ""
        core_is_sparse = ""
        input_target = actor_kwargs.get("input_target", "")
        matrix_nnz = ""
        w_trainable = ""
    else:
        core_n = int(core.N)
        core_s = int(core.S)
        core_i = int(core.I)
        core_o = int(core.O)
        core_is_sparse = bool(
            getattr(core, "is_sparse", hasattr(core, "edge_weight"))
        )
        input_target = getattr(core, "input_target", "role_correct_masks")
        if core_is_sparse:
            matrix_nnz = int(core.edge_weight.numel())
        else:
            matrix_nnz = int(core.sparsity_mask.sum().item())
        w_trainable = isinstance(getattr(core, "W", None), torch.nn.Parameter)

    return {
        "policy_name": pol_cfg.get("name", ""),
        "policy_mode": actor_kwargs.get("mode", ""),
        "connectome_species": conn_cfg.get("species", "larva" if conn_cfg else ""),
        "connectome_kind": conn_cfg.get("kind", ""),
        "core_n": core_n,
        "core_sensory_n": core_s,
        "core_internal_n": core_i,
        "core_output_n": core_o,
        "core_is_sparse": core_is_sparse,
        "input_target": input_target,
        "matrix_nnz": matrix_nnz,
        "param_count_total": total,
        "param_count_trainable": trainable,
        "W_trainable": w_trainable,
        "input_dim": "" if input_dim is None else int(input_dim),
        "action_dim": "" if action_dim is None else int(action_dim),
        "ppo_total_timesteps": ppo_cfg.get("total_timesteps", ""),
        "ppo_n_envs": ppo_cfg.get("n_envs", ""),
        "ppo_n_steps": ppo_cfg.get("n_steps", ""),
        "ppo_seq_len": ppo_cfg.get("seq_len", ""),
        "ppo_n_epochs": ppo_cfg.get("n_epochs", ""),
        "ppo_n_minibatches": ppo_cfg.get("n_minibatches", ""),
        "ppo_lr": ppo_cfg.get("lr", ""),
    }


def load_policy(policy_path: Path, device: str):
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    connectome = (
        build_connectome(cfg.get("connectome"))
        if policy_uses_connectome(cfg["policy"]["name"])
        else None
    )
    policy = build_policy(
        cfg["policy"]["name"],
        ckpt["obs_dim"],
        ckpt["action_dim"],
        actor_kwargs=cfg["policy"].get("actor_kwargs"),
        value_hidden=cfg["policy"].get("value_hidden"),
        log_std_init=cfg["policy"].get("log_std_init", -0.5),
        connectome=connectome,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    # Preserve the exact S/I/O state ordering for downstream neural analyses.
    # ``connectome`` is intentionally not retained: its sparse edge arrays are
    # already registered by the policy core and keeping both copies wastes
    # several hundred MB for the adult graph.
    if connectome is not None:
        policy._connectome_state_index = {
            "sensory_ids": connectome.sensory_ids,
            "internal_ids": connectome.internal_ids,
            "output_ids": connectome.output_ids,
            "metadata": dict(connectome.metadata or {}),
        }
    return policy, cfg


def make_eval_env_config(
    cfg: dict,
    task: str,
    terrain: str,
    eval_seed: int,
    max_time: float | None,
    action_mode: str | None = None,
) -> dict:
    env_cfg = dict(cfg.get("env", {}))
    env_cfg.update({
        "terrains": [terrain],
        "terrain_seeds": [eval_seed],
        "task": task,
        "observation_mode": "proprio_task",
    })
    env_cfg["action_mode"] = action_mode or env_cfg.get("action_mode", "command2")
    if max_time is not None:
        env_cfg["max_time"] = max_time
    return env_cfg


def make_eval_env(
    cfg: dict,
    task: str,
    terrain: str,
    eval_seed: int,
    max_time: float | None,
    action_mode: str | None = None,
) -> TerrainLocomotionEnv:
    env_cfg = make_eval_env_config(
        cfg,
        task=task,
        terrain=terrain,
        eval_seed=eval_seed,
        max_time=max_time,
        action_mode=action_mode,
    )
    return TerrainLocomotionEnv(**env_cfg)


def rollout_policy(
    *,
    model: str,
    run_dir: Path,
    policy,
    cfg: dict,
    env: TerrainLocomotionEnv | None = None,
    task: str,
    terrain: str,
    eval_seed: int,
    target_turn_sign: float | None,
    target_angle_rad: float | None,
    target_speed_mm_s: float | None,
    device: str,
    max_time: float | None,
    action_mode: str | None = None,
    hidden_init_mode: str = "zero",
    hidden_seed: int | None = None,
    recurrent_state_mode: str = "intact",
) -> dict:
    if recurrent_state_mode not in RECURRENT_STATE_MODES:
        raise ValueError(
            "recurrent-state mode must be one of "
            f"{RECURRENT_STATE_MODES}, got {recurrent_state_mode!r}"
        )
    close_env = env is None
    if env is None:
        env = make_eval_env(
            cfg, task, terrain, eval_seed, max_time, action_mode=action_mode)
    hidden_seed_effective = (
        int(eval_seed) if hidden_seed is None and hidden_init_mode == "seeded_random"
        else hidden_seed
    )
    hidden = initialize_policy_state(
        policy,
        1,
        torch.device(device),
        mode=hidden_init_mode,
        seed=hidden_seed_effective,
    )
    reset_options = {}
    if target_turn_sign is not None:
        reset_options["target_turn_sign"] = target_turn_sign
    if target_angle_rad is not None:
        reset_options["target_angle_rad"] = target_angle_rad
    if target_speed_mm_s is not None:
        reset_options["target_speed_mm_s"] = target_speed_mm_s
    if not reset_options:
        reset_options = None
    try:
        obs, info = env.reset(seed=eval_seed, options=reset_options)
        initial_obs_dim = int(np.asarray(obs).shape[-1])
        target_turn_command = info.get("target_turn_command", target_turn_sign)
        target_turn_strength = info.get(
            "target_turn_strength",
            abs(float(target_turn_command)) if target_turn_command not in ("", None) else "",
        )
        target_angle = info.get("target_angle_rad", target_angle_rad)
        target_speed = info.get("target_speed_mm_s", target_speed_mm_s)
        done = False
        total_reward = 0.0
        forward_distance = 0.0
        heading_delta = 0.0
        path_error_abs_sum = 0.0
        path_heading_error_abs_sum = 0.0
        heading_error_abs_sum = 0.0
        heading_error_abs_count = 0
        final_heading_error_abs = ""
        speed_error_abs_sum = 0.0
        speed_error_abs_count = 0
        speed_error_abs_max: float | None = None
        speed_error_abs_auc = 0.0
        command_l2_sum = 0.0
        gait_coordinate_values: list[float] = []
        duty_factor_values: list[float] = []
        ipsilateral_lag_values: list[float] = []
        cpg_frequency_values: list[float] = []
        action_l2_values: list[float] = []
        action_delta_l2_values: list[float] = []
        action_saturated_count = 0
        action_element_count = 0
        prev_action: np.ndarray | None = None
        steering_rate_error_abs_sum = 0.0
        steering_rate_error_abs_count = 0
        heading_error_abs_max: float | None = None
        heading_error_abs_auc = 0.0
        heading_success_time_s: float | None = None
        perturbation_type = "none"
        perturbation_active_steps = 0
        perturbation_seen = False
        post_perturb_elapsed = 0.0
        post_perturb_errors: list[float] = []
        post_perturb_heading_errors: list[float] = []
        post_perturb_heading_error_abs_max: float | None = None
        post_push_auc = 0.0
        post_push_elapsed = 0.0
        post_push_valid_duration = 0.0
        push_onset_seen = False
        post_perturb_0p5s_auc = 0.0
        post_perturb_0p5s_elapsed = 0.0
        post_perturb_0p5s_valid_duration = 0.0
        perturbation_recovery_time_s: float | None = None
        heading_perturbation_recovery_time_s: float | None = None
        steps = 0
        flipped = False
        env_step_dt = env.decision_interval * env.timestep

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
                if recurrent_state_mode == "hidden_reset":
                    hidden = torch.zeros_like(hidden)
                action_t, _, _, hidden = policy.act(obs_t, hidden, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            forward_distance += float(info.get("forward_velocity_mm_s", 0.0)) * env_step_dt
            heading_delta += float(info.get("heading_delta_rad", 0.0))
            path_error_abs_sum += abs(float(info.get("path_error_mm", 0.0)))
            path_heading_error_abs_sum += abs(float(info.get("path_heading_error_rad", 0.0)))
            action_l2 = float(np.linalg.norm(action))
            command_l2_sum += action_l2
            for key, target_list in (
                ("gait_coordinate", gait_coordinate_values),
                ("duty_factor_command", duty_factor_values),
                ("ipsilateral_lag_cycles_command", ipsilateral_lag_values),
                ("cpg_frequency_mean_hz", cpg_frequency_values),
            ):
                value = info.get(key)
                if value not in ("", None):
                    target_list.append(float(value))
            action_l2_values.append(action_l2)
            if prev_action is not None:
                action_delta_l2_values.append(float(np.linalg.norm(action - prev_action)))
            prev_action = action.copy()
            action_saturated_count += int(
                np.count_nonzero(np.abs(action) >= ACTION_SATURATION_ABS_THRESHOLD)
            )
            action_element_count += int(action.size)
            if info.get("target_turn_command") not in ("", None):
                target_turn_command = info["target_turn_command"]
            if info.get("target_turn_strength") not in ("", None):
                target_turn_strength = info["target_turn_strength"]
            if info.get("target_angle_rad") not in ("", None):
                target_angle = info["target_angle_rad"]
            if info.get("target_speed_mm_s") not in ("", None):
                target_speed = info["target_speed_mm_s"]
            rate_error = info.get("steering_rate_error_abs")
            if rate_error not in ("", None):
                steering_rate_error_abs_sum += float(rate_error)
                steering_rate_error_abs_count += 1
            heading_error = info.get("heading_error_abs_rad")
            if heading_error not in ("", None):
                final_heading_error_abs = float(heading_error)
                heading_error_abs_sum += final_heading_error_abs
                heading_error_abs_count += 1
                heading_error_abs_auc += final_heading_error_abs * env_step_dt
                heading_error_abs_max = (
                    final_heading_error_abs
                    if heading_error_abs_max is None
                    else max(heading_error_abs_max, final_heading_error_abs)
                )
                if heading_success_time_s is None and final_heading_error_abs <= 0.25:
                    heading_success_time_s = (steps + 1) * env_step_dt
            speed_error = info.get("speed_error_abs_mm_s")
            if speed_error not in ("", None):
                speed_error_f = float(speed_error)
                speed_error_abs_sum += speed_error_f
                speed_error_abs_count += 1
                speed_error_abs_auc += speed_error_f * env_step_dt
                speed_error_abs_max = (
                    speed_error_f
                    if speed_error_abs_max is None
                    else max(speed_error_abs_max, speed_error_f)
                )
            perturbation_type = str(info.get("perturbation_type", perturbation_type))
            perturbation_active = bool(info.get("perturbation_active", False))
            if perturbation_active:
                push_onset_seen = True
                perturbation_seen = True
                perturbation_active_steps += 1
            elif perturbation_seen:
                if rate_error not in ("", None):
                    err = float(rate_error)
                    post_perturb_errors.append(err)
                    if perturbation_recovery_time_s is None and err <= 0.5:
                        perturbation_recovery_time_s = post_perturb_elapsed
                if heading_error not in ("", None):
                    err = float(heading_error)
                    post_perturb_heading_errors.append(err)
                    post_perturb_heading_error_abs_max = (
                        err
                        if post_perturb_heading_error_abs_max is None
                        else max(post_perturb_heading_error_abs_max, err)
                    )
                    if (
                        heading_perturbation_recovery_time_s is None
                        and err <= 0.25
                    ):
                        heading_perturbation_recovery_time_s = post_perturb_elapsed
                post_perturb_elapsed += env_step_dt
            if push_onset_seen and post_push_elapsed < PRIMARY_POST_PUSH_WINDOW_S:
                window_dt = min(
                    env_step_dt,
                    PRIMARY_POST_PUSH_WINDOW_S - post_push_elapsed,
                )
                post_push_elapsed += window_dt
                if heading_error not in ("", None):
                    post_push_auc += float(heading_error) * window_dt
                    post_push_valid_duration += window_dt
            if (
                perturbation_seen
                and not perturbation_active
                and post_perturb_0p5s_elapsed < PRIMARY_POST_PUSH_WINDOW_S
            ):
                window_dt = min(
                    env_step_dt,
                    PRIMARY_POST_PUSH_WINDOW_S - post_perturb_0p5s_elapsed,
                )
                post_perturb_0p5s_elapsed += window_dt
                if heading_error not in ("", None):
                    post_perturb_0p5s_auc += float(heading_error) * window_dt
                    post_perturb_0p5s_valid_duration += window_dt
            flipped = flipped or bool(info.get("flip", False))
            steps += 1

        target_command_f = (
            float(target_turn_command)
            if target_turn_command not in ("", None)
            else None
        )
        target_strength_f = (
            float(target_turn_strength)
            if target_turn_strength not in ("", None)
            else None
        )
        target_sign = (
            float(np.sign(target_command_f))
            if target_command_f is not None
            else None
        )
        aligned_heading = (
            target_sign * heading_delta
            if target_sign is not None
            else ""
        )
        normalized_heading = (
            aligned_heading / target_strength_f
            if target_strength_f not in (None, 0.0)
            else ""
        )
        command_l2_mean = command_l2_sum / max(steps, 1)
        duration_s = steps * env_step_dt
        forward_speed_mean = forward_distance / max(duration_s, 1e-9)
        normalized_heading_per_command = (
            normalized_heading / command_l2_mean
            if normalized_heading != "" and command_l2_mean > 1e-9
            else ""
        )
        target_angle_f = (
            float(target_angle)
            if target_angle not in ("", None)
            else None
        )
        heading_gain = (
            heading_delta / target_angle_f
            if target_angle_f not in (None, 0.0)
            else ""
        )
        heading_gain_error_abs = (
            abs(heading_gain - 1.0)
            if heading_gain != ""
            else ""
        )
        heading_overshoot = (
            float(abs(heading_delta) > abs(target_angle_f) * 1.15)
            if target_angle_f not in (None, 0.0)
            else ""
        )
        heading_undershoot = (
            float(abs(heading_delta) < abs(target_angle_f) * 0.85)
            if target_angle_f not in (None, 0.0)
            else ""
        )
        row = {
            "model": model,
            "run_dir": str(run_dir),
            "train_seed": training_seed_from_run_dir(run_dir),
            "policy": str(run_dir / "best_policy.pt"),
            "task": task,
            "action_mode": getattr(
                env,
                "action_mode",
                action_mode or cfg.get("env", {}).get("action_mode", "command2"),
            ),
            "terrain": terrain,
            "eval_seed": eval_seed,
            "hidden_init_mode": hidden_init_mode,
            "hidden_seed": (
                ""
                if hidden_init_mode != "seeded_random"
                else int(hidden_seed_effective)
            ),
            "recurrent_state_mode": recurrent_state_mode,
            "target_turn_sign": "" if target_turn_sign is None else float(target_turn_sign),
            "target_turn_command": "" if target_command_f is None else target_command_f,
            "target_turn_strength": "" if target_strength_f is None else target_strength_f,
            "target_angle_rad": "" if target_angle in ("", None) else float(target_angle),
            "target_speed_mm_s": "" if target_speed in ("", None) else float(target_speed),
            "total_reward": total_reward,
            "steps": steps,
            "duration_s": duration_s,
            "forward_distance_mm": forward_distance,
            "forward_speed_mean_mm_s": forward_speed_mean,
            "heading_delta_rad": heading_delta,
            "target_aligned_heading_delta_rad": aligned_heading,
            "target_normalized_heading_delta_rad": normalized_heading,
            "target_normalized_heading_per_command_l2": normalized_heading_per_command,
            "heading_gain": heading_gain,
            "heading_gain_error_abs": heading_gain_error_abs,
            "heading_overshoot": heading_overshoot,
            "heading_undershoot": heading_undershoot,
            "turn_direction_accuracy": direction_accuracy_for_targets(
                target_turn_sign, target_angle_f, heading_delta
            ),
            "path_error_abs_mean_mm": "",
            "path_heading_error_abs_mean_rad": "",
            "heading_error_abs_mean_rad": (
                heading_error_abs_sum / heading_error_abs_count
                if heading_error_abs_count else ""
            ),
            "heading_error_abs_max_rad": (
                "" if heading_error_abs_max is None else heading_error_abs_max
            ),
            "heading_error_abs_auc_rad_s": (
                heading_error_abs_auc if heading_error_abs_count else ""
            ),
            "heading_success_time_s": (
                "" if heading_success_time_s is None else heading_success_time_s
            ),
            "final_heading_error_abs_rad": final_heading_error_abs,
            "heading_success": (
                float(final_heading_error_abs <= 0.25)
                if final_heading_error_abs != "" else ""
            ),
            "speed_error_abs_mean_mm_s": (
                speed_error_abs_sum / speed_error_abs_count
                if speed_error_abs_count else ""
            ),
            "speed_success": (
                float(speed_error_abs_sum / speed_error_abs_count <= 2.0)
                if speed_error_abs_count else ""
            ),
            "speed_error_abs_max_mm_s": (
                "" if speed_error_abs_max is None else speed_error_abs_max
            ),
            "speed_error_abs_auc_mm": (
                speed_error_abs_auc if speed_error_abs_count else ""
            ),
            "command_l2_mean": command_l2_mean,
            "gait_coordinate_mean": (
                float(np.mean(gait_coordinate_values))
                if gait_coordinate_values else ""
            ),
            "duty_factor_command_mean": (
                float(np.mean(duty_factor_values)) if duty_factor_values else ""
            ),
            "ipsilateral_lag_cycles_command_mean": (
                float(np.mean(ipsilateral_lag_values))
                if ipsilateral_lag_values else ""
            ),
            "cpg_frequency_mean_hz": (
                float(np.mean(cpg_frequency_values))
                if cpg_frequency_values else ""
            ),
            "action_l2_max": float(np.max(action_l2_values)) if action_l2_values else "",
            "action_l2_std": float(np.std(action_l2_values)) if action_l2_values else "",
            "action_delta_l2_mean": (
                float(np.mean(action_delta_l2_values))
                if action_delta_l2_values else ""
            ),
            "action_delta_l2_max": (
                float(np.max(action_delta_l2_values))
                if action_delta_l2_values else ""
            ),
            "action_delta_l2_std": (
                float(np.std(action_delta_l2_values))
                if action_delta_l2_values else ""
            ),
            "action_saturation_fraction": (
                action_saturated_count / action_element_count
                if action_element_count else ""
            ),
            "steering_rate_error_abs_mean": (
                steering_rate_error_abs_sum / steering_rate_error_abs_count
                if steering_rate_error_abs_count else ""
            ),
            "perturbation_type": perturbation_type,
            "perturbation_active_steps": perturbation_active_steps,
            "post_perturb_steering_rate_error_abs_mean": (
                float(np.mean(post_perturb_errors)) if post_perturb_errors else ""
            ),
            "post_perturb_heading_error_abs_mean_rad": (
                float(np.mean(post_perturb_heading_errors))
                if post_perturb_heading_errors else ""
            ),
            "post_perturb_heading_error_abs_max_rad": (
                "" if post_perturb_heading_error_abs_max is None
                else post_perturb_heading_error_abs_max
            ),
            "post_push_0p5s_heading_error_abs_auc_rad_s": (
                post_push_auc if post_push_valid_duration > 0.0 else ""
            ),
            "post_push_0p5s_observed_duration_s": (
                post_push_valid_duration if push_onset_seen else ""
            ),
            "post_perturb_0p5s_heading_error_abs_auc_rad_s": (
                post_perturb_0p5s_auc
                if post_perturb_0p5s_valid_duration > 0.0 else ""
            ),
            "post_perturb_0p5s_observed_duration_s": (
                post_perturb_0p5s_valid_duration if perturbation_seen else ""
            ),
            "primary_heading_error_auc_rad_s": (
                post_push_auc if post_push_valid_duration > 0.0 else ""
            ),
            "primary_heading_error_auc_failure_adjusted_rad_s": (
                post_push_auc
                + math.pi
                * max(0.0, PRIMARY_POST_PUSH_WINDOW_S - post_push_valid_duration)
                if post_push_valid_duration > 0.0
                else ""
            ),
            "primary_heading_error_mean_failure_adjusted_rad": (
                (
                    post_push_auc
                    + math.pi
                    * max(
                        0.0,
                        PRIMARY_POST_PUSH_WINDOW_S - post_push_valid_duration,
                    )
                )
                / PRIMARY_POST_PUSH_WINDOW_S
                if post_push_valid_duration > 0.0
                else ""
            ),
            "primary_heading_error_missing_window_penalty_rad": math.pi,
            "primary_heading_error_window_s": (
                PRIMARY_POST_PUSH_WINDOW_S if push_onset_seen else ""
            ),
            "perturbation_recovery_time_s": (
                "" if perturbation_recovery_time_s is None
                else perturbation_recovery_time_s
            ),
            "heading_perturbation_recovery_time_s": (
                "" if heading_perturbation_recovery_time_s is None
                else heading_perturbation_recovery_time_s
            ),
        "flip": flipped,
            "perturbation_leg": info.get("perturbation_leg", ""),
            "perturbation_body": info.get("perturbation_body", ""),
            "perturbation_side": info.get("perturbation_side", ""),
            "perturbation_force_frame": info.get("perturbation_force_frame", ""),
        }
        row.update(budget_metadata(
            policy,
            cfg,
            input_dim=initial_obs_dim,
            action_dim=int(action.shape[-1]),
        ))
        if task == "path_sine":
            row["path_error_abs_mean_mm"] = path_error_abs_sum / max(steps, 1)
            row["path_heading_error_abs_mean_rad"] = (
                path_heading_error_abs_sum / max(steps, 1)
            )
        return row
    finally:
        if close_env:
            env.close()


def evaluate_policy_run(
    *,
    model: str,
    run_dir: Path,
    policy,
    cfg: dict,
    task: str,
    terrains: list[str],
    seeds: list[int],
    steering_targets: list[float],
    target_angles_rad: list[float],
    target_speeds_mm_s: list[float],
    device: str,
    max_time: float | None,
    action_mode: str | None = None,
    hidden_init_mode: str = "zero",
    hidden_seed_offset: int = 0,
    env_factory=make_eval_env,
) -> list[dict]:
    rows = []
    if task == "steering_target":
        targets = [{"target_turn_sign": target} for target in steering_targets]
    elif task == "reorientation":
        targets = [{"target_angle_rad": target} for target in target_angles_rad]
    elif task == "speed_control":
        targets = [{"target_speed_mm_s": target} for target in target_speeds_mm_s]
    else:
        targets = [{}]
    for terrain in terrains:
        for seed in seeds:
            env = (
                env_factory(cfg, task, terrain, seed, max_time)
                if action_mode is None
                else env_factory(cfg, task, terrain, seed, max_time, action_mode)
            )
            try:
                for target_kwargs in targets:
                    rows.append(rollout_policy(
                        model=model,
                        run_dir=run_dir,
                        policy=policy,
                        cfg=cfg,
                        env=env,
                        task=task,
                        terrain=terrain,
                        eval_seed=seed,
                        target_turn_sign=target_kwargs.get("target_turn_sign"),
                        target_angle_rad=target_kwargs.get("target_angle_rad"),
                        target_speed_mm_s=target_kwargs.get("target_speed_mm_s"),
                        device=device,
                        max_time=max_time,
                        action_mode=action_mode,
                        hidden_init_mode=hidden_init_mode,
                        hidden_seed=(
                            int(seed) + int(hidden_seed_offset)
                            if hidden_init_mode == "seeded_random"
                            else None
                        ),
                    ))
            finally:
                env.close()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no evaluation rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more label=run_dir entries.")
    parser.add_argument(
        "--task",
        choices=[
            "steering_target",
            "path_sine",
            "reorientation",
            "heading_hold",
            "speed_control",
            "stumble_recovery",
        ],
        required=True,
    )
    parser.add_argument("--terrains", nargs="+", default=["flat"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--steering-targets", nargs="+", type=float, default=[-1.0, 1.0])
    parser.add_argument(
        "--target-angles-rad",
        nargs="+",
        type=float,
        default=[-1.57079632679, -1.0471975512, -0.5235987756,
                 0.5235987756, 1.0471975512, 1.57079632679],
    )
    parser.add_argument(
        "--target-speeds-mm-s",
        nargs="+",
        type=float,
        default=DEFAULT_TARGET_SPEEDS_MM_S,
    )
    parser.add_argument("--policy-name", default="best_policy.pt")
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument(
        "--action-mode",
        choices=["command2", "command3", "correction6", "local12", "hybrid14"],
        default=None,
        help="Override checkpoint env.action_mode. Default: use checkpoint config.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hidden-init-mode",
        choices=HIDDEN_INIT_MODES,
        default="zero",
        help=(
            "Evaluation recurrent-state protocol. Canonical default is zero; "
            "legacy_random is only for reproducing historical artifacts."
        ),
    )
    parser.add_argument(
        "--hidden-seed-offset",
        type=int,
        default=0,
        help="For seeded_random, hidden seed = eval seed + this offset.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.runs:
        model, run_dir_s = parse_run_spec(spec)
        run_dir = Path(run_dir_s)
        policy_path = run_dir / args.policy_name
        policy, cfg = load_policy(policy_path, args.device)
        run_rows = evaluate_policy_run(
            model=model,
            run_dir=run_dir,
            policy=policy,
            cfg=cfg,
            task=args.task,
            terrains=args.terrains,
            seeds=args.seeds,
            steering_targets=args.steering_targets,
            target_angles_rad=args.target_angles_rad,
            target_speeds_mm_s=args.target_speeds_mm_s,
            device=args.device,
            max_time=args.max_time,
            action_mode=args.action_mode,
            hidden_init_mode=args.hidden_init_mode,
            hidden_seed_offset=args.hidden_seed_offset,
        )
        for row in run_rows:
            # Keep checkpoint selection explicit in every row.  The previous
            # writer always advertised best_policy.pt even when a fixed final
            # policy was loaded, which made confirmatory provenance ambiguous.
            row["policy"] = str(policy_path)
            row["checkpoint_file"] = args.policy_name
            row["checkpoint_selection"] = checkpoint_selection_from_name(
                args.policy_name
            )
            target = ""
            for key in ("target_turn_sign", "target_angle_rad", "target_speed_mm_s"):
                if row[key] not in ("", None):
                    target = row[key]
                    break
            print(
                f"{model} {args.task} {row['terrain']} seed={row['eval_seed']} "
                f"target={target} "
                f"reward={row['total_reward']:.1f} flip={row['flip']}"
            )
        rows.extend(run_rows)

    write_csv(args.output, rows)
    print(f"saved {len(rows)} rows to {args.output}")
    for (model, task), summary in sorted(summarize_rows(rows).items()):
        print(f"{model} {task}: {summary}")


if __name__ == "__main__":
    main()

"""Frozen-policy evaluation across held-out Ant friction contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .env import ContinualAntEnv
from .train import build_policy_from_config


def enable_deterministic_evaluation() -> None:
    """Select deterministic CUDA kernels before the first policy operation.

    The adult sparse recurrence uses a many-to-one ``index_add_``.  Its fast
    CUDA kernel may otherwise change floating-point accumulation order between
    processes; an unstable learned gait can amplify that tiny difference into
    a different fall outcome.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _normalize_observation(
    observation: np.ndarray,
    checkpoint_data: dict,
    config: dict,
) -> np.ndarray:
    ppo = config.get("ppo", {})
    offpolicy = config.get("algorithm") in {
        "recurrent_td3_future_her",
        "recurrent_sac_future_her",
        "antmaze_teacher_distillation",
    }
    if not offpolicy and not ppo.get("normalize_observation", False):
        return np.asarray(observation, dtype=np.float32)
    state = checkpoint_data.get("obs_rms_state")
    if state is None:
        raise ValueError("checkpoint lacks observation statistics required by config")
    mean = np.asarray(state["mean"], dtype=np.float64)
    var = np.asarray(state["var"], dtype=np.float64)
    clip = float(
        config.get("observation_clip", 10.0)
        if offpolicy
        else ppo.get("observation_clip", 10.0)
    )
    normalized = (np.asarray(observation, dtype=np.float64) - mean) / np.sqrt(
        var + 1e-8
    )
    return np.clip(normalized, -clip, clip).astype(np.float32)


def evaluate_run(
    run_dir: Path,
    frictions: list[float],
    *,
    episodes: int,
    seed: int,
    device: str,
    checkpoint: str = "latest",
    action_scale: float | None = None,
) -> list[dict]:
    enable_deterministic_evaluation()
    with (run_dir / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    checkpoint_paths = {
        "latest": run_dir / "checkpoint.pt",
        "best": run_dir / "best_checkpoint.pt",
        "initial": run_dir / "initial_checkpoint.pt",
    }
    checkpoint_path = checkpoint_paths[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Requested {checkpoint} checkpoint does not exist: " f"{checkpoint_path}"
        )
    checkpoint_data = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    training_env = config.get("resolved_env", config.get("env", {}))
    append_goal_features = bool(training_env.get("append_zero_goal_features", False))
    training_stabilization = training_env.get("stabilization") or {}
    selected_action_scale = float(
        training_stabilization.get("action_scale", 1.0)
        if action_scale is None
        else action_scale
    )
    evaluation_stabilization = {
        "enabled": False,
        "action_scale": selected_action_scale,
    }

    probe = ContinualAntEnv(
        friction_values=[frictions[0]],
        steps_per_context=10**12,
        append_zero_goal_features=append_goal_features,
        stabilization=evaluation_stabilization,
    )
    policy = build_policy_from_config(
        config, probe.observation_space.shape[0], probe.action_space.shape[0]
    )
    probe.close()
    policy.load_state_dict(checkpoint_data["policy_state"])
    policy.to(device).eval()

    records: list[dict] = []
    for friction_index, friction in enumerate(frictions):
        env = ContinualAntEnv(
            friction_values=[friction],
            steps_per_context=10**12,
            append_zero_goal_features=append_goal_features,
            stabilization=evaluation_stabilization,
        )
        try:
            for episode in range(episodes):
                observation, _ = env.reset(
                    seed=seed + friction_index * 10_000 + episode
                )
                # Canonical evaluation starts from a deterministic rest state.
                # The rate actor's legacy init_state samples small random
                # values, which would otherwise make nominally paired Ant
                # seeds differ for reasons unrelated to the environment.
                hidden = torch.zeros(
                    1, policy.state_dim, dtype=torch.float32, device=device
                )
                episode_return = 0.0
                forward_velocity: list[float] = []
                action_cost: list[float] = []
                upright_factors: list[float] = []
                torso_tilts: list[float] = []
                torso_heights: list[float] = []
                speeds_xy: list[float] = []
                start_x = start_y = final_x = final_y = float("nan")
                terminated = truncated = False
                length = 0
                while not (terminated or truncated):
                    obs_tensor = torch.as_tensor(
                        _normalize_observation(observation, checkpoint_data, config),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _, hidden = policy.act(
                            obs_tensor, hidden, deterministic=True
                        )
                    observation, reward, terminated, truncated, info = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    episode_return += float(reward)
                    length += 1
                    if "x_velocity" in info:
                        forward_velocity.append(float(info["x_velocity"]))
                    if "reward_ctrl" in info:
                        action_cost.append(-float(info["reward_ctrl"]))
                    if "upright_factor" in info:
                        upright_factors.append(float(info["upright_factor"]))
                    if "torso_tilt_degrees" in info:
                        torso_tilts.append(float(info["torso_tilt_degrees"]))
                    if "torso_height" in info:
                        torso_heights.append(float(info["torso_height"]))
                    if "x_velocity" in info and "y_velocity" in info:
                        speeds_xy.append(
                            float(np.hypot(info["x_velocity"], info["y_velocity"]))
                        )
                    if "x_position" in info and "y_position" in info:
                        final_x = float(info["x_position"])
                        final_y = float(info["y_position"])
                        if not np.isfinite(start_x):
                            start_x, start_y = final_x, final_y
                final_inverted = bool(torso_tilts and torso_tilts[-1] > 90.0)
                inverted_completion = bool(not terminated and final_inverted)
                functional_failure = bool(
                    info.get("functional_failure", terminated) or inverted_completion
                )
                records.append(
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_path": str(checkpoint_path),
                        "friction": float(friction),
                        "episode": episode,
                        "seed": seed + friction_index * 10_000 + episode,
                        "action_scale": selected_action_scale,
                        "return": episode_return,
                        "length": length,
                        "fall": bool(terminated),
                        "high_z_termination": bool(
                            info.get("high_z_termination", False)
                        ),
                        "low_z_termination": bool(info.get("low_z_termination", False)),
                        "nonfinite_termination": bool(
                            info.get("nonfinite_termination", False)
                        ),
                        "inverted_completion": inverted_completion,
                        "functional_failure": functional_failure,
                        "mean_x_velocity": (
                            float(np.mean(forward_velocity))
                            if forward_velocity
                            else float("nan")
                        ),
                        "mean_speed_xy": (
                            float(np.mean(speeds_xy)) if speeds_xy else float("nan")
                        ),
                        "mean_action_cost": (
                            float(np.mean(action_cost)) if action_cost else float("nan")
                        ),
                        "mean_upright_factor": (
                            float(np.mean(upright_factors))
                            if upright_factors
                            else float("nan")
                        ),
                        "mean_torso_tilt_degrees": (
                            float(np.mean(torso_tilts)) if torso_tilts else float("nan")
                        ),
                        "max_torso_tilt_degrees": (
                            float(np.max(torso_tilts)) if torso_tilts else float("nan")
                        ),
                        "final_torso_tilt_degrees": (
                            float(torso_tilts[-1]) if torso_tilts else float("nan")
                        ),
                        "max_torso_height": (
                            float(np.max(torso_heights))
                            if torso_heights
                            else float("nan")
                        ),
                        "final_torso_height": (
                            float(torso_heights[-1]) if torso_heights else float("nan")
                        ),
                        "forward_displacement": final_x - start_x,
                        "lateral_displacement": final_y - start_y,
                    }
                )
        finally:
            env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frictions", type=float, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        choices=["latest", "best", "initial"],
        default="latest",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Controller output scale; defaults to the run configuration",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = evaluate_run(
        args.run,
        args.frictions,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        action_scale=args.action_scale,
    )
    output = args.output or args.run / "heldout_friction_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(records, file, indent=2)
    os.replace(temporary, output)

    for friction in args.frictions:
        rows = [row for row in records if row["friction"] == friction]
        print(
            f"friction={friction:g} "
            f"return={np.mean([r['return'] for r in rows]):.1f} "
            f"fall={np.mean([r['fall'] for r in rows]):.2f} "
            f"vx={np.mean([r['mean_x_velocity'] for r in rows]):.3f} "
            f"action_cost={np.mean([r['mean_action_cost'] for r in rows]):.3f}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

"""Train MLP/recurrent/connectome actors on a continual Ant-v5 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import torch
import yaml

from body_control.policy import build_policy
from body_control.ppo import PPOConfig, RecurrentPPO
from body_control.graph import build_connectome
from body_control.graph import diagnose_connectome
from body_control.factories import make_env_from_spec, make_vec_env


ENV_CLASS_PATH = "body_control.env.ContinualAntEnv"
CONNECTOME_POLICY_NAMES = frozenset({"connectome_rate"})


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_provenance() -> dict:
    root = Path(__file__).resolve().parent
    return {
        "source_files_sha256": {
            p.name: file_sha256(p) for p in sorted(root.glob("*.py"))
        },
        "git_revision": git_revision(),
    }


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as file:
        json.dump(value, file, indent=2)


def load_config(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


def resolved_env_config(config: dict, *, seed: int, n_envs: int) -> dict:
    """Resolve a global switch budget to an equal per-worker budget."""
    env_config = dict(config.get("env", {}))
    env_config.pop("class_path", None)
    global_budget = env_config.pop("global_steps_per_context", None)
    if global_budget is not None:
        if "steps_per_context" in env_config:
            raise ValueError(
                "Set only env.global_steps_per_context or steps_per_context"
            )
        env_config["steps_per_context"] = max(
            1, math.ceil(int(global_budget) / int(n_envs))
        )
    if env_config.get("env_id", "Ant-v5") == "Ant-v5":
        env_config.setdefault("schedule_seed", int(seed))
    return env_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_policy_from_config(
    config: dict,
    obs_dim: int,
    action_dim: int,
    *,
    connectome=None,
):
    policy_config = config["policy"]
    if connectome is None and policy_config["name"] in CONNECTOME_POLICY_NAMES:
        connectome = build_connectome(config.get("connectome"))
    policy = build_policy(
        policy_config["name"],
        obs_dim,
        action_dim,
        actor_kwargs=policy_config.get("actor_kwargs"),
        value_hidden=policy_config.get("value_hidden"),
        value_activation=policy_config.get("value_activation", "tanh"),
        log_std_init=policy_config.get("log_std_init", -0.5),
        connectome=connectome,
    )
    trailing = int(
        policy_config.get("zero_initialize_trailing_observation_features", 0)
    )
    if trailing:
        zero_policy_input_features(policy, trailing)
    return policy


def zero_policy_input_features(policy, trailing_features: int) -> None:
    """Zero newly reserved goal-input columns in actor and critic."""
    if trailing_features <= 0:
        raise ValueError("trailing_features must be positive")
    actor = policy.actor
    if type(actor).__module__.endswith("extension_actors"):
        weight = actor.cell.weight_ih_l0 if hasattr(actor, "cell") else actor.input.weight if hasattr(actor, "input") else actor.input_weight
        if not isinstance(weight, torch.nn.Parameter):
            raise ValueError("Cannot zero the fixed readout-only input projection")
        with torch.no_grad():
            weight[:, -trailing_features:].zero_()
            policy.critic.net[0].weight[:, -trailing_features:].zero_()
        return
    if hasattr(actor, "core") and hasattr(actor.core, "input_proj"):
        actor_input = actor.core.input_proj
    elif hasattr(actor, "net") and isinstance(actor.net[0], torch.nn.Linear):
        actor_input = actor.net[0]
    else:
        raise TypeError(f"Cannot locate actor input layer for {type(actor).__name__}")
    critic_input = policy.critic.net[0]
    for layer in (actor_input, critic_input):
        if trailing_features >= layer.in_features:
            raise ValueError("trailing features cover the complete observation")
        with torch.no_grad():
            layer.weight[:, -trailing_features:].zero_()


def load_expanded_policy_state(
    policy,
    source_state: dict[str, torch.Tensor],
    trailing_features: int,
) -> None:
    """Load a policy while appending zeroed observation columns.

    This is used to warm-start the 109-input flat/AntMaze-compatible policy
    from a 105-input native Ant controller. Only input-weight matrices may
    grow; every other parameter must match exactly.
    """
    if trailing_features <= 0:
        raise ValueError("trailing_features must be positive")
    target_state = policy.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("source and target policy state keys differ")
    expanded = {}
    grown = []
    for key, target in target_state.items():
        source = source_state[key]
        if source.shape == target.shape:
            expanded[key] = source
            continue
        compatible = bool(
            source.ndim >= 2
            and source.shape[:-1] == target.shape[:-1]
            and target.shape[-1] == source.shape[-1] + trailing_features
        )
        if not compatible:
            raise ValueError(
                f"cannot expand {key}: {tuple(source.shape)} -> "
                f"{tuple(target.shape)}"
            )
        value = torch.zeros_like(target)
        value[..., :-trailing_features].copy_(source.to(value.device))
        expanded[key] = value
        grown.append(key)
    if not grown:
        raise ValueError("no policy input tensors required expansion")
    policy.load_state_dict(expanded, strict=True)


def expanded_observation_normalizer_state(
    state: dict,
    trailing_features: int,
) -> dict:
    """Append neutral statistics for flat zero-valued observation inputs."""
    if trailing_features <= 0:
        raise ValueError("trailing_features must be positive")
    mean = np.asarray(state["mean"], dtype=np.float64)
    var = np.asarray(state["var"], dtype=np.float64)
    if mean.shape != var.shape or mean.ndim != 1:
        raise ValueError("expected one-dimensional observation statistics")
    return {
        "mean": np.concatenate([mean, np.zeros(trailing_features)]),
        "var": np.concatenate([var, np.ones(trailing_features)]),
        "count": float(state["count"]),
    }


def reset_trailing_observation_normalizer_features(
    state: dict,
    trailing_features: int,
) -> dict:
    """Reset changed goal features without discarding proprioceptive stats."""
    if trailing_features <= 0:
        raise ValueError("trailing_features must be positive")
    mean = np.asarray(state["mean"], dtype=np.float64).copy()
    var = np.asarray(state["var"], dtype=np.float64).copy()
    if mean.shape != var.shape or mean.ndim != 1:
        raise ValueError("expected one-dimensional observation statistics")
    if trailing_features >= mean.size:
        raise ValueError("trailing features cover the complete observation")
    mean[-trailing_features:] = 0.0
    var[-trailing_features:] = 1.0
    return {"mean": mean, "var": var, "count": float(state["count"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual embodied transfer on Gymnasium Ant-v5"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load policy weights from a compatible checkpoint before update 1",
    )
    parser.add_argument("--vec-env", choices=["sync", "subproc"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    ppo_config = PPOConfig(**config.get("ppo", {}))
    ppo_config.seed = args.seed
    n_envs = ppo_config.n_envs
    env_config = resolved_env_config(config, seed=args.seed, n_envs=n_envs)
    env_class_path = config.get("env", {}).get("class_path", ENV_CLASS_PATH)
    vec_kind = args.vec_env or config.get("vec_env", "sync")
    resolved = {
        **config,
        "seed": args.seed,
        "resolved_env": env_config,
        "resolved_env_class": env_class_path,
        "resolved_vec_env": vec_kind,
        "compatibility": {
            "environment": env_config.get("env_id", "Ant-v5"),
            "reference_environment": "SlipperyAnt-v3",
            "protocol": (
                "strict official AntMaze-v5 goal task"
                if env_class_path.endswith("AntMazeTransferEnv")
                else "floor-friction intervention ported to native mujoco"
            ),
        },
    }
    initialization = None
    initial_data = None
    if args.initialize_from is not None:
        if not args.initialize_from.exists():
            raise FileNotFoundError(args.initialize_from)
        initialization = {
            "checkpoint": str(args.initialize_from.resolve()),
            "checkpoint_sha256": file_sha256(args.initialize_from),
            "optimizer_and_reward_normalizer_reset": True,
        }
        initialization_config = config.get("initialization") or {}
        initialization.update(
            {
                "expand_trailing_observation_features": int(
                    initialization_config.get("expand_trailing_observation_features", 0)
                ),
                "transfer_observation_normalizer": bool(
                    initialization_config.get("transfer_observation_normalizer", False)
                ),
                "reset_trailing_observation_normalizer_features": int(
                    initialization_config.get(
                        "reset_trailing_observation_normalizer_features", 0
                    )
                ),
            }
        )
        resolved["initialization"] = initialization
    source_config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    protocol_sha256 = canonical_hash(resolved)
    resolved["provenance"] = {
        "source_config": str(args.config.resolve()),
        "source_config_sha256": source_config_sha256,
        "protocol_sha256": protocol_sha256,
    }

    checkpoint = args.output / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise SystemExit(f"{checkpoint} exists; use --resume or choose another output")
    resume_from = checkpoint if args.resume and checkpoint.exists() else None
    args.output.mkdir(parents=True, exist_ok=True)
    existing_config_path = args.output / "config.yaml"
    if args.resume and existing_config_path.exists():
        with existing_config_path.open() as file:
            existing_config = yaml.safe_load(file)
        existing_protocol_hash = existing_config.get("provenance", {}).get(
            "protocol_sha256"
        )
        if (
            existing_protocol_hash is not None
            and existing_protocol_hash != protocol_sha256
        ):
            raise SystemExit(
                "Refusing to resume with a changed protocol: "
                f"{existing_protocol_hash} != {protocol_sha256}"
            )
    with (args.output / "config.yaml").open("w") as file:
        yaml.safe_dump(resolved, file, sort_keys=False)

    started_at = datetime.now(timezone.utc)
    run_metadata = {
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "output": str(args.output.resolve()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git_revision": git_revision(),
        "code_provenance": code_provenance(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "device": args.device,
        "vec_env": vec_kind,
        "n_envs": n_envs,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "source_config_sha256": source_config_sha256,
        "protocol_sha256": protocol_sha256,
    }
    write_json(args.output / "run_metadata.json", run_metadata)

    env_functions = [
        partial(
            make_env_from_spec,
            env_class_path,
            env_config,
            args.seed + worker,
        )
        for worker in range(n_envs)
    ]
    vec_env = make_vec_env(env_functions, kind=vec_kind, start_method="spawn")
    connectome = (
        build_connectome(config.get("connectome"))
        if config["policy"]["name"] in CONNECTOME_POLICY_NAMES
        else None
    )
    policy = build_policy_from_config(
        config,
        vec_env.observation_space.shape[0],
        vec_env.action_space.shape[0],
        connectome=connectome,
    )
    if initialization is not None and resume_from is None:
        initial_data = torch.load(
            args.initialize_from, map_location=args.device, weights_only=False
        )
        expansion = int(initialization["expand_trailing_observation_features"])
        if expansion:
            load_expanded_policy_state(policy, initial_data["policy_state"], expansion)
        else:
            policy.load_state_dict(initial_data["policy_state"], strict=True)
        reset_trailing = int(
            config["policy"].get(
                "reset_trailing_observation_features_on_initialization", 0
            )
        )
        if reset_trailing:
            zero_policy_input_features(policy, reset_trailing)
    if connectome is not None:
        graph_diagnostics = diagnose_connectome(connectome)
        write_json(args.output / "graph_diagnostics.json", graph_diagnostics)
        run_metadata["graph"] = {
            "node_count": graph_diagnostics["node_count"],
            "edge_count": graph_diagnostics["edge_count"],
            "storage": graph_diagnostics["storage"],
            "partitions": graph_diagnostics["partitions"],
            "weight_multiset_sha256": graph_diagnostics["weights"][
                "sorted_multiset_sha256"
            ],
            "spectral_radius_estimate": graph_diagnostics["spectral_radius_estimate"],
        }
    run_metadata["policy"] = {
        "name": config["policy"]["name"],
        "actor_class": policy.actor.__class__.__name__,
        "actor_trainable_parameters": sum(
            value.numel() for value in policy.actor.parameters() if value.requires_grad
        ),
        "total_trainable_parameters": sum(
            value.numel() for value in policy.parameters() if value.requires_grad
        ),
        "state_dim": policy.state_dim,
    }
    if initialization is not None:
        run_metadata["initialization"] = initialization
    write_json(args.output / "run_metadata.json", run_metadata)
    trainer = RecurrentPPO(
        vec_env, policy, ppo_config, device=args.device, protocol_hash=protocol_sha256
    )
    if initial_data is not None and initialization["transfer_observation_normalizer"]:
        if trainer.obs_rms is None:
            raise ValueError("cannot transfer an observation normalizer when disabled")
        source_normalizer = initial_data.get("obs_rms_state")
        if source_normalizer is None:
            raise ValueError("initial checkpoint lacks obs_rms_state")
        expansion = int(initialization["expand_trailing_observation_features"])
        if expansion:
            source_normalizer = expanded_observation_normalizer_state(
                source_normalizer, expansion
            )
        reset_normalizer = int(
            initialization["reset_trailing_observation_normalizer_features"]
        )
        if reset_normalizer:
            source_normalizer = reset_trailing_observation_normalizer_features(
                source_normalizer, reset_normalizer
            )
        trainer.obs_rms.load_state_dict(source_normalizer)

    try:
        result = trainer.learn(log_dir=args.output, resume_from=resume_from)
    except BaseException as exc:
        run_metadata.update(
            {
                "status": (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                ),
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        write_json(args.output / "run_metadata.json", run_metadata)
        raise
    finally:
        vec_env.close()

    torch.save(
        {
            "policy_state": policy.state_dict(),
            "config": resolved,
            "obs_dim": int(vec_env.observation_space.shape[0]),
            "action_dim": int(vec_env.action_space.shape[0]),
        },
        args.output / "policy.pt",
    )
    completed_returns = result.get("completed_returns", [])
    summary = {
        "seed": args.seed,
        "policy": config["policy"]["name"],
        "target_timesteps": ppo_config.total_timesteps,
        "completed_episodes": len(result.get("completed_returns", [])),
        "final_100_return": (
            float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
        ),
        "n_contexts_observed": len(trainer._terrain_returns),
        "completed_timesteps": int(
            result.get("history", [{}])[-1].get("timesteps", 0)
            if result.get("history")
            else 0
        ),
        "training_wall_time_seconds": float(result.get("elapsed_wall", 0.0)),
        "protocol_sha256": protocol_sha256,
    }
    with (args.output / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    run_metadata.update(
        {
            "status": "complete",
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds_this_invocation": time.time() - started_at.timestamp(),
            "completed_timesteps": summary["completed_timesteps"],
            "training_wall_time_seconds": summary["training_wall_time_seconds"],
        }
    )
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and torch.cuda.is_available():
        device_index = (
            requested_device.index
            if requested_device.index is not None
            else torch.cuda.current_device()
        )
        run_metadata["cuda_peak_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device_index)
        )
        run_metadata["cuda_peak_memory_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device_index)
        )
    write_json(args.output / "run_metadata.json", run_metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

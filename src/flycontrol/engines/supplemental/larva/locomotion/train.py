"""Single training entry point for all locomotion experiments.

Usage:
    python -m larva.locomotion.train \
        --config larva/locomotion/configs/connectome_cfc.yaml \
        --seed 0 \
        --output outputs/locomotion/connectome_cfc_seed0

Vec-env selection:
    The yaml may set a top-level ``vec_env: sync | subproc`` (default
    ``sync``). ``subproc`` spawns one OS process per env — required for
    real parallelism on multi-core CPU. macOS forces ``spawn`` start
    method because fork() is unsafe with MuJoCo.

Checkpointing:
    A ``checkpoint.pt`` is written atomically (tmp + os.replace) at the
    end of every update. It contains policy + optim + RNG + history.
    Use ``--resume`` to continue from it. Without ``--resume``, the
    presence of a checkpoint in --output is treated as a hard error to
    prevent silent overwrite of a long-running experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import signal
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from .connectome import (
    load_adult,
    load_banc,
    load_larva,
    normalize_connectome,
    randomize,
    transform_edge_polarity,
)
from .graph_diagnostics import diagnose_connectome
from .models import build_policy
from .ppo import PPOConfig, RecurrentPPO
from .structured_null import (
    banc_leg_vnc_induced_subgraph,
    load_node_metadata,
    load_target_stratum_metadata,
    target_stratified_degree_rewire,
)
from .vec_env import make_env_from_spec, make_vec_env


# Module-level so SubprocVecEnv (spawn) can pickle it. Do not capture
# any local state — it must be reconstructible from its arguments only.
ENV_CLASS_PATH = "larva.locomotion.env.TerrainLocomotionEnv"
CONNECTOME_POLICY_NAMES = frozenset(
    {"connectome_cfc", "connectome_rate", "role_correct_connectome_rate"}
)


def policy_uses_connectome(name: str) -> bool:
    return str(name) in CONNECTOME_POLICY_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_resume_checkpoint(path: Path, cfg: dict) -> dict:
    """Validate that a latest-update checkpoint is internally atomic/consistent.

    A process may die midway through the next rollout.  Resume is safe only from
    the last fully written update, so reject a checkpoint whose update,
    timesteps, or history disagree before constructing any expensive envs.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"policy_state", "optim_state", "update", "timesteps", "history"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"resume checkpoint lacks required keys: {missing}")
    ppo_cfg = cfg.get("ppo", {})
    steps_per_update = int(ppo_cfg.get("n_envs", 1)) * int(
        ppo_cfg.get("n_steps", 128)
    )
    total_timesteps = int(ppo_cfg.get("total_timesteps", 0))
    update = int(checkpoint["update"])
    timesteps = int(checkpoint["timesteps"])
    history = list(checkpoint["history"])
    if update < 0 or steps_per_update <= 0:
        raise ValueError("resume checkpoint has invalid update geometry")
    if timesteps != update * steps_per_update:
        raise ValueError(
            "resume checkpoint update/timesteps mismatch: "
            f"update={update}, timesteps={timesteps}, "
            f"steps_per_update={steps_per_update}"
        )
    if len(history) != update:
        raise ValueError(
            f"resume checkpoint history length {len(history)} != update {update}"
        )
    if history:
        last = history[-1]
        if int(last.get("update", -1)) != update or int(
            last.get("timesteps", -1)
        ) != timesteps:
            raise ValueError("resume checkpoint last history row is inconsistent")
    return {
        "update": update,
        "timesteps": timesteps,
        "history_rows": len(history),
        "steps_per_update": steps_per_update,
        "planned_updates": total_timesteps // steps_per_update,
        "next_update": update + 1,
        "orphan_tmp_exists": path.with_suffix(path.suffix + ".tmp").exists(),
    }


def load_frozen_critic_initialization(policy, checkpoint_path: Path) -> dict:
    """Load only a critic from a registered initial checkpoint.

    Capacity-control actors consume different numbers of random draws during
    construction.  Copying the same seed-matched critic prevents that
    architecture detail from silently changing the optimization tool.
    """
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_kind") != "initial":
        raise ValueError(
            f"critic source must be an initial checkpoint: {checkpoint_path}"
        )
    policy_state = payload.get("policy_state")
    if not isinstance(policy_state, dict):
        raise ValueError(f"critic source lacks policy_state: {checkpoint_path}")
    source = {
        key.removeprefix("critic."): value
        for key, value in policy_state.items()
        if key.startswith("critic.")
    }
    expected = policy.critic.state_dict()
    if set(source) != set(expected):
        raise ValueError(
            "critic state keys differ: "
            f"missing={sorted(set(expected) - set(source))}, "
            f"extra={sorted(set(source) - set(expected))}"
        )
    for key, value in source.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(
                f"critic tensor shape differs for {key}: "
                f"{tuple(value.shape)} != {tuple(expected[key].shape)}"
            )
    policy.critic.load_state_dict(source, strict=True)
    return {
        "mode": "copied_from_seed_matched_initial_checkpoint",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_train_seed": int(payload.get("cfg", {}).get("seed", -1)),
        "tensor_count": len(source),
    }


def zero_actor_output_initialization(policy) -> dict:
    """Start learned controllers at the same neutral mean action.

    Hidden features and input encoders retain their architecture-specific
    initialization. Only the final learned linear map is reset; its first
    update enables upstream learning. No bypass may defeat neutrality.
    """
    actor = policy.actor
    if getattr(actor, "direct_skip", None) is not None or getattr(actor, "goal_adapter", None) is not None:
        raise ValueError("zero output initialization does not support actor bypasses")
    if hasattr(actor, "core"):
        layer = getattr(actor.core, "output_layer", None)
    elif hasattr(actor, "net"):
        layer = actor.net[-1]
    elif hasattr(actor, "head"):
        layer = actor.head
    else:
        layer = None
    if isinstance(layer, torch.nn.Sequential):
        layer = layer[-1]
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("zero output initialization requires a learned final linear layer")
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()
    return {"mode": "zero_final_linear", "weight_shape": list(layer.weight.shape),
            "bias_shape": list(layer.bias.shape) if layer.bias is not None else None}


def _array_sha256(*arrays: np.ndarray | None) -> str | None:
    """Stable checksum for the exact graph tensors used by a run."""
    if any(array is None for array in arrays):
        return None
    digest = hashlib.sha256()
    for array in arrays:
        assert array is not None
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_code_snapshot() -> dict:
    """Hash the executable training path, including an uncommitted diff."""
    source_names = (
        "train.py", "ppo.py", "models.py", "connectome.py",
        "structured_null.py", "role_correct.py", "env.py", "cpg.py",
        "terrain.py", "terrain_course.py", "ood_metrics.py", "vec_env.py",
        "adaptive_rate.py", "terrain_metrics.py", "tasks.py", "cfc.py",
    )
    source_paths = [Path(__file__).with_name(name) for name in source_names]
    source_sha256 = {
        str(path.resolve()): _file_sha256(path)
        for path in source_paths if path.is_file()
    }
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "larva/locomotion"],
            check=True,
            capture_output=True,
        ).stdout
        git_diff_sha256 = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_commit = ""
        git_diff_sha256 = ""
    canonical = json.dumps(source_sha256, sort_keys=True, separators=(",", ":"))
    return {
        "git_commit": git_commit,
        "git_diff_sha256": git_diff_sha256,
        "source_sha256": source_sha256,
        "source_bundle_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _write_run_manifest(
    output: Path,
    *,
    config_path: Path,
    train_seed: int,
    requested_device: str,
    policy,
    connectome,
    training_randomness: dict,
) -> None:
    """Record the actual architecture and graph, not only requested YAML."""
    actor_trainable = sum(
        parameter.numel() for parameter in policy.actor.parameters()
        if parameter.requires_grad)
    critic_trainable = sum(
        parameter.numel() for parameter in policy.critic.parameters()
        if parameter.requires_grad)
    total_trainable = sum(
        parameter.numel() for parameter in policy.parameters()
        if parameter.requires_grad)
    graph = None
    if connectome is not None:
        if connectome.W is None:
            checksum = _array_sha256(
                connectome.edge_pre,
                connectome.edge_post,
                connectome.edge_weight,
            )
            edge_count = int(connectome.edge_weight.size)
            storage = "sparse_edges"
        else:
            checksum = _array_sha256(connectome.W)
            edge_count = int(np.count_nonzero(connectome.W))
            storage = "dense"
        graph = {
            "num_neurons": int(connectome.N),
            "num_edges": edge_count,
            "density": float(edge_count / (connectome.N * connectome.N)),
            "sensory_dim": int(connectome.sensory_dim),
            "internal_dim": int(connectome.internal_dim),
            "output_dim": int(connectome.output_dim),
            "storage": storage,
            "sha256": checksum,
            "metadata": connectome.metadata or {},
        }
    manifest = {
        "schema_version": 1,
        "config_path": str(config_path.resolve()),
        "train_seed": int(train_seed),
        "requested_device": requested_device,
        "policy_class": policy.actor.__class__.__name__,
        "core_class": type(policy.actor.core).__name__ if hasattr(policy.actor, "core") else None,
        "group_dynamics": getattr(policy.actor, "dynamics_manifest", None),
        "actor_trainable_parameters": int(actor_trainable),
        "critic_trainable_parameters": int(critic_trainable),
        "total_trainable_parameters": int(total_trainable),
        "training_randomness": training_randomness,
        "training_code": _training_code_snapshot(),
        "graph": graph,
    }
    routing_manifest = getattr(policy.actor, "routing_manifest", None)
    if routing_manifest is not None:
        manifest["role_correct_routing"] = routing_manifest
    with (output / "run_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, default=str)


def build_connectome(cfg: dict | None):
    """Return a possibly randomized ConnectomeData from the config block."""
    cfg = cfg or {}
    # Experiment-local, immutable, audited graph cache. All training/evaluation
    # processes verify the exact serialized graph before using its arrays.
    if cfg.get("frozen_graph_path"):
        import pickle
        path = Path(cfg["frozen_graph_path"])
        if _file_sha256(path) != cfg["frozen_graph_sha256"]:
            raise ValueError(f"frozen graph checksum mismatch: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)
    species = str(cfg.get("species", "larva")).lower()
    requested_normalization = str(cfg.get("normalization", "row_l1"))
    requested_gain = float(cfg.get("global_gain", 1.0))
    kind = str(cfg.get("kind", "original"))
    before_normalization = bool(cfg.get("randomize_before_normalization", False))
    if kind == "target_stratified_degree_rewire" and not before_normalization:
        raise ValueError(
            "target_stratified_degree_rewire requires "
            "randomize_before_normalization: true"
        )
    load_normalization = "none" if before_normalization else requested_normalization
    load_gain = 1.0 if before_normalization else requested_gain
    if species == "adult":
        data = load_adult(
            adult_dir=cfg["adult_dir"],
            sensory_selector=cfg["sensory_selector"],
            output_selector=cfg["output_selector"],
            min_syn_count=cfg.get("min_syn_count", 1),
            normalization=load_normalization,
            global_gain=load_gain,
            chunksize=cfg.get("chunksize", 250_000),
            polarity=cfg.get("polarity"),
        )
    elif species == "banc":
        data = load_banc(
            banc_dir=cfg["banc_dir"],
            sensory_selector=cfg["sensory_selector"],
            output_selector=cfg["output_selector"],
            min_syn_count=cfg.get("min_syn_count", 1),
            normalization=load_normalization,
            global_gain=load_gain,
            chunksize=cfg.get("chunksize", 250_000),
            polarity=cfg.get("polarity"),
        )
    elif species == "larva":
        data = load_larva(
            rescale=cfg.get("rescale", 4e-2),
            sensory_type=cfg.get("sensory_type", "all"),
        )
    else:
        raise ValueError(f"Unknown connectome species: {species!r}")

    subgraph = cfg.get("subgraph")
    if subgraph is not None:
        if species != "banc" or subgraph != "leg_vnc":
            raise ValueError(
                "the only supported configured subgraph is BANC 'leg_vnc'"
            )
        metadata_path = cfg.get("metadata_path")
        if metadata_path is None:
            raise ValueError("BANC leg_vnc subgraph requires metadata_path")
        data = banc_leg_vnc_induced_subgraph(
            data, load_node_metadata(metadata_path)
        )

    if kind == "original":
        graph = data
    elif kind == "target_stratified_degree_rewire":
        metadata_path = cfg.get("metadata_path")
        if metadata_path is None:
            raise ValueError("target-stratified rewire requires metadata_path")
        base_fields_raw = cfg.get("target_strata_base_fields")
        base_fields = (
            None
            if base_fields_raw is None
            else tuple(str(field) for field in base_fields_raw)
        )
        extra_fields = tuple(cfg.get("target_strata_fields", ()))
        if species == "banc" and base_fields is None:
            metadata = load_node_metadata(metadata_path)
            null_family = "BANC-ANATOMY-STRATIFIED-REWIRE"
        else:
            if base_fields is None:
                raise ValueError(
                    "non-BANC target-stratified rewire requires "
                    "target_strata_base_fields"
                )
            metadata = load_target_stratum_metadata(
                metadata_path, tuple(dict.fromkeys((*base_fields, *extra_fields)))
            )
            null_family = str(
                cfg.get("null_family", "ANATOMY-STRATIFIED-REWIRE")
            )
        graph = target_stratified_degree_rewire(
            data,
            metadata,
            seed=int(cfg.get("randomize_seed", 0)),
            n_rounds=int(cfg.get("rewire_rounds", 5)),
            extra_target_fields=extra_fields,
            base_target_fields=base_fields,
            null_family=null_family,
            preserve_incoming_weight_multiset=bool(cfg.get("preserve_incoming_weight_multiset", False)),
        )
    else:
        graph = randomize(data, kind, seed=cfg.get("randomize_seed", 0))
    polarity_transform = cfg.get("post_randomization_polarity_transform")
    if polarity_transform is not None:
        if not before_normalization:
            raise ValueError(
                "post_randomization_polarity_transform requires "
                "randomize_before_normalization: true"
            )
        graph = transform_edge_polarity(graph, str(polarity_transform))
    if before_normalization:
        graph = normalize_connectome(
            graph, requested_normalization, requested_gain
        )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a locomotion policy.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/locomotion/run"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Explicit development/smoke override; persisted in output config.yaml.",
    )
    parser.add_argument("--vec-env", type=str, default=None,
                        choices=["sync", "subproc"],
                        help="Override yaml vec_env setting.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output>/checkpoint.pt if present.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    from .terrain_objective import validate_terrain_objective_configuration
    validate_terrain_objective_configuration(cfg)
    if args.total_timesteps is not None:
        if args.total_timesteps <= 0:
            raise ValueError("--total-timesteps must be positive")
        cfg.setdefault("ppo", {})["total_timesteps"] = int(args.total_timesteps)
    resume_policy = str(cfg.get("resume_policy", "allow_rollout_boundary"))
    if resume_policy not in {"allow_rollout_boundary", "restart_only"}:
        raise ValueError(
            "resume_policy must be 'allow_rollout_boundary' or 'restart_only'"
        )
    if args.resume and resume_policy == "restart_only":
        raise SystemExit(
            "This protocol declares resume_policy: restart_only. Restart the "
            "interrupted seed in a new output directory; mid-episode simulator "
            "state is not bit-reproducible."
        )
    ppo_raw = dict(cfg.get("ppo", {}))
    initialization_seed = int(
        args.seed
        if ppo_raw.get("initialization_seed") is None
        else ppo_raw["initialization_seed"]
    )
    environment_seed = int(
        args.seed
        if ppo_raw.get("environment_seed") is None
        else ppo_raw["environment_seed"]
    )
    policy_sampling_seed = int(
        args.seed
        if ppo_raw.get("policy_sampling_seed") is None
        else ppo_raw["policy_sampling_seed"]
    )
    minibatch_seed = int(
        args.seed
        if ppo_raw.get("minibatch_seed") is None
        else ppo_raw["minibatch_seed"]
    )
    set_seed(initialization_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "seed": args.seed}, f)

    # ---- Checkpoint policy (must run BEFORE env construction so we
    #      don't burn 30s building 8 mujoco sims and then bail) -------
    ckpt_path = args.output / "checkpoint.pt"
    resume_from: Path | None = None
    if args.resume:
        if not ckpt_path.exists():
            print(f"[train] --resume given but {ckpt_path} not found; "
                  f"starting fresh.", file=sys.stderr, flush=True)
        else:
            resume_audit = validate_resume_checkpoint(ckpt_path, cfg)
            print(
                "[train] resume checkpoint audit: "
                + json.dumps(resume_audit, sort_keys=True),
                flush=True,
            )
            resume_from = ckpt_path
    else:
        if ckpt_path.exists():
            raise SystemExit(
                f"[train] {ckpt_path} exists. Pass --resume to continue "
                f"from it, or choose a different --output to start fresh. "
                f"Refusing to silently overwrite an in-progress experiment.")

    # ---- Vec-env construction ------------------------------------------
    vec_kind = (args.vec_env
                or cfg.get("vec_env", "sync")).lower()
    env_cfg = cfg.get("env", {})
    n_envs = cfg.get("ppo", {}).get("n_envs", 1)

    # functools.partial over a top-level function is picklable -- this
    # is the exact contract SubprocVecEnv requires.
    env_fns = [partial(make_env_from_spec, ENV_CLASS_PATH,
                       env_cfg, environment_seed + i)
               for i in range(n_envs)]
    vec_env = make_vec_env(env_fns, kind=vec_kind, start_method="spawn")

    obs_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.shape[0]

    pol_cfg = cfg["policy"]
    connectome = (
        build_connectome(cfg.get("connectome"))
        if policy_uses_connectome(pol_cfg["name"])
        else None
    )
    if connectome is not None:
        with open(args.output / "graph_diagnostics.json", "w") as f:
            json.dump(diagnose_connectome(connectome), f, indent=2)
    policy = build_policy(
        pol_cfg["name"], obs_dim, action_dim,
        actor_kwargs=pol_cfg.get("actor_kwargs"),
        value_hidden=pol_cfg.get("value_hidden"),
        log_std_init=pol_cfg.get("log_std_init", -0.5),
        connectome=connectome,
    )
    initialization_cfg = dict(cfg.get("initialization") or {})
    unknown_initialization = set(initialization_cfg).difference({
        "critic_state_from", "actor_output_zero"
    })
    if unknown_initialization:
        raise ValueError(
            f"unknown initialization keys: {sorted(unknown_initialization)}"
        )
    critic_initialization = None
    actor_initialization = None
    if initialization_cfg.get("actor_output_zero", False):
        actor_initialization = zero_actor_output_initialization(policy)
    if initialization_cfg.get("critic_state_from"):
        critic_initialization = load_frozen_critic_initialization(
            policy, Path(initialization_cfg["critic_state_from"])
        )
    _write_run_manifest(
        args.output,
        config_path=args.config,
        train_seed=args.seed,
        requested_device=args.device,
        policy=policy,
        connectome=connectome,
        training_randomness={
            "train_seed": int(args.seed),
            "separate_rng_streams": bool(
                ppo_raw.get("separate_rng_streams", False)
            ),
            "initialization_seed": initialization_seed,
            "environment_seed": environment_seed,
            "policy_sampling_seed": policy_sampling_seed,
            "minibatch_seed": minibatch_seed,
            "initial_hidden_mode": str(
                ppo_raw.get("initial_hidden_mode", "legacy_random")
            ),
            "critic_initialization": critic_initialization,
            "actor_initialization": actor_initialization,
        },
    )

    ppo_cfg = PPOConfig(**ppo_raw)
    ppo_cfg.seed = args.seed
    trainer = RecurrentPPO(vec_env, policy, ppo_cfg, device=args.device)

    # Explicit signal handlers so Ctrl-C / kill -INT actually terminate.
    # Why this is necessary: under multiprocessing+spawn, the parent's
    # pipe.recv() loops through PEP-475 EINTR retry on macOS, which can
    # swallow KeyboardInterrupt entirely. We bypass that by installing a
    # handler that calls vec_env.close() (broadcasts 'close' to workers
    # AND SIGTERMs them in parallel) then raises SystemExit. The exit
    # propagates through learn()'s normal control flow, so the
    # checkpoint of the last completed update is preserved on disk.
    def _on_signal(signum, _frame):
        sig_name = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(
            signum, f"signal-{signum}")
        print(f"\n[train] received {sig_name}; tearing down workers...",
              file=sys.stderr, flush=True)
        try:
            vec_env.close()
        except Exception as e:
            print(f"[train] vec_env.close() raised: {e!r}",
                  file=sys.stderr, flush=True)
        # 128 + signum is the conventional shell exit code for
        # "killed by signal N".
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        result = trainer.learn(log_dir=args.output, resume_from=resume_from)
    except SystemExit:
        raise
    finally:
        # Idempotent: close() bails fast if already closed.
        close = getattr(vec_env, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    torch.save({
        "policy_state": policy.state_dict(),
        "config": cfg,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "checkpoint_kind": "final_policy",
        "obs_rms_state": trainer.obs_rms.state_dict() if trainer.obs_rms is not None else None,
    }, args.output / "policy.pt")
    best_ckpt = args.output / "best_checkpoint.pt"
    if best_ckpt.exists():
        best = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        torch.save({
            "policy_state": best["policy_state"],
            "config": cfg,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "best_score": best.get("best_score"),
            "best_update": best.get("best_update"),
            "checkpoint_kind": "best_policy",
            "obs_rms_state": best.get("obs_rms_state"),
        }, args.output / "best_policy.pt")
    with open(args.output / "summary.json", "w") as f:
        json.dump({
            "n_episodes_completed": len(result["completed_returns"]),
            "mean_return": float(np.mean(result["completed_returns"]
                                         if result["completed_returns"]
                                         else [0.0])),
        }, f, indent=2)
    print(f"saved policy + history to {args.output}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    # Set start method once, here, before any MuJoCo is initialized.
    # 'spawn' is required on macOS for MuJoCo workers; safe everywhere
    # else too. Wrapped in try/except because the start method may
    # already be set (e.g. when running under pytest).
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()

"""Explicit Ant training dependencies: PPO, distillation, shared routers, mazes."""

from pathlib import Path
from types import SimpleNamespace
import copy
import shutil
import subprocess
import sys

from flycontrol.paths import read_json, write_json, sha256

NAMES = {"connectome": "adult"}
ENVIRONMENTS = {"umaze": "AntMaze_UMaze-v5", "medium": "AntMaze_Medium_Diverse_GR-v5", "large": "AntMaze_Large_Diverse_GR-v5"}


def stage_dir(root, method, seed, stage):
    return root / f"training/{method}_seed{seed}/{stage}"


def configuration(root, spec, name, config_group="maze"):
    import yaml
    source = root / "configs" / config_group / (name + ".yaml")
    config = yaml.safe_load(source.read_text())
    if spec["smoke"]:
        if "ppo" in config:
            config["ppo"].update(total_timesteps=128, n_envs=2, n_steps=32, seq_len=8, n_epochs=1, n_minibatches=2)
        if "imitation" in config:
            config["imitation"].update(demonstration_steps=128, dagger_steps=128, updates_per_stage=2,
                                        checkpoint_interval_updates=1, batch_size=4)
        if "dagger_updates_per_stage" in config.get("imitation", {}):
            config["imitation"]["dagger_updates_per_stage"] = 2
    dest = root / "configs/resolved_maze" / config_group / (name + ".yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    return dest, config


def train_stage(root, spec, name, run, seed, device, source=None, shared=None, collection_seed=None, config_group="maze"):
    cp = run / "best_checkpoint.pt"
    marker = run / "COMPLETE.json"
    if marker.exists():
        if read_json(marker)["checkpoint_sha256"] != sha256(cp):
            raise ValueError(f"Changed completed checkpoint: {cp}")
        return cp
    config_path, config = configuration(root, spec, name, config_group)
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "body_control.train_maze" if "imitation" in config else "body_control.train",
               "--config", str(config_path), "--seed", str(seed), "--output", str(run), "--device", device]
    if "imitation" in config:
        if source is None:
            raise ValueError(f"Missing trained source for {name}")
        command += ["--source", str(source), "--teacher", str(root / "inputs/teacher.zip"),
                    "--sampling-seed", str(seed), "--collection-seed", str(seed * 100000 if collection_seed is None else collection_seed),
                    "--evaluation-seed", "9410000", "--evaluation-episodes", "1", "--resume-if-available"]
        if shared is not None and not (run / "teacher_labels_stage0.npz").exists():
            shutil.copy2(shared, run / "teacher_labels_stage0.npz")
            write_json(run / "collection_summary.json", dict(stages=[read_json(shared.with_suffix(".json"))]))
    else:
        command += ["--vec-env", "subproc"]
        if source is not None:
            command += ["--initialize-from", str(source)]
        if (run / "checkpoint.pt").exists():
            command += ["--resume"]
    print(f"Ant stage: {run.relative_to(root)}", flush=True)
    with (run / "train.log").open("a") as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not cp.exists():
        if "imitation" in config:
            raise FileNotFoundError(cp)
        # A short smoke rollout may finish before any episode is available to
        # the original best-rollout selector. Never discard that seed.
        shutil.copy2(run / "checkpoint.pt", cp)
        write_json(run / "checkpoint_selection.json", dict(rule="final checkpoint: no completed candidate episode", source_sha256=sha256(run / "checkpoint.pt")))
    write_json(marker, dict(checkpoint_sha256=sha256(cp), config_sha256=sha256(config_path),
        source_sha256=sha256(source) if source else None, pretrained_paper_weights=False))
    return cp


def prerequisites(root, spec, device):
    nodes = read_json(root / "configs/maze/prerequisites.json")
    for node in nodes:
        run = root / "training/prerequisites" / node["id"]
        source = root / "training/prerequisites" / node["source"] / "best_checkpoint.pt" if node["source"] else None
        train_stage(root, spec, node["id"], run, node["seed"], device, source=source, collection_seed=node["collection_seed"])


def collect(root, spec, task, label, device):
    import torch
    from stable_baselines3 import SAC
    from body_control.train_maze import collect_teacher_labels
    path = root / f"training/shared_inputs/{task}_{label}.npz"
    meta = path.with_suffix(".json")
    if meta.exists():
        if read_json(meta)["sha256"] != sha256(path):
            raise ValueError("Shared teacher histories changed")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    teacher = SAC.load(root / "inputs/teacher.zip", device=device)
    teacher.policy.eval()
    steps = (120000 if label == "waypoint" else 60000) if not spec["smoke"] else 128
    summary = collect_teacher_labels(path=path, target_steps=steps, seed=9420000 if task == "large" else 0,
        teacher=teacher, policy=SimpleNamespace(state_dim=0), normalizer={}, clip=60. if task == "large" else 30.,
        device=torch.device(device), student_rollout=False, goal_feature_mode="world", env_id=ENVIRONMENTS[task],
        student_goal_mode="final", label_space=label)
    write_json(meta, dict(summary, sha256=sha256(path)))
    return path


def insert_router(source, router, output, config=None, seed=0):
    import numpy as np
    import torch
    from body_control.train import build_policy_from_config
    data = torch.load(source, map_location="cpu", weights_only=False)
    common = torch.load(router, map_location="cpu", weights_only=False)
    if config is not None:
        torch.manual_seed(seed)
        policy = build_policy_from_config(config, 109, 8)
        state = policy.state_dict()
        for key, value in data["policy_state"].items():
            target = "actor.base." + key[len("actor."):] if key.startswith("actor.") else key
            if state[target].shape != value.shape:
                raise ValueError(f"Source shape mismatch: {key}")
            state[target] = value
        data["policy_state"] = state
    keys = [key for key in data["policy_state"] if key.startswith("actor.goal_router.")]
    if len(keys) != 6:
        raise ValueError("Expected the registered three-layer goal router")
    for key in keys:
        if data["policy_state"][key].shape != common["policy_state"][key].shape:
            raise ValueError(f"Router shape mismatch: {key}")
        data["policy_state"][key] = common["policy_state"][key].clone()
    data["obs_rms_state"] = copy.deepcopy(data["obs_rms_state"])
    for key in ("mean", "var"):
        data["obs_rms_state"][key] = np.asarray(data["obs_rms_state"][key]).copy()
        data["obs_rms_state"][key][-4:] = np.asarray(common["obs_rms_state"][key])[-4:]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    write_json(output.with_suffix(".json"), dict(source_sha256=sha256(source), router_sha256=sha256(router),
        normalization_columns_replaced="last four only", prepared_sha256=sha256(output)))
    return output


def to_umaze(root, spec, method, seed, device):
    name = NAMES.get(method, method)
    source = None
    for stage in ("base", "flat", "umaze"):
        source = train_stage(root, spec, f"{name}_{stage}", stage_dir(root, method, seed, stage), seed, device, source=source)
    return source


def to_medium(root, spec, method, seed, device):
    name = NAMES.get(method, method)
    to_umaze(root, spec, method, seed, device)
    if method in {"connectome", "mlp"}:
        source = root / f"training/prerequisites/{name}_prerequisite_4/best_checkpoint.pt"
    else:
        umaze = to_umaze(root, spec, method, 0, device)
        router = root / "training/prerequisites/mlp_prerequisite_4/best_checkpoint.pt"
        _, conf = configuration(root, spec, f"{name}_medium")
        source = root / f"training/shared_sources/{name}_medium.pt"
        if not source.exists():
            insert_router(umaze, router, source, conf)
    shared = collect(root, spec, "medium", "action", device)
    return train_stage(root, spec, f"{name}_medium", stage_dir(root, method, seed, "medium"), seed, device,
                       source=source, shared=shared, collection_seed=0)


def train(root, spec, method, seed, device):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    prerequisites(root, spec, device)
    medium = to_medium(root, spec, method, seed, device)
    router_source = to_medium(root, spec, "mlp", 0, device)
    waypoint_data = collect(root, spec, "large", "waypoint", device)
    router = train_stage(root, spec, "shared_router_large", root / "training/large_router", 20261022, device,
                          source=router_source, shared=waypoint_data, collection_seed=9420000)
    shared = collect(root, spec, "large", "action", device)
    source = root / f"training/shared_sources/{method}_seed{seed}_large.pt"
    if not source.exists():
        insert_router(medium, router, source)
    train_stage(root, spec, f"{NAMES.get(method, method)}_large", stage_dir(root, method, seed, "large"), seed, device,
                source=source, shared=shared, collection_seed=9420000)


def evaluate(root, spec, method, seed, device="cpu"):
    from body_control.evaluate import evaluate_run
    from body_control.evaluate_maze import evaluate_antmaze_run, antmaze_physical_failure
    protocol = read_json(root / "protocol.json")
    episodes = 1 if spec["smoke"] else protocol["episodes"]
    for task in protocol["tasks"]:
        run = stage_dir(root, method, seed, task)
        path = root / f"evaluation/{method}_seed{seed}/{task}.json"
        cp = run / "best_checkpoint.pt"
        if path.exists():
            if read_json(path.with_suffix(".provenance.json"))["checkpoint_sha256"] != sha256(cp):
                raise ValueError("Ant evaluation checkpoint changed")
            continue
        if task == "flat":
            records = evaluate_run(run, protocol["friction_values"], episodes=episodes,
                seed=protocol["evaluation_seeds"][task], device=device, checkpoint="best", action_scale=.9)
        else:
            records = evaluate_antmaze_run(run, ENVIRONMENTS[task], episodes=episodes,
                seed=protocol["evaluation_seeds"][task], device=device, checkpoint="best",
                waypoint_planner="official_grid_waypoints" if task == "umaze" else "none", continuing_task=False,
                action_smoothing=.8 if task == "umaze" else .95,
                action_scale=.8 if method == "connectome" and task != "umaze" else 1.)
            for row in records:
                row["physical_failure"] = bool(antmaze_physical_failure(row))
        write_json(path, records)
        write_json(path.with_suffix(".provenance.json"), dict(checkpoint=str(cp.relative_to(root)),
            checkpoint_sha256=sha256(cp), evaluation_sha256=sha256(path)))
    if method in {"connectome", "mlp"}:
        from body_control.motion_quality import evaluate_motion_quality
        for role in ("flat_motion", "friction", "medium_original"):
            task = "medium" if role == "medium_original" else "flat"
            run = stage_dir(root, method, seed, task)
            path = root / f"evaluation/{method}_seed{seed}/{role}.json"
            cp = run / "best_checkpoint.pt"
            if path.exists():
                if read_json(path.with_suffix(".provenance.json"))["checkpoint_sha256"] != sha256(cp):
                    raise ValueError("Ant diagnostic checkpoint changed")
                continue
            if role == "flat_motion":
                records = evaluate_motion_quality(run, friction=1., episodes=episodes,
                    seed=protocol["evaluation_seeds"]["flat"], device=device, checkpoint="best", action_scale=.9)
            elif role == "friction":
                records = evaluate_run(run, protocol["robustness_frictions"], episodes=1 if spec["smoke"] else protocol["friction_episodes"],
                    seed=protocol["friction_seed"], device=device, checkpoint="best", action_scale=.9)
            else:
                records = evaluate_antmaze_run(run, ENVIRONMENTS["medium"], episodes=episodes,
                    seed=9310000, device=device, checkpoint="best", waypoint_planner="none", continuing_task=False,
                    action_smoothing=.95, action_scale=1.)
                for row in records:
                    row["physical_failure"] = bool(antmaze_physical_failure(row))
            write_json(path, records)
            write_json(path.with_suffix(".provenance.json"), dict(checkpoint=str(cp.relative_to(root)),
                checkpoint_sha256=sha256(cp), evaluation_sha256=sha256(path)))

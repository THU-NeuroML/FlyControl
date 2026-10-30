"""One run, one fixed configuration, and a continuous chain of generated outputs."""

from copy import deepcopy
from pathlib import Path
import os
import shutil
import time
import importlib.metadata
import platform
import sys

from flycontrol.paths import CONFIGS, ENGINES, read_json, write_json, sha256, source_digest, configuration_manifest
from flycontrol.runtime import worker

STAGES = ("prepare", "train", "evaluate", "probe", "analyze", "plot")
BODIES = ("cassie", "tripod", "go1", "hexapod")


def initialize(root, config, smoke=False, methods=None, seeds=None):
    root = Path(root).resolve()
    path = Path(config)
    if not path.exists():
        path = CONFIGS / (config.removesuffix(".json") + ".json")
    spec = read_json(path)
    spec["smoke"] = smoke
    spec["methods"] = methods or spec["methods"]
    if not set(spec["methods"]) <= set(read_json(path)["methods"]):
        raise ValueError("Method is not part of this experiment")
    spec["seeds"] = seeds or (spec["seeds"][:1] if smoke else spec["seeds"])
    if len(set(spec["seeds"])) != len(spec["seeds"]) or len(set(spec["methods"])) != len(spec["methods"]):
        raise ValueError("Methods and training seeds must be unique")
    if not set(spec["seeds"]) <= set(read_json(path)["seeds"]):
        raise ValueError("Seed is not part of this experiment")
    spec["source_sha256"] = source_digest()
    destination = root / "run.json"
    if destination.exists():
        if read_json(destination) != spec:
            raise ValueError("Run settings or source changed; choose a fresh --run-dir")
        load_run(root)
        return spec
    if (root / "configs").exists():
        raise ValueError("Incomplete initialization: choose a fresh --run-dir")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CONFIGS / "internal", root / "configs")
    # Keep numerical engines isolated while all generated paths resolve
    # inside this new run.
    (root / "engines").symlink_to(ENGINES, target_is_directory=True)
    protocol_path = root / "configs" / spec["protocol"]
    protocol = read_json(protocol_path)
    if smoke and spec["family"] not in ("ant", "supplemental"):
        protocol["ppo"].update(total_timesteps=128, n_envs=2, n_steps=32,
                               seq_len=8, n_epochs=1, n_minibatches=2)
        protocol["evaluation_budgets"] = [0, 128]
        if spec["family"] == "body":
            protocol["conditions"] = ["flat"]
            protocol["evaluation_seeds"] = protocol["evaluation_seeds"][:1]
        if spec["family"] == "turning":
            protocol["final_phase_seeds"] = protocol["final_phase_seeds"][:1]
        if spec["family"] == "speed":
            evaluation = root / "configs/figure4/evaluation.json"
            ev = read_json(evaluation)
            ev["cases"] = read_json(root / "configs/figure4/analysis.json")["mechanism_cases"]
            write_json(evaluation, ev)
            for method in spec["methods"]:
                for seed in spec["seeds"]:
                    p = root / f"configs/figure4/experiments/{method}_seed{seed}.json"
                    conf = read_json(p)
                    for stage, steps in (("base", 64), ("continuation", 128)):
                        conf[stage + "_ppo"].update(total_timesteps=steps, n_envs=2, n_steps=32,
                            seq_len=8, n_epochs=1, n_minibatches=2)
                    write_json(p, conf)
    if spec["family"] == "turning":
        registered = root / "configs/figure3/registered_protocol.json"
        control = read_json(registered)
        control["ppo"] = protocol["ppo"]
        control["evaluation_budgets"] = protocol["evaluation_budgets"]
        if smoke: control["final_phase_seeds"] = control["final_phase_seeds"][:1]
        write_json(registered, control)
    write_json(protocol_path, protocol)
    write_json(root / "protocol.json", protocol)
    if spec["family"] == "body":
        source = root / "training/source"
        for part in ("common", spec["experiment"]):
            for p in (ENGINES / part).rglob("*.py"):
                dest = source / p.relative_to(ENGINES / part)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        write_json(root / "training/source_manifest.json", {str(p.relative_to(source)): sha256(p) for p in source.rglob("*.py")})
        write_json(root / "training/protocol.json", protocol)
        write_json(root / "training/control.json", dict(stop_all=False, yield_runs=[]))
    write_json(root / "configuration_manifest.json", configuration_manifest(root))
    versions = {}
    for name in ("numpy", "pandas", "scipy", "torch", "mujoco", "gymnasium", "gymnasium-robotics", "dm-control", "stable-baselines3"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(root / "environment.json", dict(python=sys.version, platform=platform.platform(),
        packages=versions, source_sha256=spec["source_sha256"]))
    write_json(destination, spec)
    return spec


def load_run(root):
    root = Path(root).resolve()
    if not (root / "run.json").exists():
        raise FileNotFoundError("Initialize with flycontrol prepare --config NAME --run-dir DIR")
    spec = read_json(root / "run.json")
    if spec["source_sha256"] != source_digest():
        raise ValueError("Code has changed since this run began; use a fresh --run-dir")
    if read_json(root / "configuration_manifest.json") != configuration_manifest(root):
        raise ValueError("Effective configurations changed after initialization; use a fresh --run-dir")
    return spec


def checkpoint(root, spec, method, seed):
    protocol = read_json(root / "protocol.json")
    family = spec["family"]
    if family == "body":
        name = f"{spec['experiment']}_{method}_{protocol['readout']}_seed{seed}"
        return root / f"training/runs/{name}/policy_{protocol['ppo']['total_timesteps']:07d}.pt"
    if family == "terrain":
        return root / f"training/runs/{method}_seed{seed}/policy_{protocol['ppo']['total_timesteps']:08d}.pt"
    if family == "turning":
        return root / f"training/{method}_seed{seed}/policy_{protocol['ppo']['total_timesteps']:08d}.pt"
    if family == "speed":
        cfg = read_json(root / f"configs/figure4/experiments/{method}_seed{seed}.json")
        return root / f"training/{method}_seed{seed}/continuation/policy_{cfg['continuation_ppo']['total_timesteps']:08d}.pt"
    if family == "ant":
        return root / f"training/{method}_seed{seed}/large/best_checkpoint.pt"
    raise ValueError(family)


def prepare(root, spec):
    if spec["family"]=="development_terrain":
        from flycontrol.experiments.supp_s3.terrain import prepare
        return prepare(root,spec)
    if spec["family"] == "supplemental":
        from flycontrol.experiments.shared.workflow import prepare as run
        return run(root, spec)
    from flycontrol.inputs import prepare as prepare_inputs
    prepare_inputs(root, spec)
    from flycontrol.mechanisms import prepare as prepare_mechanisms
    prepare_mechanisms(root, spec)
    if spec["family"] == "body":
        dest = root / "training/inputs"
        if not dest.exists():
            dest.symlink_to("../inputs", target_is_directory=True)


def train(root, spec, device):
    if spec["family"]=="development_terrain":
        from flycontrol.experiments.supp_s3.terrain import train
        return train(root,spec,device)
    if spec["family"] == "supplemental":
        from flycontrol.experiments.shared.workflow import train as run
        return run(root, spec, device)
    if spec["family"] == "historical":
        from flycontrol.experiments.supp_s1.workflow import train
        return train(root,spec,device)
    records_path = root / "training/models.json"
    records = read_json(records_path) if records_path.exists() else []
    for method in spec["methods"]:
        for seed in spec["seeds"]:
            path = checkpoint(root, spec, method, seed)
            previous = next((x for x in records if (x["method"], x["seed"]) == (method, seed)), None)
            if previous:
                if sha256(path) != previous["sha256"]:
                    raise ValueError(f"Completed checkpoint changed: {path}")
                continue
            family = spec["experiment"] if spec["family"] == "body" else spec["family"]
            worker(root, family, "train", method, seed, device, device=device)
            if not path.exists():
                raise FileNotFoundError(f"Training did not produce the required checkpoint: {path}")
            record = dict(method=method, seed=seed, checkpoint=str(path.relative_to(root)), sha256=sha256(path))
            records.append(record)
            write_json(records_path, records)

    if spec.get("registered_push"):
        from flycontrol.experiments.fig3.workflow import registered_train
        registered_train(root, spec, device)


def evaluate(root, spec, workers):
    if spec["family"]=="development_terrain":
        from flycontrol.experiments.supp_s3.terrain import worker as development_worker
        return development_worker(root,"evaluate")
    if spec["family"] == "supplemental":
        from flycontrol.experiments.shared.workflow import evaluate as run
        return run(root, spec, workers)
    if spec["family"] == "historical":
        from flycontrol.experiments.supp_s1.workflow import worker as historical_worker
        return historical_worker(root,"evaluate")
    records = read_json(root / "training/models.json")
    expected = {(m, s) for m in spec["methods"] for s in spec["seeds"]}
    if {(r["method"], r["seed"]) for r in records} != expected:
        raise ValueError("Training cohort is incomplete")
    if spec["family"] == "terrain":
        for row in records:
            if sha256(root / row["checkpoint"]) != row["sha256"]:
                raise ValueError("Training checkpoint changed")
        worker(root, "terrain", "evaluate", "all", 0, workers)
        return
    for row in records:
        path = root / row["checkpoint"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Checkpoint changed before evaluation: {path}")
        family = spec["experiment"] if spec["family"] == "body" else spec["family"]
        worker(root, family, "evaluate", row["method"], row["seed"], workers)

    if spec.get("registered_push"):
        from flycontrol.experiments.fig3.workflow import registered_evaluate
        registered_evaluate(root, spec, workers)


def execute(root, stage, *, device="cpu", workers=1):
    root = Path(root).resolve()
    spec = load_run(root)
    index = STAGES.index(stage)
    if index and not (root / f"status/{STAGES[index-1]}.json").exists():
        raise FileNotFoundError(f"First run: flycontrol {STAGES[index-1]} --run-dir {root}")
    started = time.time()
    if stage == "prepare":
        prepare(root, spec)
    elif stage == "train":
        train(root, spec, device)
    elif stage == "evaluate":
        evaluate(root, spec, workers)
    elif stage == "probe":
        from flycontrol.mechanisms import probe
        probe(root, spec, device)
    elif stage == "analyze":
        from flycontrol.analysis import analyze
        analyze(root, spec)
    elif stage == "plot":
        from flycontrol.plotting import plot
        plot(root, spec)
    write_json(root / f"status/{stage}.json", dict(completed=time.time(), elapsed_s=time.time()-started,
        source_sha256=spec["source_sha256"], smoke=spec["smoke"]))

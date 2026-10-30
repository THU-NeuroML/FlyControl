"""Paired terrain evaluation from fresh checkpoints, with native state replay."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np

from flycontrol.paths import run_root, read_json, write_json, sha256
from flycontrol.pipeline import checkpoint


def evaluate_group(job):
    from flycontrol.experiments.fig2.current import evaluate as numerical
    from flycontrol.experiments.fig2.current.steady_task import SteadyObstacleTask
    root = run_root()
    spec = read_json(root / "run.json")
    height, phase = job
    env = SteadyObstacleTask(fixed_cell=("blocks", height), capture_micro_trace=True)
    try:
        env.reset(seed=phase)
        snapshot = numerical.Snapshot(env)
        checked = False
        for method in spec["methods"]:
            for seed in spec["seeds"]:
                output = root / f"evaluation/{method}_seed{seed}/h{height:g}_phase{phase}"
                dest = output / "result.json"
                if dest.exists():
                    saved = read_json(dest)
                    if saved["checkpoint_sha256"] != sha256(checkpoint(root, spec, method, seed)) or saved["trace_sha256"] != sha256(output / "trace.npz"):
                        raise ValueError("Existing terrain episode no longer matches this run")
                    continue
                snapshot.restore(env)
                result, trace = numerical.rollout(env, method, seed)
                if not checked:
                    snapshot.restore(env)
                    replay, repeated = numerical.rollout(env, method, seed)
                    if result != replay or not all(np.array_equal(trace[k], repeated[k]) for k in trace):
                        raise ValueError("Native physics snapshot failed exact replay")
                    checked = True
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / "trace.npz", **trace)
                result["trace_sha256"] = sha256(output / "trace.npz")
                result["snapshot_replay_verified"] = True
                write_json(dest, result)
    finally:
        env.close()


def evaluate(root, spec, method, seed, cp, workers=1):
    study = read_json(root / "configs/figure2/study.json")
    phases = [int(x) for group in study["phase_bins"] for x in group]
    heights = [0.0, 0.2, 0.3, 0.45, 0.525]
    if spec["smoke"]:
        phases, heights = phases[:1], heights[:1]
    jobs = [(height, phase) for height in heights for phase in phases]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        list(pool.map(evaluate_group, jobs))
    write_json(root / "evaluation/grid.json", dict(heights=heights, phases=phases,
        methods=spec["methods"], seeds=spec["seeds"]))

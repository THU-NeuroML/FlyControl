"""Fixed phase-grid evaluation with all failures and full state traces retained."""

from .context import ROOT, CONFIG, GRAPH, write_json, configure_process, sha
import os

# All scientific evaluation uses CPU for both methods. Prevent each worker's
# fork_rng from creating contexts on every shared GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json, time, multiprocessing as mp, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from .task import ObstacleTask, CALIBRATION_CELLS
from .policy import build_policy
from fly_control.ppo import ObservationRunningMeanStd


def worker(job):
    name, checkpoint, terrain, value, seed, tag = job
    out = ROOT / "evaluation" / tag / name / f"{terrain}_{value:g}_seed{seed}"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    policy = None
    normalizer = None
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        init_path = Path(checkpoint).parent / "initialization.json"
        if not init_path.exists():
            init_path = (
                CONFIG / "experiments" / (Path(checkpoint).parent.name + ".json")
            )
        init = json.loads(init_path.read_text())
        policy = build_policy(
            GRAPH,
            688,
            6,
            init["method"],
            init["seed"],
            "cpu",
            "normalized",
            init.get("initial_log_std", -1.0),
        )
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()
        if ckpt.get("obs_rms_state"):
            normalizer = ObservationRunningMeanStd((688,))
            normalizer.load_state_dict(ckpt["obs_rms_state"])
    env = ObstacleTask(fixed_cell=(terrain, value), capture_micro_trace=True)
    started = time.time()
    try:
        obs, _ = env.reset(seed=seed)
        sim = env._sim
        initial = hashlib.sha256(sim.physics.data.qpos.tobytes()).hexdigest()
        geom = hashlib.sha256(
            sim.physics.model.geom_pos.tobytes()
            + sim.physics.model.geom_size.tobytes()
            + sim.physics.model.geom_type.tobytes()
        ).hexdigest()
        trace = {
            k: []
            for k in (
                "time_s",
                "qpos",
                "qvel",
                "position",
                "feet",
                "up",
                "action",
                "observation",
            )
        }

        def capture(action, observation):
            raw = sim.fly.get_observation(sim)
            fields = dict(
                time_s=float(sim.curr_time),
                qpos=sim.physics.data.qpos,
                qvel=sim.physics.data.qvel,
                position=raw["fly"][0],
                feet=raw["end_effectors"],
                up=raw["cardinal_vectors"][2],
                action=action,
                observation=observation,
            )
            for k, v in fields.items():
                trace[k].append(np.asarray(v).copy())

        capture(np.zeros(6), obs)
        state = policy.init_state(1, "cpu") if policy else None
        while True:
            action = np.zeros(6, dtype=np.float32)
            if policy:
                x = (
                    normalizer.normalize(obs[None], clip=10.0)
                    if normalizer
                    else obs[None]
                )
                with torch.no_grad():
                    a, _, _, state = policy.act(
                        torch.as_tensor(x), state, deterministic=True
                    )
                action = a[0].numpy()
            obs, r, term, trunc, info = env.step(action)
            for frame in env.last_micro_trace:
                for key, frame_value in frame.items():
                    trace[key].append(frame_value)
            if term or trunc:
                break
        arr = {k: np.asarray(v) for k, v in trace.items()}
        np.savez_compressed(out / "trace.npz", **arr)
        result = dict(
            name=name,
            terrain=terrain,
            value_mm=value,
            phase_seed=seed,
            success=bool(info["success"]),
            duration_s=env.elapsed,
            return_raw=env.episode_return,
            forward_mm=float(arr["position"][-1, 0] - arr["position"][0, 0]),
            max_lateral_mm=env.max_lateral,
            numerical_error=bool(info["numerical_error"]),
            physical_failure=bool(info["physical_termination"]),
            deadline=bool(info["task_deadline_reached"]),
            action_rms=float(np.sqrt(np.mean(arr["action"][1:] ** 2))),
            initial_qpos_sha256=initial,
            geometry_sha256=geom,
            cleared_stable_ever=bool(np.any(arr["observation"][:, -1] >= 1.0 - 1e-6)),
            chest_crossed_ever=bool(np.any(arr["position"][:, 0] >= env.far)),
            corridor_violated=bool(env.max_lateral > 1.5),
            crossed_within_corridor=bool(info["success"] and env.max_lateral <= 1.5),
            wall_s=time.time() - started,
            trace_sha256=sha(out / "trace.npz"),
            checkpoint_sha256=sha(checkpoint) if checkpoint else None,
            source=str(out / "result.json"),
        )
        write_json(out / "result.json", result)
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cpg")
    p.add_argument("--checkpoint")
    p.add_argument("--first", type=int, default=530000)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--tag", default="calibration")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--calibration", action="store_true")
    a = p.parse_args()
    cells = (
        CALIBRATION_CELLS
        if a.calibration or a.smoke
        else json.loads((CONFIG / "protocol.json").read_text()).get(
            "evaluation_cells",
            json.loads((CONFIG / "protocol.json").read_text())["cells"],
        )
    )
    if a.smoke:
        cells = [("gapped", 0.8), ("blocks", 0.45)]
    jobs = [
        (a.name, a.checkpoint, t, v, s, a.tag)
        for t, v in cells
        for s in range(a.first, a.first + a.count)
    ]
    rows = []
    write_json(
        ROOT / "qa" / f"{a.tag}_{a.name}_plan.json",
        dict(jobs=jobs, started=time.time()),
    )
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=mp.get_context("spawn"),
        initializer=configure_process,
    ) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            write_json(ROOT / "analysis" / f"{a.tag}_{a.name}_episodes.json", rows)
            print(f"{len(rows)}/{len(jobs)} episodes complete", flush=True)
    import pandas as pd

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "analysis" / f"{a.tag}_{a.name}_episodes.csv", index=False)
    summary = table.groupby(["terrain", "value_mm"])[
        ["success", "physical_failure", "forward_mm", "duration_s", "numerical_error"]
    ].mean()
    summary.to_csv(ROOT / "analysis" / f"{a.tag}_{a.name}_summary.csv")
    print(summary.to_string(), flush=True)
    write_json(
        ROOT / "qa" / f"{a.tag}_{a.name}_COMPLETE.json",
        dict(episodes=len(rows), completed=time.time()),
    )


if __name__ == "__main__":
    main()

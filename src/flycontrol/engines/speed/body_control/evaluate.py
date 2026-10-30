"""Deterministic evaluation with the original microstep trace and metric definitions."""

from .context import CONFIG, configure_process, sha, write_json
from .policy import build_policy
from .task import ContinuousSpeedTask
from .metrics import summarize
from pathlib import Path
import json
import time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

STATE = {}
FIELDS = [
    "elapsed_s",
    "interval_start_s",
    "interval_duration_s",
    "target_applied_mm_s",
    "body_forward_velocity_mm_s",
    "filtered_body_forward_velocity_mm_s",
    "body_lateral_velocity_mm_s",
    "heading_error_rad",
    "heading_rate_rad_s",
    "reward_tracking",
    "reward_speed",
    "reward_yaw",
    "reward_heading",
    "reward_failure",
    "reward_shaping",
    "gait_averaged_yaw_rad_s",
    "gait_coordinate",
    "duty_factor_command",
    "cpg_frequency_mean_hz",
]


def initialize(args):
    configure_process()
    STATE.update(args)
    path = Path(args["checkpoint"])
    policy = build_policy(args["method"], args["seed"], "cpu", 40)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["timesteps"] == args["steps"]
    policy.load_state_dict(payload["policy_state"])
    assert payload["obs_rms_state"] is None
    policy.eval()
    STATE.update(
        policy=policy,
        env=ContinuousSpeedTask(40, fixed_case={}),
        checkpoint_sha256=sha(path),
        plan_sha256=sha(CONFIG / "evaluation.json"),
    )


def episode(case):
    p = (
        Path(STATE["folder"])
        / f"{case['kind']}_c{case['command_seed']}_p{case['physics_seed']}.json"
    )
    if p.exists():
        old = json.loads(p.read_text())
        assert old["checkpoint_sha256"] == STATE["checkpoint_sha256"]
        assert old["plan_sha256"] == STATE["plan_sha256"]
        assert old["trace_sha256"] == sha(p.with_suffix(".npz"))
        return old
    env, policy = STATE["env"], STATE["policy"]
    env.fixed_case = case
    obs, initial = env.reset()
    hidden = policy.init_state(1, "cpu")
    rows = []
    started = time.time()
    while True:
        with torch.no_grad():
            action, _, _, hidden = policy.act(
                torch.from_numpy(obs).unsqueeze(0), hidden, deterministic=True
            )
        obs, reward, terminated, truncated, info = env.step(action[0].numpy())
        for m in env.last_micro_infos:
            row = {k: float(m.get(k, 0.0)) for k in FIELDS}
            bad = [k for k, v in row.items() if not np.isfinite(v)]
            if bad:
                assert m["physical_failure"]
                for k in bad:
                    row[k] = 0.0
            row.update(
                drive=m["physical_action"][0],
                turn=m["physical_action"][1],
                failure=float(m["physical_failure"]),
            )
            rows.append(row)
        if terminated or truncated:
            break
    data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    metrics = summarize(data, env.command_meta, info["physical_failure"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p.with_suffix(".npz"), **data, planned_commands=env._commands)
    result = dict(
        seed=STATE["seed"],
        method=STATE["method"],
        steps=STATE["steps"],
        phase=STATE["phase"],
        case=case,
        initial=initial,
        command=env.command_meta,
        metrics=metrics,
        checkpoint_sha256=STATE["checkpoint_sha256"],
        plan_sha256=STATE["plan_sha256"],
        trace_sha256=sha(p.with_suffix(".npz")),
        wall_s=time.time() - started,
    )
    write_json(p, result)
    return result


def evaluate(method, seed, output, checkpoint, workers=4, cases=None):
    protocol = json.loads((CONFIG / "evaluation.json").read_text())
    cases = protocol["cases"] if cases is None else cases
    output.mkdir(parents=True, exist_ok=True)
    args = dict(
        method=method,
        seed=seed,
        steps=json.loads((CONFIG / "experiments" / f"{method}_seed{seed}.json").read_text())["continuation_ppo"]["total_timesteps"],
        phase="validation",
        folder=str(output),
        checkpoint=str(checkpoint),
    )
    with ProcessPoolExecutor(
        max_workers=min(workers, len(cases)),
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(args,),
    ) as pool:
        rows = [
            f.result()
            for f in as_completed([pool.submit(episode, case) for case in cases])
        ]
    rows.sort(
        key=lambda r: (
            r["case"]["kind"],
            r["case"]["command_seed"],
            r["case"]["physics_seed"],
        )
    )
    keys = sorted(
        {
            k
            for r in rows
            for k, v in r["metrics"].items()
            if isinstance(v, (int, float))
        }
    )
    summary = dict(
        seed=seed,
        method=method,
        steps=json.loads((CONFIG / "experiments" / f"{method}_seed{seed}.json").read_text())["continuation_ppo"]["total_timesteps"],
        episodes=len(rows),
        means={
            k: float(np.mean([r["metrics"][k] for r in rows if k in r["metrics"]]))
            for k in keys
        },
        counts={k: sum(k in r["metrics"] for r in rows) for k in keys},
        checkpoint_sha256=sha(checkpoint),
    )
    write_json(output / "SUMMARY.json", summary)
    write_json(
        output / "COMPLETE.json",
        dict(
            episodes=len(rows),
            completed=time.time(),
            checkpoint_sha256=sha(checkpoint),
            summary_sha256=sha(output / "SUMMARY.json"),
        ),
    )
    return summary

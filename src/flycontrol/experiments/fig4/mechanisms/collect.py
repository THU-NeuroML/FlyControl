"""Record observations and verify complete 12-s rollouts against newly evaluated traces."""

import json
import time
import numpy as np
import torch
from body_control.task import ContinuousSpeedTask
from flycontrol.experiments.fig4.mechanisms.study import O, CASES, checkpoint, load_policy, name_case, evaluation_path, sha, write_json


@torch.inference_mode()
def run(method, seed):
    p = load_policy(method, seed)
    hsh = sha(checkpoint(method, seed))
    env = ContinuousSpeedTask(40, fixed_case={})
    for case in CASES:
        path = O / "rollouts" / f"{method}_{seed}" / (name_case(case) + ".npz")
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta["checkpoint_sha256"] == hsh and meta["sha256"] == sha(path)
            continue
        start = time.time()
        env.fixed_case = case
        obs, initial = env.reset()
        h = p.init_state(1, "cpu")
        observations = []
        actions = []
        trace = []
        speed_raw = []
        while True:
            observations.append(obs.copy())
            speed_raw.append(env.cycle_speed)
            act, _, _, h = p.act(torch.from_numpy(obs)[None], h, deterministic=True)
            actions.append(act[0].numpy())
            obs, _, term, trunc, info = env.step(actions[-1])
            trace.extend(
                [
                    [
                        r[k]
                        for k in [
                            "elapsed_s",
                            "target_applied_mm_s",
                            "filtered_body_forward_velocity_mm_s",
                            "body_forward_velocity_mm_s",
                        ]
                    ]
                    for r in env.last_micro_infos
                ]
            )
            if len(actions) % 75 == 0:
                print(
                    method,
                    seed,
                    case["kind"],
                    len(actions),
                    round(time.time() - start, 1),
                    flush=True,
                )
            if term or trunc:
                break
        if len(actions) != 300 or info["physical_failure"]:
            write_json(path.with_suffix('.incomplete.json'), dict(method=method, seed=seed, case=case,
                observations=len(actions), physical_failure=bool(info['physical_failure']), checkpoint_sha256=hsh))
            raise RuntimeError('The prespecified neural history ended before 12 s. Failure is recorded; do not pad or silently replace it.')
        oldp = evaluation_path(method, seed, case).with_suffix(".npz")
        old = np.load(oldp)
        t = np.asarray(trace)
        diffs = {
            k: float(np.max(np.abs(t[:, i] - old[k])))
            for i, k in enumerate(
                [
                    "elapsed_s",
                    "target_applied_mm_s",
                    "filtered_body_forward_velocity_mm_s",
                    "body_forward_velocity_mm_s",
                ]
            )
        }
        # Same CPU deterministic path must reproduce the source evaluation exactly.
        assert max(diffs.values()) < 1e-7, diffs
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=np.asarray(observations),
            gaussian_actions=np.asarray(actions),
            trace=t,
            raw_measured_speed=np.asarray(speed_raw),
        )
        write_json(
            meta_path,
            dict(
                method=method,
                seed=seed,
                case=case,
                training_decisions=15872,
                checkpoint_sha256=hsh,
                sha256=sha(path),
                reference_sha256=sha(oldp),
                baseline_max_abs_differences=diffs,
                wall_s=time.time() - start,
                initial=initial,
                observations=300,
                physical_failure=False,
            ),
        )
        print(
            "COMPLETE",
            method,
            seed,
            case["kind"],
            round(time.time() - start, 1),
            flush=True,
        )
    env.close()
    assert sha(checkpoint(method, seed)) == hsh

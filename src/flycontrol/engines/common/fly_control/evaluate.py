"""Frozen-checkpoint, reward-only development evaluation on one initial state.

Each worker has its own Python process so different frozen training sources
cannot share imported modules. All methods use the same CPU evaluation backend.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    conf = json.loads((root / "protocol.json").read_text())
    job = conf["jobs"][args.job]
    study = Path(job["training_root"])
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry).resolve() != Path(__file__).resolve().parent
    ]
    sys.path.insert(0, str(study / "source"))
    import numpy as np
    import torch
    from body_control.env import TaskEnv, BODY_CONFIG
    from body_control import env as environment_module
    from body_control import build_policy
    from fly_control.io import NoTeacherImports, write_json

    sys.meta_path.insert(0, NoTeacherImports())
    os.nice(10)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name, digest in json.loads(
        (study / "source_manifest.json").read_text()
    ).items():
        assert (
            hashlib.sha256((study / "source" / name).read_bytes()).hexdigest() == digest
        )
    run = study / "runs" / job["run"]
    initial = json.loads((run / "initialization.json").read_text())
    interface = initial.get("body_interface", "native")
    body = job["body"]
    method = job["method"]
    seed = job["evaluation_seed"]
    # The earliest corrected native study predates the interface keyword.
    options = {} if interface == "native" else dict(interface=interface)
    factory = getattr(environment_module, "make_env", TaskEnv)
    e = factory(body, **options)
    dimension = e.observation_space.shape[0]
    actions = e.nu
    e.close()
    policy = build_policy(
        study / "inputs/graph.npz",
        dimension,
        actions,
        method,
        initial["seed"],
        "cpu",
        initial["readout"],
        initial.get("body_config", BODY_CONFIG[body])["log_std"],
        **initial.get("policy_interface", {}),
    )
    checkpoint = run / f"policy_{job['training_steps']:07d}.pt"
    policy.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["policy_state"]
    )
    policy.eval()
    out = root / "evaluation" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "provenance.json",
        dict(
            job=job,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            frozen_source_manifest_sha256=hashlib.sha256(
                (study / "source_manifest.json").read_bytes()
            ).hexdigest(),
            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            teacher_used=False,
            evaluation_device="cpu",
        ),
    )
    for condition in conf["conditions"]:
        path = out / f"{condition}.json"
        if path.exists():
            continue
        command = job.get("command", BODY_CONFIG[body]["speed"])
        horizon = job.get("smoke_horizon_s", 60.0 if condition == "flat_60s" else 20.0)
        onset = job.get("perturb_onset_s", 8.0)
        e = factory(body, horizon_s=horizon, eval_speed=command, **options)
        try:
            obs, _ = e.reset(seed=seed)
            state = policy.init_state(1, "cpu")
            if condition == "friction_060":
                e.model.geom_friction[:, 0] *= 0.6
            prefix = 10 + (e.model.nq - 7) + (e.model.nv - 6)
            base = e.physics.observation_space.shape[0]
            physical_indices = np.r_[np.arange(prefix), np.arange(base - 3, base)]
            if interface == "cassie_centered_curriculum":
                # The new x-position input is a physical sensor, too.
                assert e.observation_space.shape[0] == base + 1
                physical_indices = np.r_[physical_indices, base]
            held = obs.copy()
            delay_steps = round(0.08 / e.dt)
            assert abs(delay_steps * e.dt - 0.08) < 1e-10
            delayed = deque(
                [np.zeros(actions, dtype=np.float32) for _ in range(delay_steps)]
            )
            mask = (
                np.random.default_rng(seed + 2000000).random(round(horizon / e.dt))
                < 0.25
            )
            elapsed = 0.0
            lateral = 0.0
            max_tilt = 0.0
            delivered = 0.0
            drops = 0
            delayed_ticks = 0
            reward_total = 0.0
            late_x = 0.0
            late_time = 0.0
            history = deque(maxlen=round(1 / e.dt))
            recovery = None
            strict_history = deque(maxlen=round(1 / e.dt))
            strict_recovery = None
            traces = []
            action_hash = hashlib.sha256()
            failed = False
            strength = (
                0.1
                if condition == "push_010"
                else 0.2 if condition == "push_020" else 0.0
            )
            planned = (
                strength * e.model.body_mass.sum() * np.sqrt(9.81 * e.reference_height)
            )
            direction = 1 if seed % 2 == 0 else -1
            with torch.no_grad():
                for step in range(round(horizon / e.dt)):
                    if step % 50 == 0:
                        control = json.loads((root / "control.json").read_text())
                        if control.get("stop_all") or control.get("yield_all"):
                            raise SystemExit(75)
                    policy_obs = obs.copy()
                    if (
                        condition == "sensor_loss_025"
                        and onset - 1e-9 <= elapsed < onset + 4 - 1e-9
                        and mask[step]
                    ):
                        policy_obs[physical_indices] = held[physical_indices]
                        drops += 1
                    else:
                        held = obs.copy()
                    raw, _, _, state = policy.act(
                        torch.as_tensor(policy_obs[None]), state, deterministic=True
                    )
                    raw = raw[0].numpy()
                    executed = raw
                    if condition == "action_delay_080":
                        delayed.append(raw.copy())
                        delayed_action = delayed.popleft()
                        if elapsed >= onset - 1e-9:
                            executed = delayed_action
                            delayed_ticks += 1
                    e.data.xfrc_applied[:] = 0.0
                    if strength and onset - 1e-9 <= elapsed < onset + 0.4 - 1e-9:
                        e.data.xfrc_applied[e.physics.trunk_id, 1] = (
                            direction * planned / 0.4
                        )
                    force = float(e.data.xfrc_applied[e.physics.trunk_id, 1])
                    old_x = float(e.data.qpos[0])
                    obs, reward, term, trunc, info = e.step(executed)
                    dt = info["control_duration_s"]
                    elapsed += dt
                    reward_total += reward
                    delivered += abs(force) * dt
                    action_hash.update(executed.tobytes())
                    if elapsed > 10 + 1e-8:
                        late_x += e.data.qpos[0] - old_x
                        late_time += dt
                    rot = e.data.xmat[e.physics.trunk_id].reshape(3, 3)
                    body_vx = float((rot.T @ e.data.qvel[:3])[0])
                    tilt = info["torso_tilt_degrees"]
                    lateral = max(
                        lateral, abs(float(e.data.qpos[1] - e.physics.origin[1]))
                    )
                    max_tilt = max(max_tilt, tilt)
                    history.append(
                        (
                            abs(body_vx) <= 0.05
                            if command < 0.02
                            else body_vx >= 0.5 * command
                        )
                        and tilt < 20
                    )
                    strict_history.append(
                        abs(body_vx - command) <= max(0.05, 0.5 * command) and tilt < 20
                    )
                    if strength and onset + 1.4 - 1e-9 <= elapsed <= onset + 6.4 + 1e-9:
                        if (
                            recovery is None
                            and len(history) == history.maxlen
                            and all(history)
                        ):
                            recovery = elapsed - (onset + 0.4)
                        if (
                            strict_recovery is None
                            and len(strict_history) == strict_history.maxlen
                            and all(strict_history)
                        ):
                            strict_recovery = elapsed - (onset + 0.4)
                    traces.append(
                        np.r_[
                            elapsed,
                            e.data.qpos.copy(),
                            e.data.qvel.copy(),
                            executed,
                            body_vx,
                            tilt,
                            force,
                        ]
                    )
                    failed = bool(term)
                    if term or trunc:
                        break
            speed = float((e.data.qpos[0] - e.physics.origin[0]) / horizon)
            survived = bool(not failed and elapsed >= horizon - 1e-6)
            result = dict(
                body=body,
                method=method,
                training_seed=initial["seed"],
                training_steps=job["training_steps"],
                evaluation_seed=seed,
                condition=condition,
                command=command,
                duration_s=elapsed,
                survived=survived,
                velocity_mps=speed,
                lateral_max_m=lateral,
                max_tilt_deg=max_tilt,
                late_velocity_mps=late_x / late_time if late_time else None,
                success=bool(
                    survived
                    and (
                        abs(speed) <= 0.05 if command < 0.02 else speed >= 0.7 * command
                    )
                    and lateral <= 0.75
                ),
                return_raw=reward_total,
                horizon_s=horizon,
                perturb_onset_s=onset,
                recovery_time_s=recovery,
                legacy_speed_tracking_recovery_time_s=strict_recovery,
                planned_push_impulse_Ns=float(planned),
                delivered_push_impulse_Ns=float(delivered),
                failed_before_push=bool(strength and elapsed < onset - 1e-9),
                sensor_packets_dropped=drops,
                delayed_action_ticks=delayed_ticks,
                action_delay_seconds=0.08 if condition == "action_delay_080" else 0.0,
                teacher_used=False,
                external_support_used=False,
                evaluation_device="cpu",
                action_sha256=action_hash.hexdigest(),
                nq=e.model.nq,
                nv=e.model.nv,
                nu=e.model.nu,
                trace_columns="time,qpos,qvel,executed_gaussian_action,body_vx,tilt,world_y_force",
            )
            if strength and elapsed >= onset + 0.4 - 1e-9:
                assert abs(delivered - planned) <= 1e-8 * max(1, planned), (
                    delivered,
                    planned,
                )
            np.savez_compressed(out / f"{condition}.npz", trajectory=np.asarray(traces))
            write_json(path, result)
            print(
                job["id"],
                condition,
                "success",
                result["success"],
                "speed",
                round(speed, 3),
                "lateral",
                round(lateral, 3),
                flush=True,
            )
        finally:
            e.close()
    write_json(
        out / "complete.json",
        dict(conditions=len(conf["conditions"]), teacher_used=False),
    )


if __name__ == "__main__":
    main()

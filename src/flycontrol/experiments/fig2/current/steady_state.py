"""Isolated warm-start diagnostic; the frozen training/evaluation code is imported read-only."""

from pathlib import Path
import os, sys, json, hashlib

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "egl")
from .paths import DATA, OUTPUT as ROOT
from fig2.runtime import activate

activate()
from terrain_control import task
from terrain_control.context import configure_process, write_json, sha

configure_process()
import numpy as np


def invalidate(env):
    if hasattr(env, "forward_cache_stats"):
        env.forward_cache_stats["cache_valid"] = False
    env._sim.physics.forward()


def raw(env):
    return env._sim.fly.get_observation(env._sim)


def capture_state(env):
    p = env._sim.physics
    d = p.data
    fields = {
        k: np.asarray(getattr(d, k)).copy()
        for k in [
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qacc_warmstart",
            "mocap_pos",
            "mocap_quat",
            "userdata",
            "qfrc_applied",
            "xfrc_applied",
        ]
    }
    fields.update(
        cpg_phases=env._cpg.curr_phases.copy(),
        cpg_magnitudes=env._cpg.curr_magnitudes.copy(),
        last_adhesion=np.asarray(env._sim.fly._last_adhesion).copy(),
        interval_force_min=env._interval_force_min.copy(),
        interval_force_peak_norm=env._interval_force_peak_norm.copy(),
        interval_force_samples=np.asarray(env._interval_force_samples),
    )
    return fields


def rebase_task_clock(env):
    env._physics_steps = 0
    env.elapsed = 0.0
    env._sim.curr_time = 0.0
    env._sim.physics.data.time = 0.0
    env.max_time = 1.2
    env.far = 4.3
    env.near = 3.0
    r = raw(env)
    env.start = r["fly"][0].copy()
    env._prev_fly_pos = env.start.copy()
    heading = r["fly_orientation"][:2].copy()
    env._initial_heading = heading / np.linalg.norm(heading)
    env.start_min_foot_x = float(r["end_effectors"][:, 0].min())
    env.previous_action = np.zeros(6)
    env.previous_progress = 0.0
    env.previous_heading_potential = 0.0
    env.max_lateral = 0.0
    env.clear_duration = 0.0
    env.episode_return = 0.0
    env.success = False
    env.last_info = {}
    env.last_micro_trace = []
    env.last_micro_infos = []
    env._history_tick = None
    env._history_frames = None
    return env.observation()


def install_state(env, state):
    """Restore a shared warmed body, preserving contact controls and velocities.

    Only horizontal translation is applied. No uprightness/velocity is fabricated.
    The original start x/y, obstacle coordinates and onset-gate coordinates are used.
    """
    p = env._sim.physics
    initial_position = raw(env)["fly"][0].copy()
    for k in [
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
        "userdata",
        "qfrc_applied",
        "xfrc_applied",
    ]:
        arr = getattr(p.data, k)
        assert arr.shape == state[k].shape, (k, arr.shape, state[k].shape)
        arr[:] = state[k]
    env._cpg.curr_phases = state["cpg_phases"].copy()
    env._cpg.curr_magnitudes = state["cpg_magnitudes"].copy()
    env._sim.fly._last_adhesion = state["last_adhesion"].copy()
    env._interval_force_min = state["interval_force_min"].copy()
    env._interval_force_peak_norm = state["interval_force_peak_norm"].copy()
    env._interval_force_samples = int(state["interval_force_samples"])
    invalidate(env)
    before = raw(env)
    free = np.flatnonzero(np.asarray(p.model.jnt_type) == 0)
    assert len(free) == 1
    adr = int(p.model.jnt_qposadr[free[0]])
    displacement = initial_position[:2] - before["fly"][0, :2]
    p.data.qpos[adr : adr + 2] += displacement
    invalidate(env)
    after = raw(env)
    np.testing.assert_allclose(
        after["fly"][0, :2], initial_position[:2], atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(p.data.qvel, state["qvel"], atol=0, rtol=0)
    assert after["end_effectors"][:, 0].max() < 3.0, "Obstacle overlaps initial feet"
    return rebase_task_clock(env)


def warm_bank(phase, duration=1.0):
    """Fixed-duration flat CPG run, independent of architecture or obstacle outcomes."""
    path = ROOT / "state_bank" / f"phase{phase}_{duration:g}s.npz"
    if path.exists():
        return json.loads(path.with_suffix(".json").read_text())
    env = task.ObstacleTask(
        fixed_cell=("blocks", 0.0), horizon=3.0, capture_micro_trace=True
    )
    samples = []
    times = []
    states = {}
    started = __import__("time").time()
    try:
        env.reset(seed=phase)
        env.far = 1e6
        while env.elapsed < duration - 1e-8:
            _, _, term, _, info = env.step(np.zeros(6, dtype=np.float32))
            for z in env.last_micro_trace:
                # Signed yaw and body velocity from the 172D frame (fixed scales).
                o = z["observation"].reshape(-1, 172)[-1]
                samples.append(
                    [
                        *z["position"],
                        float(z["up"][2]),
                        *o[46:49] * 10.0,
                        float(np.arctan2(o[25], o[24])),
                    ]
                )
                times.append(float(z["time_s"]))
            if info["physical_termination"] or info["numerical_error"]:
                break
        a = np.asarray(samples)
        t = np.asarray(times)
        end = env.elapsed
        late = a[(t > end - 0.25 + 1e-7) & (t <= end + 1e-7)]
        prev = a[(t > end - 0.50 + 1e-7) & (t <= end - 0.25 + 1e-7)]
        assert len(late) == 25 and len(prev) == 25, (len(late), len(prev), end)
        speed_l = float(late[:, 4].mean())
        speed_p = float(prev[:, 4].mean())
        yaw = np.unwrap(late[:, 7])
        metrics = dict(
            phase_seed=int(phase),
            warmup_s=end,
            min_up_last_250ms=float(late[:, 3].min()),
            mean_forward_speed_last_250ms=speed_l,
            mean_forward_speed_prev_250ms=speed_p,
            speed_change_fraction=float(
                abs(speed_l - speed_p) / max(abs(speed_p), 1.0)
            ),
            mean_body_z_change_mm=float(abs(late[:, 2].mean() - prev[:, 2].mean())),
            yaw_drift_last_250ms_rad=float(np.ptp(yaw)),
            final_heading_rad=float(a[-1, 7]),
            min_cpg_amplitude=float(env._cpg.curr_magnitudes.min()),
            physical_failure=bool(info["physical_termination"]),
            numerical_failure=bool(info["numerical_error"]),
            final_position_mm=a[-1, :3].tolist(),
        )
        metrics["stable"] = bool(
            end >= duration - 1e-8
            and metrics["min_up_last_250ms"] >= 0.95
            and speed_l >= 1.0
            and metrics["speed_change_fraction"] <= 0.15
            and metrics["mean_body_z_change_mm"] <= 0.03
            and metrics["yaw_drift_last_250ms_rad"] <= 0.05
            and metrics["min_cpg_amplitude"] >= 0.99
            and not metrics["physical_failure"]
            and not metrics["numerical_failure"]
        )
        fields = capture_state(env)
        fields.update(warmup_time_s=t, warmup_measurements=a)
        np.savez_compressed(path, **fields)
        metrics.update(
            state_sha256=sha(path), wall_s=__import__("time").time() - started
        )
        write_json(path.with_suffix(".json"), metrics)
        return metrics
    finally:
        env.close()

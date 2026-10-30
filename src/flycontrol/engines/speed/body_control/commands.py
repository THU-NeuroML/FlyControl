"""Exogenous continuous speed trajectories on one shared 10ms time grid."""

from .context import CONFIG
import hashlib, json
import numpy as np

PLAN = json.loads((CONFIG / "protocol.json").read_text())


def schedule(seed, kind=None):
    rng = np.random.default_rng(int(seed))
    dt = PLAN["micro_dt_s"]
    duration = PLAN["horizon_s"]
    t = np.arange(round(duration / dt) + 1) * dt
    kind = kind or str(rng.choice(["ramps", "sine"]))
    ramps = []
    if kind == "ramps":
        p = PLAN["ramp"]
        high = bool(rng.integers(2))
        v = float(rng.uniform(*(p["high_range"] if high else p["low_range"])))
        times = [0.0, p["initial_hold_s"]]
        values = [v, v]
        cursor = times[-1]
        while cursor < duration:
            high = not high
            nxt = float(rng.uniform(*(p["high_range"] if high else p["low_range"])))
            rate = float(rng.uniform(*p["requested_rate_range"]))
            length = max(p["minimum_ramp_duration_s"], abs(nxt - v) / rate)
            end = cursor + length
            ramps.append(
                dict(
                    start_s=cursor,
                    end_s=end,
                    start_speed=v,
                    end_speed=nxt,
                    slope=(nxt - v) / length,
                    complete_planned=end <= duration + 1e-10,
                )
            )
            times.append(end)
            values.append(nxt)
            cursor = end + float(rng.uniform(*p["endpoint_hold_s"]))
            times.append(cursor)
            values.append(nxt)
            v = nxt
        c = np.interp(t, times, values)
        extra = dict(knots=list(zip(times, values)), ramps=ramps)
    elif kind == "sine":
        p = PLAN["sine"]
        f = float(rng.uniform(*p["frequency_hz"]))
        a = float(rng.uniform(*p["amplitude_mm_s"]))
        center = float(rng.uniform(8 + a, 16 - a))
        phase = float(rng.uniform(0, 2 * np.pi))
        c = center + a * np.sin(2 * np.pi * f * t + phase)
        extra = dict(
            frequency_hz=f,
            amplitude_mm_s=a,
            center_mm_s=center,
            phase_rad=phase,
            ramps=[],
        )
    else:
        raise ValueError(kind)
    assert c.min() >= 8 - 1e-9 and c.max() <= 16 + 1e-9
    assert np.max(np.abs(np.diff(c))) <= 8 * dt + 1e-9
    assert np.any(np.diff(c) > 1e-8) and np.any(np.diff(c) < -1e-8)
    meta = dict(
        kind=kind,
        seed=int(seed),
        duration_s=duration,
        decision_dt_s=dt,
        command_sha256=hashlib.sha256(c.astype("<f8").tobytes()).hexdigest(),
        increases=int(np.sum(np.diff(c) > 1e-8)),
        decreases=int(np.sum(np.diff(c) < -1e-8)),
        **extra,
    )
    return c, meta

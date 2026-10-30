"""Fixed-time, failure-aware metrics; no model-dependent windows."""

from .commands import PLAN
import numpy as np


def causal_mean(x, n=10):
    x = np.asarray(x, dtype=float)
    cs = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    return (cs[i + 1] - cs[np.maximum(0, i + 1 - n)]) / np.minimum(i + 1, n)


def summarize(data, meta, failed):
    t = data["elapsed_s"]
    dt = data["interval_duration_s"]
    target = data["target_applied_mm_s"]
    raw = data["body_forward_velocity_mm_s"]
    smooth = causal_mean(raw)
    start = PLAN["measurement"]["analysis_start_s"]
    horizon = PLAN["horizon_s"]
    duration = horizon - start
    weights = np.maximum(0.0, np.minimum(t, horizon) - np.maximum(t - dt, start))
    observed = float(weights.sum())
    missing = max(0.0, duration - observed)
    error = smooth - target
    valid = weights > 0
    penalty = PLAN["measurement"]["failure_speed_error_mm_s"]
    failed_samples = np.asarray(data.get("failure", np.zeros(len(t))), dtype=bool)
    error = np.where(failed_samples, penalty, error)
    mae = (float(np.sum(np.abs(error) * weights)) + missing * penalty) / duration
    rmse = (
        (float(np.sum(error**2 * weights)) + missing * penalty**2) / duration
    ) ** 0.5
    e = np.r_[np.abs(error[valid]), np.full(round(missing / 0.01), penalty)]
    actions = np.c_[data["drive"], data["turn"]]
    changes = np.abs(np.diff(actions, axis=0, prepend=np.zeros((1, 2))))
    result = dict(
        speed_mae=mae,
        speed_rmse=rmse,
        error_p95=float(np.quantile(e, 0.95)) if len(e) else penalty,
        within1_fraction=float(np.sum(weights * (np.abs(error) <= 1)) / duration),
        within2_fraction=float(np.sum(weights * (np.abs(error) <= 2)) / duration),
        failure_rate=float(failed),
        missing_duration_s=missing,
        drive_tv_per_s=float(changes[valid, 0].sum() / max(observed, 0.01)),
        turn_tv_per_s=float(changes[valid, 1].sum() / max(observed, 0.01)),
        heading_mean_deg=float(
            np.rad2deg(np.sum(weights * np.abs(data["heading_error_rad"])))
            / max(observed, 0.01)
        ),
        saturation_fraction=(
            float(np.mean(np.abs(actions[valid]) > 0.95)) if valid.any() else 0.0
        ),
        complete=bool(not failed and t[-1] >= horizon - 1e-7),
    )
    ramps = []
    for trim in PLAN["measurement"]["ramp_sensitivity_s"] + [
        PLAN["measurement"]["ramp_trim_s"]
    ]:
        for segment in meta["ramps"]:
            if not segment["complete_planned"]:
                continue
            lo, hi = segment["start_s"] + trim, segment["end_s"] - trim
            mask = (t >= lo) & (t <= hi)
            complete = bool(
                t[-1] >= segment["end_s"] - 1e-8
                and mask.sum() >= 3
                and not np.any(failed_samples & (t <= segment["end_s"] + 1e-8))
            )
            slope = float(np.polyfit(t[mask], raw[mask], 1)[0]) if complete else None
            ramps.append(
                dict(
                    trim_s=trim,
                    start_s=segment["start_s"],
                    end_s=segment["end_s"],
                    target_slope=segment["slope"],
                    actual_slope=slope,
                    absolute_error=(
                        abs(slope - segment["slope"])
                        if complete
                        else PLAN["measurement"]["failure_slope_error_mm_s2"]
                    ),
                    observed_complete=complete,
                )
            )
    if ramps:
        primary = [
            r for r in ramps if r["trim_s"] == PLAN["measurement"]["ramp_trim_s"]
        ]
        result["slope_error"] = float(np.mean([r["absolute_error"] for r in primary]))
        for name, sgn in [("acceleration", 1), ("deceleration", -1)]:
            result[name + "_slope_error"] = float(
                np.mean(
                    [
                        r["absolute_error"]
                        for r in primary
                        if r["target_slope"] * sgn > 0
                    ]
                )
            )
        for trim in PLAN["measurement"]["ramp_sensitivity_s"]:
            result[f"slope_error_trim_{trim}"] = float(
                np.mean([r["absolute_error"] for r in ramps if r["trim_s"] == trim])
            )
    if meta["kind"] == "sine":
        mask = t >= 4.0
        f = meta["frequency_hz"]
        theta = 2 * np.pi * f * t + meta["phase_rad"]
        if result["complete"]:
            X = np.c_[
                np.ones(mask.sum()),
                t[mask] - 8.0,
                np.sin(theta[mask]),
                np.cos(theta[mask]),
            ]
            for key, speed in [("sine", smooth), ("sine_raw", raw)]:
                coef = np.linalg.lstsq(X, speed[mask], rcond=None)[0]
                gain = float(np.hypot(coef[2], coef[3]) / meta["amplitude_mm_s"])
                result[key + "_gain"] = gain
                result[key + "_gain_error"] = abs(gain - 1)
                result[key + "_lag_ms"] = float(
                    -np.arctan2(coef[3], coef[2]) / (2 * np.pi * f) * 1000
                )
    result["ramp_events"] = ramps
    return result

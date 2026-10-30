"""Condition-resolved measurements for newly generated body trajectories.

No episode selection or action-to-energy proxy.
Window comparisons use the intersection of observed C/M evaluation pairs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fig4.statistics import holm
from locomotion.statistics import signflip
from .paths import ROOT, read_json, write_json, sha256

KEY = ["body", "condition", "training_seed", "evaluation_seed"]
GROUP = ["body", "condition", "training_seed"]
METHODS = ["connectome", "mlp"]
WINDOW_METRICS = {
    "tilt_rms_deg": ("deg", -1),
    "angular_xy_rms_rad_s": ("rad/s", -1),
    "lateral_rms_m": ("m", -1),
    "lateral_drift_m_s": ("m/s", -1),
    "lateral_velocity_rms_m_s": ("m/s", -1),
    "heading_rms_deg": ("deg", -1),
    "heading_change_deg": ("deg", -1),
    "height_sd_m": ("m", -1),
    "speed_mean_error_m_s": ("m/s", -1),
    "speed_rmse_1s_m_s": ("m/s", -1),
    "speed_variability_1s_m_s": ("m/s", -1),
    "forward_speed_m_s": ("m/s", 1),
    "heading_drift_component_m": ("m", -1),
    "body_lateral_component_m": ("m", -1),
}
BASE_METRICS = {
    "task_score": ("fraction", 1),
    "success": ("fraction", 1),
    "survived": ("fraction", 1),
    "duration_s": ("s", 1),
    "lateral_max_m": ("m", -1),
    "max_tilt_deg": ("deg", -1),
    "velocity_mps": ("m/s", 1),
    "horizon_speed_error_m_s": ("m/s", -1),
    "physical_failure": ("fraction", -1),
    "lateral_only_failure": ("fraction", -1),
    "speed_only_failure": ("fraction", -1),
    "speed_and_lateral_failure": ("fraction", -1),
    "recovered_within_5s": ("fraction", 1),
    "posture_speed_criterion_5s": ("fraction", 1),
    **{f"corridor_{int(x * 100):03d}_time_s": ("s", 1) for x in [0.25, 0.5, 0.75, 1.0]},
    **{f"tilt_{x:02d}_time_s": ("s", 1) for x in [10, 20, 30]},
}


def windows(row):
    onset = row["perturb_onset_s"]
    return {
        "early": (0.0, 1.0),
        "pre": ((0.5, 1.0) if row["body"] == "cassie" else (2.0, 8.0)),
        "post": (onset + 0.4, onset + 2.4),
        "late": (10.0, 20.0),
        "endurance": (40.0, 60.0),
    }


def restricted_time(t, bad, duration, horizon):
    """Time to first adverse state or physical termination, horizon-censored."""
    hits = np.flatnonzero(bad)
    return min(float(t[hits[0]]) if len(hits) else horizon, duration, horizon)


def extract(item):
    row, path = item
    blob = path.read_bytes()
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        tr = archive["trajectory"]
    nq, nv, nu = (int(row[k]) for k in ["nq", "nv", "nu"])
    assert tr.ndim == 2 and tr.shape[1] == 4 + nq + nv + nu and np.isfinite(tr).all(), (
        path
    )
    t = tr[:, 0]
    assert np.all(np.diff(t) > 0) and abs(t[-1] - row["duration_s"]) < 1e-7, path
    q = tr[:, 1 : 1 + nq]
    v = tr[:, 1 + nq : 1 + nq + nv]
    w, x, y, z = q[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    tilt = tr[:, -2]
    assert abs(float(tilt.max()) - row["max_tilt_deg"]) < 1e-7, path
    lateral = q[:, 1] - q[0, 1]
    row = dict(row)
    slow = (
        abs(row["velocity_mps"]) > 0.05
        if row["command"] < 0.02
        else row["velocity_mps"] < 0.7 * row["command"]
    )
    off = row["lateral_max_m"] > 0.75
    failure = (
        "success"
        if row["success"]
        else "physical_failure"
        if not row["survived"]
        else "speed_and_lateral_failure"
        if slow and off
        else "speed_only_failure"
        if slow
        else "lateral_only_failure"
    )
    row.update(
        failure_class=failure,
        trajectory_file=str(path.relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else str(path),
        trajectory_sha256=hashlib.sha256(blob).hexdigest(),
        trace_rows=len(t),
        first_saved_time_s=t[0],
        horizon_speed_error_m_s=abs(row["velocity_mps"] - row["command"]),
    )
    for name in [
        "physical_failure",
        "speed_only_failure",
        "lateral_only_failure",
        "speed_and_lateral_failure",
    ]:
        row[name] = float(failure == name)
    push = row["condition"].startswith("push_")
    row["recovered_within_5s"] = (
        float(pd.notna(row["recovery_time_s"]) and row["recovery_time_s"] <= 5 + 1e-7)
        if push
        else np.nan
    )
    row["push_dose_fraction"] = (
        row["delivered_push_impulse_Ns"] / row["planned_push_impulse_Ns"]
        if push
        else np.nan
    )
    # Apply the original recovery criterion to sham (flat) episodes too, so
    # baseline movement quality is not mistaken for an induced push response.
    good = (
        (abs(tr[:, -3]) <= 0.05)
        if row["command"] < 0.02
        else (tr[:, -3] >= 0.5 * row["command"])
    ) & (tilt < 20)
    ticks = round(1 / np.median(np.diff(t)))
    sustained = (
        np.convolve(good.astype(int), np.ones(ticks, dtype=int), mode="full")[: len(t)]
        == ticks
    )
    onset = row["perturb_onset_s"]
    eligible = (t >= onset + 1.4 - 1e-7) & (t <= onset + 5.4 + 1e-7)
    row["posture_speed_criterion_5s"] = float(np.any(sustained & eligible))
    if push:
        assert row["posture_speed_criterion_5s"] == row["recovered_within_5s"], path
    for threshold in [0.25, 0.5, 0.75, 1.0]:
        row[f"corridor_{int(threshold * 100):03d}_time_s"] = restricted_time(
            t, abs(lateral) > threshold, row["duration_s"], row["horizon_s"]
        )
    for threshold in [10, 20, 30]:
        row[f"tilt_{threshold:02d}_time_s"] = restricted_time(
            t, tilt > threshold, row["duration_s"], row["horizon_s"]
        )
    wr = []
    for name, (lo, hi) in windows(row).items():
        record = {k: row[k] for k in KEY + ["method"]}
        record.update(
            window=name, start_s=lo, end_s=hi, complete=bool(t[-1] >= hi - 1e-7)
        )
        if record["complete"]:
            mask = (t > lo + 1e-7) & (t <= hi + 1e-7)
            qq, vv, yy = q[mask], v[mask], yaw[mask]
            # All windows are regular except a possible final terminal half-tick.
            weights = np.diff(np.r_[max(lo, 0.0), t[mask]])
            weights /= weights.sum()
            rms = lambda a: float(np.sqrt(np.sum(np.asarray(a) ** 2 * weights)))
            endpoints = np.r_[max(lo, t[0]), hi]
            dx = np.diff(np.interp(endpoints, t, q[:, 0]))[0]
            dy = np.diff(np.interp(endpoints, t, q[:, 1]))[0]
            duration = endpoints[1] - endpoints[0]
            # Whole-second pace, rather than stride-phase instantaneous speed.
            bins = np.arange(lo, hi + 1e-7, 1.0)
            bins = bins[bins >= t[0] - 1e-7]
            speeds = np.diff(np.interp(bins, t, q[:, 0])) / np.diff(bins)
            mid = (yy[1:] + yy[:-1]) / 2
            dxx, dyy = np.diff(qq[:, :2], axis=0).T
            along = np.cos(mid) * dxx + np.sin(mid) * dyy
            across = -np.sin(mid) * dxx + np.cos(mid) * dyy
            hd = np.sin(mid) * along
            bd = np.cos(mid) * across
            np.testing.assert_allclose(hd + bd, dyy, atol=1e-10)
            record.update(
                tilt_rms_deg=rms(tilt[mask]),
                angular_xy_rms_rad_s=rms(np.linalg.norm(vv[:, 3:5], axis=1)),
                lateral_rms_m=rms(lateral[mask]),
                lateral_drift_m_s=abs(dy) / duration,
                lateral_velocity_rms_m_s=rms(vv[:, 1]),
                heading_rms_deg=rms(np.degrees(np.arctan2(np.sin(yy), np.cos(yy)))),
                heading_change_deg=float(np.degrees(np.max(abs(yy - yy[0])))),
                height_sd_m=float(
                    np.sqrt(
                        np.sum(weights * (qq[:, 2] - np.sum(weights * qq[:, 2])) ** 2)
                    )
                ),
                speed_mean_error_m_s=abs(dx / duration - row["command"]),
                speed_rmse_1s_m_s=float(
                    np.sqrt(np.mean((speeds - row["command"]) ** 2))
                )
                if len(speeds)
                else np.nan,
                speed_variability_1s_m_s=float(np.std(speeds))
                if len(speeds) >= 2
                else np.nan,
                forward_speed_m_s=dx / duration,
                heading_drift_component_m=float(abs(hd.sum())),
                body_lateral_component_m=float(abs(bd.sum())),
            )
        wr.append(record)
    # Decimated views are for display only; all endpoints use every saved tick.
    grid = np.arange(0.25, 60.001, 0.25)
    curve = np.full((len(grid), 4), np.nan)
    observed = grid <= t[-1] + 1e-7
    for i, signal in enumerate([lateral, tilt, yaw * 180 / np.pi, q[:, 0] - q[0, 0]]):
        curve[observed, i] = np.interp(grid[observed], t, signal)
    return row, wr, curve


def paired_summary(frame, metric, unit, direction, window="episode", expected_seeds=6):
    """Match evaluation states first; require the full planned seed cohort for tests."""
    # Greater forward speed is not an improvement in Cassie's zero-command
    # balance task. Preserve raw speed in the data, but test its absolute error.
    if metric in {"velocity_mps", "forward_speed_m_s"}:
        frame = frame[frame.body != "cassie"]
    pivot = frame.pivot(index=KEY, columns="method", values=metric).reindex(
        columns=METHODS
    )
    pivot = pivot.dropna(subset=METHODS)
    if pivot.empty:
        return [], []
    seeds = pivot.groupby(GROUP).mean()
    counts = pivot.groupby(GROUP).size()
    sr, comparisons = [], []
    for (body, condition), pairs in seeds.groupby(level=["body", "condition"]):
        arr = pairs[METHODS].to_numpy()
        delta = direction * (arr[:, 0] - arr[:, 1])
        n = len(delta)
        idx = np.random.default_rng(661).integers(0, n, (10000, n))
        ci = np.quantile(delta[idx].mean(1), [0.025, 0.975])
        c = dict(
            body=body,
            condition=condition,
            window=window,
            metric=metric,
            unit=unit,
            direction="C - M" if direction == 1 else "M - C",
            n_training_pairs=n,
            n_evaluation_pairs=int(counts.loc[(body, condition)].sum()),
            connectome_mean=arr[:, 0].mean(),
            mlp_mean=arr[:, 1].mean(),
            difference=delta.mean(),
            ci95_low=ci[0],
            ci95_high=ci[1],
            positive_seeds=int((delta > 1e-10).sum()),
            negative_seeds=int((delta < -1e-10).sum()),
            tied_seeds=int((abs(delta) <= 1e-10).sum()),
            leave_one_seed_out_min=float(
                min(np.delete(delta, i).mean() for i in range(n))
            )
            if n > 1
            else np.nan,
            leave_one_seed_out_max=float(
                max(np.delete(delta, i).mean() for i in range(n))
            )
            if n > 1
            else np.nan,
            p_raw=signflip(delta) if n == expected_seeds else np.nan,
            inferential_scope=f"exploratory {expected_seeds}-seed comparison"
            if n == expected_seeds
            else "descriptive incomplete training-seed coverage",
        )
        comparisons.append(c)
        for key, vals in pairs.iterrows():
            sr.append(
                dict(
                    zip(GROUP, key),
                    window=window,
                    metric=metric,
                    unit=unit,
                    n_evaluation_pairs=int(counts.loc[key]),
                    connectome=float(vals["connectome"]),
                    mlp=float(vals["mlp"]),
                    difference=float(direction * (vals["connectome"] - vals["mlp"])),
                )
            )
    return sr, comparisons


def summarize(episodes, window_frame, output, expected_seeds=6):
    seed_rows, comparisons = [], []
    for metric, (unit, direction) in BASE_METRICS.items():
        s, c = paired_summary(
            episodes, metric, unit, direction, expected_seeds=expected_seeds
        )
        seed_rows.extend(s)
        comparisons.extend(c)
    for window, frame in window_frame.groupby("window", sort=False):
        for metric, (unit, direction) in WINDOW_METRICS.items():
            s, c = paired_summary(
                frame, metric, unit, direction, window, expected_seeds
            )
            seed_rows.extend(s)
            comparisons.extend(c)
    # Within-policy perturbation contrasts remove the baseline difference.
    for scope, frame, metrics in [
        ("episode", episodes, BASE_METRICS),
        *[(w, f, WINDOW_METRICS) for w, f in window_frame.groupby("window")],
    ]:
        for metric, (unit, direction) in metrics.items():
            bycondition = frame.pivot(
                index=["body", "method", "training_seed", "evaluation_seed"],
                columns="condition",
                values=metric,
            ).astype(float)
            for cond in [
                "push_010",
                "push_020",
                "friction_060",
                "action_delay_080",
                "sensor_loss_025",
            ]:
                changes = (
                    (bycondition[cond] - bycondition["flat"])
                    .rename(metric)
                    .reset_index()
                )
                changes["condition"] = cond
                s, c = paired_summary(
                    changes,
                    metric,
                    unit,
                    direction,
                    scope + "__change_from_flat",
                    expected_seeds,
                )
                seed_rows.extend(s)
                comparisons.extend(c)
    tests = pd.DataFrame(comparisons)
    for suffix, grouped in [
        ("global", [(None, tests.index)]),
        ("body", [(b, f.index) for b, f in tests.groupby("body")]),
    ]:
        for _, indices in grouped:
            valid = tests.loc[indices, "p_raw"].dropna().index
            tests.loc[valid, "p_holm_" + suffix] = holm(
                tests.loc[valid, "p_raw"].to_numpy()
            )
            tests.loc[indices, "family_size_" + suffix] = len(valid)
    tests.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_pairs.csv", index=False)
    window_frame.groupby(["body", "condition", "window", "method"])["complete"].agg(
        ["sum", "count"]
    ).to_csv(output / "window_coverage.csv")
    episodes.groupby(["body", "condition", "method", "failure_class"]).size().rename(
        "episodes"
    ).to_csv(output / "failure_modes.csv")
    episodes[episodes.condition.str.startswith("push_")].groupby(
        ["body", "condition", "method"]
    ).agg(
        episodes=("push_dose_fraction", "size"),
        full_dose=("push_dose_fraction", lambda x: int((x >= 1 - 1e-7).sum())),
        no_dose=("push_dose_fraction", lambda x: int((x < 1e-7).sum())),
        recovered=("recovered_within_5s", "sum"),
    ).to_csv(output / "push_exposure.csv")
    return tests

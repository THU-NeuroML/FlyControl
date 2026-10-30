"""Audit generated episodes, then aggregate within training seed before inference."""

import itertools
import os
import sys
import numpy as np
import pandas as pd

from flycontrol.paths import ENGINES, read_json, write_json, sha256

METRICS = {
    "body": ["task_score", "success", "survived", "velocity_mps", "lateral_max_m", "max_tilt_deg"],
    "turning": ["terminal_error", "relative_command_roughness", "command_variation", "success", "failure"],
    "speed": ["speed_mae", "slope_error", "sine_gain_error", "failure_rate"],
    "terrain": ["success", "crossed_within_corridor", "residual_variation_per_observed_second", "max_lateral_mm"],
}


def signflip(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Sign-flip requires finite paired seed differences")
    if len(values) > 24:
        raise ValueError("Exact sign-flip is bounded to this study's 24 seeds")
    count = 0
    total = 1 << len(values)
    for start in range(0, total, 32768):
        bits = (np.arange(start, min(total, start + 32768), dtype=np.uint32)[:, None] >> np.arange(len(values), dtype=np.uint32)) & 1
        null = (2 * bits.astype(float) - 1) @ values
        count += int(np.count_nonzero(abs(null) >= abs(values.sum()) - 1e-12))
    return count / total


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values)-np.arange(len(values))) * values[order])
    result = np.empty_like(values)
    result[order] = np.minimum(1, adjusted)
    return result


def summarize(episodes, spec, metrics):
    groups = ["method", "seed", "condition"]
    seedmeans = episodes.groupby(groups, sort=True)[metrics].mean().reset_index()
    counts = episodes.groupby(groups)[metrics].count().reset_index()
    contrasts = []
    for condition, data in seedmeans.groupby("condition", sort=True):
        for metric in metrics:
            paired = data.pivot(index="seed", columns="method", values=metric)
            if not {"connectome", "mlp"} <= set(paired.columns):
                continue
            paired = paired.reindex(spec["seeds"])
            valid = paired[["connectome", "mlp"]].notna().all(axis=1)
            delta = (paired.loc[valid, "connectome"] - paired.loc[valid, "mlp"]).to_numpy()
            lo = hi = p = None
            # n=1 smoke runs carry observations, never inferential intervals.
            if len(delta) >= 2 and not spec["smoke"]:
                draws = np.random.default_rng(20261020).choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
                lo, hi = map(float, np.quantile(draws, [0.025, 0.975]))
                p = signflip(delta)
            contrasts.append(dict(condition=condition, metric=metric, n_pairs=len(delta),
                omitted_incomplete_pairs=int((~valid).sum()), adult_minus_mlp=float(delta.mean()) if len(delta) else None,
                ci95_low=lo, ci95_high=hi, p_raw=p))
    result = pd.DataFrame(contrasts, columns=["condition", "metric", "n_pairs", "omitted_incomplete_pairs", "adult_minus_mlp", "ci95_low", "ci95_high", "p_raw"])
    result["p_holm_behavior_family"] = np.nan
    good = result.p_raw.notna()
    if good.any():
        # A non-estimable endpoint still occupies its predeclared family slot.
        corrected = holm(result.p_raw.fillna(1).to_numpy())
        result.loc[good, "p_holm_behavior_family"] = corrected[good]
    return seedmeans, counts, result


def analyze(root, spec):
    if spec['family'] in ('supplemental','historical','development_terrain'):
        from flycontrol.experiments.shared.analysis import analyze
        return analyze(root,spec)
    # Analysis imports only the pure metric functions from the numerical code.
    os.environ["FLYCONTROL_RUN"] = str(root)
    family = spec["family"]
    if family == "ant":
        from flycontrol.experiments.fig5.analysis import analyze
        return analyze(root, spec)
    records = read_json(root / "training/models.json")
    models = {(r["method"], r["seed"]): r["sha256"] for r in records}
    for r in records:
        if sha256(root / r["checkpoint"]) != r["sha256"]:
            raise ValueError("A trained model changed after evaluation")
    protocol = read_json(root / "protocol.json")
    rows, identities, expected, sources, window_rows = [], [], set(), {}, []
    if family == "body":
        from flycontrol.experiments.fig6.comprehensive import extract
        expected = set(itertools.product(spec["methods"], spec["seeds"], protocol["evaluation_seeds"], protocol["conditions"]))
        for path in sorted((root / "evaluation").rglob("*.json")):
            if path.stem not in protocol["conditions"]:
                continue
            r = read_json(path)
            identity = (r["method"], r["training_seed"], r["evaluation_seed"], r["condition"])
            provenance = read_json(path.parent / "provenance.json")
            if provenance["checkpoint_sha256"] != models[identity[:2]]:
                raise ValueError("Evaluation used a checkpoint outside this run")
            # Cassie is a balance task; the other bodies use locomotion success.
            r["task_score"] = min(1., r["duration_s"] / r["horizon_s"]) if r["body"] == "cassie" else float(r["success"])
            if not spec["smoke"]:
                r, windows, _ = extract((r, path.with_suffix(".npz")))
                window_rows.extend(windows)
            r["seed"] = r["training_seed"]
            rows.append(r); identities.append(identity)
            sources[str(path.relative_to(root))] = sha256(path)
            sources[str(path.with_suffix(".npz").relative_to(root))] = sha256(path.with_suffix(".npz"))
    elif family == "turning":
        from flycontrol.experiments.fig3.behavior import metrics
        expected = set(itertools.product(spec["methods"], spec["seeds"], protocol["final_phase_seeds"], [-90., -60., -30., 30., 60., 90.]))
        for path in sorted((root / "evaluation").glob("*/phase*.json")):
            r = read_json(path)
            identity = (r["method"], r["training_seed"], r["phase_seed"], r["case"]["angle_deg"])
            if r["checkpoint_sha256"] != models[identity[:2]]:
                raise ValueError("Evaluation used a checkpoint outside this run")
            values = metrics(path)
            values["condition"] = f"{r['case']['angle_deg']:g}°"
            rows.append(values); identities.append(identity)
            sources[str(path.relative_to(root))] = sha256(path)
    elif family == "speed":
        sys.path[:0] = [str(ENGINES / "speed"), str(ENGINES / "common")]
        from body_control.metrics import summarize as speed_metrics
        cases = read_json(root / "configs/figure4/evaluation.json")["cases"]
        expected = {(m, s, c["kind"], c["command_seed"], c["physics_seed"]) for m in spec["methods"] for s in spec["seeds"] for c in cases}
        for path in sorted((root / "evaluation").glob("*/*_c*_p*.json")):
            r = read_json(path); c = r["case"]
            identity = (r["method"], r["seed"], c["kind"], c["command_seed"], c["physics_seed"])
            if r["checkpoint_sha256"] != models[identity[:2]] or r["trace_sha256"] != sha256(path.with_suffix(".npz")):
                raise ValueError("Speed model/trace checksum changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as trace:
                values = speed_metrics(trace, r["command"], bool(r["metrics"]["failure_rate"]))
            values = {k: v for k, v in values.items() if isinstance(v, (int, float, bool))}
            rows.append(dict(method=r["method"], seed=r["seed"], condition="all histories", **values))
            identities.append(identity)
            sources[str(path.relative_to(root))] = sha256(path)
            sources[str(path.with_suffix(".npz").relative_to(root))] = sha256(path.with_suffix(".npz"))
    elif family == "terrain":
        grid = read_json(root / "evaluation/grid.json")
        expected = set(itertools.product(spec["methods"], spec["seeds"], grid["heights"], grid["phases"]))
        for path in sorted((root / "evaluation").glob("*/*/result.json")):
            r = read_json(path)
            identity = (r["method"], r["training_seed"], r["height"], r["phase_seed"])
            if r["checkpoint_sha256"] != models[identity[:2]] or r["trace_sha256"] != sha256(path.parent / "trace.npz"):
                raise ValueError("Terrain model/trace checksum changed")
            rows.append(dict(r, seed=r["training_seed"], condition=f"{r['height']:g} mm")); identities.append(identity)
            sources[str(path.relative_to(root))] = sha256(path)
            sources[str((path.parent / "trace.npz").relative_to(root))] = sha256(path.parent / "trace.npz")
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError(f"Incomplete/duplicate evaluation grid: {len(identities)} rows, {len(expected)} expected; {len(expected-set(identities))} missing")
    episodes = pd.DataFrame(rows)
    metrics = [m for m in METRICS[family] if m in episodes]
    seedmeans, counts, comparisons = summarize(episodes, spec, metrics)
    output = root / "analysis"; output.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output / "episodes.csv", index=False)
    seedmeans.to_csv(output / "seed_means.csv", index=False)
    counts.to_csv(output / "observed_counts.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    if window_rows:
        pd.DataFrame(window_rows).to_csv(output / "body_windows.csv", index=False)
    from flycontrol.analysis.behavior import behavior
    behavior(root, spec, episodes)
    from flycontrol.analysis.mechanisms import analyze as analyze_neural
    analyze_neural(root,spec)
    write_json(output / "manifest.json", dict(sources=sources, episodes=len(rows), complete_grid=True,
        statistical_unit="training seed; average evaluation repeats within each seed first",
        difference="Adult minus MLP; units and desirability depend on metric",
        correction="Holm across all reported behavior metric × condition contrasts in this run",
        scope="Behavior diagnostics; full paper mechanism/statistical families are separate",
        smoke=spec["smoke"], metrics=metrics,
        tables={p.name: sha256(p) for p in output.glob("*.csv")}))

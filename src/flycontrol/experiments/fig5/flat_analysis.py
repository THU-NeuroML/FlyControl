"""Audited 20-pair flat-ground statistics and descriptive S7 friction results."""

import itertools
import numpy as np
import pandas as pd

from flycontrol.analysis import signflip, holm
from flycontrol.paths import CONFIGS, read_json, write_json, sha256, source_digest

REFERENCE = CONFIGS / "reference/ant_capacity_20261030.json"
ENDPOINTS = {"velocity": ("flat", "mean_x_velocity"),
             "efficiency": ("flat_motion", "path_efficiency"),
             "failure": ("flat", "functional_failure")}
COLORS = {"connectome": "#B65A46", "mlp": "#5E7D95"}
LABELS = {"connectome": "Adult", "mlp": "Matched MLP"}


def audit(frame, role, spec, protocol):
    """Require the complete specified reset grid, including unsuccessful policies."""
    levels = protocol["frictions"] if role == "friction" else [1.]
    n = 1 if spec["smoke"] else protocol["friction_episodes" if role == "friction" else "episodes"]
    start = protocol["friction_reset_seed" if role == "friction" else "flat_reset_seed"]
    keys = ["method", "training_seed", "friction", "seed"]
    expected = {(m, s, f, start + i*10000 + r) for m, s in itertools.product(spec["methods"], spec["seeds"])
                for i, f in enumerate(levels) for r in range(n)}
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError(f"Incomplete/duplicate {role} cohort or incorrect reset seeds")
    if not frame.action_scale.eq(protocol["action_scale"]).all():
        raise ValueError("Evaluation action scale changed")
    if role == "flat_motion":
        v = frame.path_efficiency.to_numpy(dtype=float)
        if not np.isfinite(v).all() or (abs(v) > 1+1e-12).any():
            raise ValueError("Invalid signed path efficiency")
        ratio = frame.forward_progress_m / frame.planar_path_length_m.clip(lower=1e-12)
        if not np.allclose(v, ratio, atol=1e-10):
            raise ValueError("Path efficiency does not match displacement/path length")
    else:
        if not np.isfinite(frame.mean_x_velocity.astype(float)).all():
            raise ValueError("Non-finite all-episode velocity")
        inverted = ~frame.fall.astype(bool) & frame.final_torso_tilt_degrees.gt(90)
        if not (frame.inverted_completion == inverted).all():
            raise ValueError("Inverted completion disagrees with native termination/terminal tilt")
        if not (frame.functional_failure == (frame.fall | frame.inverted_completion)).all():
            raise ValueError("Functional failure must include native termination or inverted completion")


def compute(root, spec, frames, timecourses, sources, origin):
    protocol = read_json(root / "protocol.json")
    reference = read_json(REFERENCE)
    if not spec["smoke"] and (set(spec["methods"]) != {"connectome", "mlp"} or sorted(spec["seeds"]) != list(range(20))):
        raise ValueError("Full flat-ground inference requires both methods and all 20 training seeds")
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for role, frame in frames.items():
        audit(frame, role, spec, protocol)
        frame.to_csv(output / f"{role}_episodes.csv", index=False)
    n = len(spec["seeds"])
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(0, n, (protocol["bootstrap_resamples"], n))
    means, tests, summary = [], [], []
    for endpoint, (role, field) in ENDPOINTS.items():
        frame = frames[role]
        wide = frame.groupby(["training_seed", "method"])[field].mean().unstack().reindex(spec["seeds"])
        for method in spec["methods"]:
            values = wide[method].to_numpy(float)
            means.extend(dict(method=method, seed=s, metric=field, value=float(v)) for s, v in zip(spec["seeds"], values))
            lo, hi = [np.nan, np.nan] if spec["smoke"] else np.quantile(values[draws].mean(1), [.025, .975])
            summary.append(dict(endpoint=endpoint, method=method, mean=values.mean(), ci95_low=lo, ci95_high=hi, n_seeds=n))
        if {"connectome", "mlp"} <= set(wide):
            delta = (wide.connectome - wide.mlp).to_numpy(float)
            row = dict(contrast="matched_flat_"+endpoint, metric=field, n_pairs=n,
                adult_mean=wide.connectome.mean(), mlp_mean=wide.mlp.mean(), difference=delta.mean(),
                p_raw=np.nan, p_holm_29=np.nan, ci95_low=np.nan, ci95_high=np.nan)
            if not spec["smoke"]:
                lo, hi = np.quantile(delta[draws].mean(1), [.025, .975])
                row.update(p_raw=signflip(delta), ci95_low=lo, ci95_high=hi)
            tests.append(row)
    # Preserve every original hypothesis. The new matched-control endpoints are additions.
    original = pd.DataFrame(reference["original_26"])
    if len(original) != 26 or original.key.duplicated().any():
        raise ValueError("Invalid original 26-test family")
    results = pd.DataFrame(tests, columns=["contrast", "metric", "n_pairs", "adult_mean", "mlp_mean", "difference",
                                         "p_raw", "p_holm_29", "ci95_low", "ci95_high"])
    if not spec["smoke"]:
        family = original.rename(columns={"key": "contrast", "p_exact_two_sided": "p_raw"})
        family["origin"] = "archived original comparison"
        current = results.copy(); current["origin"] = origin
        family = pd.concat([family, current], ignore_index=True)
        family["p_holm_29"] = holm(family.p_raw)
        results["p_holm_29"] = family.p_holm_29.to_numpy()[-3:]
        family.to_csv(output / "family_29.csv", index=False)
    results.to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(means).to_csv(output / "seed_means.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "endpoint_summary.csv", index=False)

    friction = frames["friction"].groupby(["method", "training_seed", "friction"])[["mean_x_velocity", "functional_failure"]].mean().reset_index()
    friction.to_csv(output / "S7_seed_means.csv", index=False)
    summaries = []
    for method in spec["methods"]:
        for metric in ("mean_x_velocity", "functional_failure"):
            mat = friction[friction.method.eq(method)].pivot(index="training_seed", columns="friction", values=metric).reindex(index=spec["seeds"], columns=protocol["frictions"]).to_numpy(float)
            lo, hi = (np.full(4, np.nan), np.full(4, np.nan)) if spec["smoke"] else np.quantile(mat[draws].mean(1), [.025, .975], axis=0)
            summaries.extend(dict(method=method, metric=metric, friction=f, mean=y, ci95_low=l, ci95_high=h, n_seeds=n)
                for f, y, l, h in zip(protocol["frictions"], mat.mean(0), lo, hi))
    pd.DataFrame(summaries).to_csv(output / "S7_summary.csv", index=False)

    # Empirical CDF over all observed thresholds, averaging equally over training seeds.
    motion = frames["flat_motion"]
    grid = np.unique(np.r_[-1., motion.path_efficiency.to_numpy(float), 1.])
    weights = np.stack([(draws == i).sum(1) for i in range(n)], axis=1) / n
    cdf = []
    for method in spec["methods"]:
        q = motion[motion.method.eq(method)]
        matrix = np.array([(q[q.training_seed.eq(s)].path_efficiency.to_numpy(float)[:, None] <= grid).mean(0)*100 for s in spec["seeds"]])
        for start in range(0, len(grid), 100):
            block = matrix[:, start:start+100]
            lo, hi = (np.full(block.shape[1], np.nan), np.full(block.shape[1], np.nan)) if spec["smoke"] else np.quantile(weights @ block, [.025, .975], axis=0)
            cdf.extend(dict(method=method, threshold=t, mean_percent=y, ci95_low=l, ci95_high=h)
                for t, y, l, h in zip(grid[start:start+100], block.mean(0), lo, hi))
    pd.DataFrame(cdf).to_csv(output / "path_efficiency_cdf.csv", index=False)
    timecourses.to_csv(output / "timecourses.csv", index=False)
    tables = {p.name: sha256(p) for p in sorted(output.glob("*.csv"))}
    write_json(output / "manifest.json", dict(origin=origin, smoke=spec["smoke"], source_sha256=source_digest(),
        reference_sha256=sha256(REFERENCE), sources=sources, tables=tables, seed_count=n,
        statistical_unit="training seed", excluded_episodes=0,
        inference="none (smoke)" if spec["smoke"] else "new flat endpoints: Holm 29; original maze: unchanged Holm 26; S7: descriptive",
        bootstrap_resamples=protocol["bootstrap_resamples"], bootstrap_seed=protocol["bootstrap_seed"]))


def analyze(root, spec):
    frames, sources, trajectories = {}, {}, []
    for role in ("flat", "flat_motion", "friction", "timecourse"):
        rows = []
        for method, seed in itertools.product(spec["methods"], spec["seeds"]):
            path = root / f"evaluation/{method}_seed{seed}/{role}.json"
            meta = read_json(path.with_suffix(".provenance.json"))
            if sha256(path) != meta["evaluation_sha256"] or sha256(root / meta["checkpoint"]) != meta["checkpoint_sha256"]:
                raise ValueError(f"Evaluation provenance mismatch: {path}")
            sources[str(path.relative_to(root))] = sha256(path)
            sources[meta["checkpoint"]] = meta["checkpoint_sha256"]
            rows.extend(dict(r, method=method, training_seed=seed) for r in read_json(path))
        if role == "timecourse":
            trajectories = pd.DataFrame(rows)
        else:
            frames[role] = pd.DataFrame(rows)
    curves = []
    for metric in ("forward", "absolute_lateral"):
        q = trajectories.groupby(["method", "time_seconds"])[metric].agg(["mean", "std"]).reset_index()
        q = q.rename(columns={"mean": "mean_m", "std": "sd_across_seed_means_m"}); q["metric"] = metric
        curves.append(q)
    compute(root, spec, frames, pd.concat(curves), sources, "fresh training and evaluation")


def reference(root):
    """Reanalyze published observations without claiming a new training replication."""
    if (root / "run.json").exists():
        raise ValueError("Use a separate output directory for archived data, not a training run")
    archive = read_json(REFERENCE)
    write_json(root / "protocol.json", archive["protocol"])
    write_json(root / "REFERENCE.json", dict(origin=archive["description"], archive_sha256=sha256(REFERENCE), source_sha256=source_digest()))
    spec = dict(experiment="ant-flat", methods=["connectome", "mlp"], seeds=list(range(20)), smoke=False)
    frames = {key: pd.DataFrame(value["data"], columns=value["columns"]) for key, value in archive["tables"].items()}
    curves = []
    for method, metrics in archive["seed_timecourses"].items():
        for metric, values in metrics.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (20, 1001) or not np.isfinite(values).all():
                raise ValueError("Invalid archived seed timecourses")
            curves.extend(dict(method=method, metric=metric, time_seconds=t, mean_m=y, sd_across_seed_means_m=sd)
                for t, y, sd in zip(archive["time_seconds"], values.mean(0), values.std(0, ddof=1)))
    curves = pd.DataFrame(curves)
    compute(root, spec, frames, curves, {}, "archived paper observations; no training rerun")
    plot(root, spec)
    print(pd.read_csv(root / "analysis/paired_comparisons.csv").to_string(index=False))


def plot(root, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = root / "analysis"
    manifest = read_json(analysis / "manifest.json")
    for name, digest in manifest["tables"].items():
        if sha256(analysis / name) != digest:
            raise ValueError("Analysis tables changed; rerun analysis")
    for name, digest in manifest["sources"].items():
        if sha256(root / name) != digest:
            raise ValueError("Evaluation inputs changed; rerun analysis")
    if manifest["reference_sha256"] != sha256(REFERENCE):
        raise ValueError("Archived comparison family changed; rerun analysis")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    out = root / "figures/paper"; out.mkdir(parents=True, exist_ok=True)
    exports = []
    mode = "SMOKE TEST — not paper results" if spec["smoke"] else "20 paired training seeds"

    def save(fig, ax, name, source):
        ax.set_title(mode, fontsize=8, pad=12)
        fig.canvas.draw()
        # Single-axis quantitative exports: check labels are contained by the figure.
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer)
        if tight.x0 < 0 or tight.y0 < 0 or tight.x1 > fig.bbox.width or tight.y1 > fig.bbox.height:
            raise ValueError(f"Clipped figure labels: {name}")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300)
        exports.append(dict(panel=name, source_table=source, source_sha256=sha256(analysis / source), labels_inside_canvas=True))
        plt.close(fig)

    curves = pd.read_csv(analysis / "timecourses.csv")
    for metric, label in (("forward", "Forward displacement (m)"), ("absolute_lateral", "Absolute lateral displacement (m)")):
        fig, ax = plt.subplots(figsize=(4.4, 3.3), layout="constrained")
        for method in spec["methods"]:
            q = curves[curves.method.eq(method) & curves.metric.eq(metric)].sort_values("time_seconds")
            ax.plot(q.time_seconds, q.mean_m, color=COLORS[method], label=LABELS[method])
            ax.fill_between(q.time_seconds, q.mean_m-q.sd_across_seed_means_m, q.mean_m+q.sd_across_seed_means_m, color=COLORS[method], alpha=.18, lw=0)
        ax.set(xlabel="Time (s)", ylabel=label); ax.legend(frameon=False)
        save(fig, ax, "F5b_"+metric, "timecourses.csv")
    cdf = pd.read_csv(analysis / "path_efficiency_cdf.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout="constrained")
    for method in spec["methods"]:
        q = cdf[cdf.method.eq(method)]
        ax.step(q.threshold, q.mean_percent, where="post", color=COLORS[method], label=LABELS[method])
        ax.fill_between(q.threshold, q.ci95_low, q.ci95_high, step="post", color=COLORS[method], alpha=.18, lw=0)
    tests = pd.read_csv(analysis / "paired_comparisons.csv")
    if not spec["smoke"] and tests.loc[tests.metric.eq("path_efficiency"), "p_holm_29"].iloc[0] < .05:
        ax.text(.5, .95, "*", transform=ax.transAxes, ha="center", va="top", fontsize=12)
    ax.set(xlim=(-1, 1), ylim=(0, 100), xlabel="Signed path efficiency", ylabel="Cumulative episodes (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, ax, "F5c_path_efficiency", "path_efficiency_cdf.csv")
    means = pd.read_csv(analysis / "seed_means.csv")
    summary = pd.read_csv(analysis / "endpoint_summary.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout="constrained")
    for i, method in enumerate(spec["methods"]):
        q = means[means.method.eq(method) & means.metric.eq("functional_failure")]
        ax.scatter(i+np.linspace(-.09, .09, len(q)), q.value*100, color=COLORS[method], s=13, alpha=.55)
        r = summary[summary.method.eq(method) & summary.endpoint.eq("failure")].iloc[0]
        ax.plot(i, r["mean"]*100, "_", color="black", ms=12)
        if not spec["smoke"]:
            ax.vlines(i, r.ci95_low*100, r.ci95_high*100, color="black", lw=1.2)
    ax.set(xticks=range(len(spec["methods"])), xticklabels=[LABELS[m] for m in spec["methods"]],
        xlim=(-.5, len(spec["methods"])-.5), ylim=(-2, 102), ylabel="Functional failure (%)")
    save(fig, ax, "F5d_failure", "endpoint_summary.csv")
    friction = pd.read_csv(analysis / "S7_summary.csv")
    for metric, label in (("mean_x_velocity", "Forward velocity (m/s)"), ("functional_failure", "Functional failure (%)")):
        fig, ax = plt.subplots(figsize=(4.4, 3.3), layout="constrained")
        scale = 100 if metric == "functional_failure" else 1
        for method in spec["methods"]:
            q = friction[friction.method.eq(method) & friction.metric.eq(metric)].sort_values("friction")
            ax.plot(q.friction, q["mean"]*scale, "o-", color=COLORS[method], label=LABELS[method])
            ax.fill_between(q.friction, q.ci95_low*scale, q.ci95_high*scale, color=COLORS[method], alpha=.18, lw=0)
        ax.set(xlabel="Floor friction coefficient", ylabel=label, xticks=[.5, .75, 1.25, 1.5]); ax.legend(frameon=False)
        if metric == "functional_failure": ax.set_ylim(0, 100)
        save(fig, ax, "S7_"+metric, "S7_summary.csv")
    write_json(out / "manifest.json", dict(panels=exports, origin=manifest["origin"], smoke=spec["smoke"],
        timecourse_band="±1 SD across seed means; terminal positions held to 50 s",
        other_intervals="pointwise 95% paired-seed bootstrap; none for smoke",
        layout="Single-axis quantitative panels; original qualitative demo images are not regenerated",
        s7="Descriptive, no significance tests", failure_panel="No nonsignificance label"))

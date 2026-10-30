"""Editable paired-seed diagnostic figures from this run's audited analysis."""

import numpy as np
import pandas as pd

from flycontrol.paths import read_json, write_json, sha256

LABELS = {
    "task_score": "Task score", "success": "Success fraction", "survived": "Survival fraction",
    "velocity_mps": "Forward velocity (m/s)", "lateral_max_m": "Maximum lateral displacement (m)",
    "max_tilt_deg": "Maximum tilt (°)", "terminal_error": "Terminal heading error (°)",
    "relative_command_roughness": "Relative command roughness", "command_variation": "Command total variation",
    "failure": "Failure fraction", "speed_mae": "Speed MAE (mm/s)", "slope_error": "Ramp slope error (mm/s²)",
    "sine_gain_error": "Sinusoid amplitude-ratio error", "failure_rate": "Failure fraction",
    "crossed_within_corridor": "Corridor crossing fraction", "max_lateral_mm": "Maximum lateral displacement (mm)",
    "residual_variation_per_observed_second": "Residual variation per second",
}


def plot(root, spec):
    if spec["family"] in ("supplemental","historical","development_terrain"):
        from flycontrol.plotting.paper import plot
        return plot(root,spec)
    if spec["family"] == "ant":
        from flycontrol.experiments.fig5.analysis import plot
        plot(root, spec)
        from flycontrol.plotting.paper import plot as panels
        return panels(root,spec)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    analysis = root / "analysis"
    manifest = read_json(analysis / "manifest.json")
    for name, digest in manifest["tables"].items():
        if sha256(analysis / name) != digest:
            raise ValueError("Analysis tables changed; rerun analysis before plotting")
    for relative, digest in manifest["sources"].items():
        if sha256(root / relative) != digest:
            raise ValueError("Evaluation inputs changed; rerun analysis before plotting")
    data = pd.read_csv(analysis / "seed_means.csv")
    comparisons = pd.read_csv(analysis / "paired_comparisons.csv")
    destination = root / "figures"; destination.mkdir(parents=True, exist_ok=True)
    exported, exclusions = [], []
    for metric in manifest["metrics"]:
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        conditions = list(data.condition.unique())
        if spec["family"] == "body":
            conditions = read_json(root / "protocol.json")["conditions"]
        elif spec["family"] == "turning":
            conditions.sort(key=lambda x: float(x.removesuffix("°")))
        elif spec["family"] == "terrain":
            conditions.sort(key=lambda x: float(x.removesuffix(" mm")))
        paired_plot = {"connectome", "mlp"} <= set(data.method)
        tick_labels = []
        for i, condition in enumerate(conditions):
            subset = data[data.condition == condition].pivot(index="seed", columns="method", values=metric)
            if paired_plot:
                values = (subset.connectome - subset.mlp).dropna().to_numpy()
                if not len(values):
                    tick_labels.append(str(condition).replace("_", "\n") + "\n(n=0)")
                    exclusions.append(dict(metric=metric, condition=condition, excluded_count=len(spec["seeds"])))
                    continue
                jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(i + jitter, values, s=14, color="#4979a5", alpha=0.65, linewidths=0, zorder=3)
                row = comparisons[(comparisons.metric == metric) & (comparisons.condition == condition)].iloc[0]
                mean, lo, hi = row.adult_minus_mlp, row.ci95_low, row.ci95_high
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(i, mean, yerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt="_", ms=12,
                                color="#17324d", capsize=3, lw=1.2, zorder=4)
            else:
                values = subset.iloc[:, 0].dropna().to_numpy()
                jitter = np.linspace(-.12, .12, len(values)) if len(values) > 1 else np.zeros(len(values))
                ax.scatter(i + jitter, values, s=15, color="#4979a5", linewidths=0)
            exclusions.append(dict(metric=metric, condition=condition, excluded_count=len(spec["seeds"])-len(values)))
            tick_labels.append(str(condition).replace("_", "\n") + f"\n(n={len(values)})")
        if paired_plot:
            ax.axhline(0, color="#999999", lw=.6, zorder=1)
        ax.set_xticks(range(len(conditions)), tick_labels)
        ax.set_xlim(-.5, len(conditions)-.5)
        if not paired_plot and metric in {"task_score", "success", "survived", "failure", "failure_rate", "crossed_within_corridor"}:
            ax.set_ylim(-.03, 1.03)
        ax.set_ylabel(("Adult − MLP\n" if paired_plot else "") + LABELS.get(metric, metric))
        mode = "SMOKE TEST — not paper results" if spec["smoke"] else f"{len(spec['seeds'])} training seeds"
        ax.set_title(f"{spec['experiment'].capitalize()} · {mode}", fontsize=9, pad=12)
        fig.supxlabel("Dots: training-seed means" + ("; bars: paired bootstrap 95% CI" if paired_plot and not spec["smoke"] else "; no inferential interval"), fontsize=8)
        fig.savefig(destination / f"{metric}.pdf")
        fig.savefig(destination / f"{metric}.svg")
        fig.savefig(destination / f"{metric}.png", dpi=600)
        exported.extend(f"{metric}.{extension}" for extension in ("pdf", "svg", "png"))
        plt.close(fig)
    write_json(destination / "manifest.json", dict(files=exported, smoke=spec["smoke"],
        source_table_sha256=sha256(analysis / "seed_means.csv"),
        role="Reproduction diagnostics; quantitative paired seed differences, no archived values",
        exclusions=exclusions, alignment="Single plot per exported figure; multi-panel alignment not applicable"))

    from flycontrol.plotting.paper import plot as paper_panels
    paper_panels(root,spec)

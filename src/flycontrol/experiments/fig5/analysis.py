"""Seven-model maze summaries and the complete 26-contrast Adult/MLP family."""

import itertools
import numpy as np
import pandas as pd

from flycontrol.paths import read_json, write_json, sha256
from flycontrol.analysis import signflip, holm

MAZE_METRICS = ["success", "physical_failure", "success_weighted_geodesic_efficiency", "mean_goal_velocity"]


def analyze(root, spec):
    protocol = read_json(root / "protocol.json")
    output = root / "analysis"; output.mkdir(parents=True, exist_ok=True)
    frames, hashes = {}, {}
    for task in (*protocol["tasks"], "flat_motion", "friction", "medium_original"):
        methods = spec["methods"] if task in protocol["tasks"] else [m for m in spec["methods"] if m in {"connectome", "mlp"}]
        rows = []
        for method, seed in itertools.product(methods, spec["seeds"]):
            path = root / f"evaluation/{method}_seed{seed}/{task}.json"
            records = read_json(path)
            meta = read_json(path.with_suffix(".provenance.json"))
            if sha256(root / meta["checkpoint"]) != meta["checkpoint_sha256"] or sha256(path) != meta["evaluation_sha256"]:
                raise ValueError(f"Evaluation provenance mismatch: {path}")
            n = (4 if task == "friction" else 1) if spec["smoke"] else protocol["episodes"]
            if len(records) != n or len({r["seed"] for r in records}) != n:
                raise ValueError(f"Incomplete/duplicate {task} cohort for {method}/{seed}")
            hashes[str(path.relative_to(root))] = sha256(path)
            rows.extend(dict(r, method=method, training_seed=seed, task=task) for r in records)
        if rows:
            frames[task] = pd.DataFrame(rows)
            frames[task].to_csv(output / f"{task}_episodes.csv", index=False)
    means = []
    for task in ("umaze", "medium", "large"):
        data = frames[task]
        for metric in MAZE_METRICS:
            for (method, seed), group in data.groupby(["method", "training_seed"]):
                values = group[metric].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite all-episode endpoint: {task}/{metric}")
                means.append(dict(task=task, method=method, seed=seed, metric=metric, value=float(values.mean()), episodes=len(values)))
    means = pd.DataFrame(means)
    means.to_csv(output / "seed_means.csv", index=False)
    means.groupby(["task", "method", "metric"]).value.agg(["mean", "std", "count"]).reset_index().to_csv(output / "model_summary.csv", index=False)
    contrasts = []
    primary = [
        ("flat_speed", "flat", "mean_x_velocity"), ("flat_efficiency", "flat_motion", "path_efficiency"),
        ("flat_failure", "flat", "functional_failure"), ("umaze_velocity", "umaze", "mean_goal_velocity"),
        ("umaze_success", "umaze", "success"), ("umaze_failure", "umaze", "physical_failure"),
        ("umaze_efficiency", "umaze", "success_weighted_path_efficiency"),
        ("medium_success", "medium", "success"), ("medium_failure", "medium", "physical_failure"),
        ("medium_efficiency", "medium", "success_weighted_geodesic_efficiency"),
        ("medium_velocity", "medium", "mean_goal_velocity"), ("medium_action_cost", "medium", "mean_action_cost"),
        ("medium_original_success", "medium_original", "success"), ("medium_original_failure", "medium_original", "physical_failure"),
    ]
    if {"connectome", "mlp"} <= set(spec["methods"]):
        contrasts = [(key, frames[task], metric) for key, task, metric in primary]
        for label, turns in (("3–4", [3, 4]), ("5", [5]), ("6–7", [6, 7])):
            contrasts.append(("medium_turns_" + label, frames["medium"][frames["medium"].shortest_path_turns.isin(turns)], "success"))
        for friction in protocol["robustness_frictions"]:
            for metric in ("mean_x_velocity", "functional_failure"):
                contrasts.append((f"friction_{friction}_{metric}", frames["friction"][frames["friction"].friction == friction], metric))
        medium = frames["medium"].copy()
        medium["deadline_auc_percent"] = 100 * (50 - medium.time_to_goal_seconds.where(medium.success, 50)) / 50
        contrasts.append(("deadline_auc", medium, "deadline_auc_percent"))
        if len(contrasts) != 26:
            raise ValueError("Incomplete registered comparison family")
    results = []
    indices = np.random.default_rng(20260907).integers(0, len(spec["seeds"]), size=(50000, len(spec["seeds"])))
    for key, frame, metric in contrasts:
        selected = frame[frame.method.isin(["connectome", "mlp"])]
        wide = selected.groupby(["training_seed", "method"])[metric].mean().unstack().reindex(spec["seeds"])
        row = dict(contrast=key, metric=metric, n_pairs=len(spec["seeds"]), status="complete")
        if not {"connectome", "mlp"} <= set(wide) or wide.isna().any().any():
            row["status"] = "incomplete stratum; no inference"
        else:
            delta = (wide.connectome - wide.mlp).to_numpy()
            row.update(adult_mean=float(wide.connectome.mean()), mlp_mean=float(wide.mlp.mean()), difference=float(delta.mean()))
            if len(delta) >= 2 and not spec["smoke"]:
                ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                row.update(p_raw=signflip(delta), ci95_low=float(ci[0]), ci95_high=float(ci[1]))
        results.append(row)
    tests = pd.DataFrame(results)
    if "p_raw" in tests:
        # Missing strata still occupy their original family slots; p=1 prevents
        # reducing the multiplicity penalty when a stratum is not estimable.
        tests["p_holm_26"] = holm(tests.p_raw.fillna(1.))
        tests.loc[tests.p_raw.isna(), "p_holm_26"] = np.nan
    tests.to_csv(output / "paired_comparisons.csv", index=False)
    # Supplementary friction belongs to the original ten-pair, 25-contrast
    # family, including its undisplayed flat and maze contrasts.
    friction_seeds=[s for s in protocol['friction_training_seeds'] if s in spec['seeds']]
    supplementary=[]
    for key,frame,metric in contrasts:
        if key=='deadline_auc':continue
        q=frame[frame.training_seed.isin(friction_seeds)&frame.method.isin(['connectome','mlp'])]
        wide=q.groupby(['training_seed','method'])[metric].mean().unstack().reindex(friction_seeds)
        row=dict(contrast=key,metric=metric,n_pairs=len(friction_seeds),p_raw=np.nan)
        if {'connectome','mlp'}<=set(wide) and not wide.isna().any().any():
            delta=(wide.connectome-wide.mlp).to_numpy();row.update(adult_mean=wide.connectome.mean(),mlp_mean=wide.mlp.mean(),difference=delta.mean())
            if not spec['smoke'] and len(delta)==10:
                draw=np.random.default_rng(20260907).integers(0,10,(50000,10));lo,hi=np.quantile(delta[draw].mean(1),[.025,.975]);row.update(p_raw=signflip(delta),ci95_low=lo,ci95_high=hi)
        supplementary.append(row)
    if supplementary:
        q=pd.DataFrame(supplementary);assert len(q)==25
        q['p_holm25']=holm(q.p_raw.fillna(1));q.loc[q.p_raw.isna(),'p_holm25']=np.nan;q.to_csv(output/'supplementary_25_comparisons.csv',index=False)

    # The S7 display uses ten seeds, while all twenty remain in the main family.
    if "friction" in frames:
        frames["friction"][frames["friction"].training_seed.isin(protocol["friction_training_seeds"])].to_csv(output / "supplementary_friction_episodes.csv", index=False)
    write_json(output / "manifest.json", dict(sources=hashes, smoke=spec["smoke"],
        statistical_unit="optimization seed, with shared source/teacher dependencies retained",
        complete_grid=True, reported_family_size=26, seed_count=len(spec["seeds"]),
        seed_means_sha256=sha256(output / "seed_means.csv")))


def plot(root, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    output = root / "figures"; output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "analysis/manifest.json")
    if sha256(root / "analysis/seed_means.csv") != manifest["seed_means_sha256"]:
        raise ValueError("Analyzed seed table changed")
    for relative, expected in manifest["sources"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Evaluation changed after analysis")
    data = pd.read_csv(root / "analysis/seed_means.csv")
    names = {"connectome": "Adult", "mlp": "MLP", "gru": "GRU", "lstm": "LSTM",
             "transformer": "Transformer", "reservoir": "Reservoir", "reservoir_ro": "Reservoir+RO"}
    labels = {"success": "Success fraction", "physical_failure": "Physical failure fraction",
              "success_weighted_geodesic_efficiency": "Success-weighted geodesic efficiency", "mean_goal_velocity": "Goal-directed velocity (m/s)"}
    for (task, metric), group in data.groupby(["task", "metric"]):
        fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
        for i, method in enumerate(spec["methods"]):
            values = group[group.method == method].value.to_numpy()
            color = "#4979a5" if method == "connectome" else "#87939c"
            jitter = np.linspace(-.15, .15, len(values)) if len(values) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), values, s=13, color=color, alpha=.65, linewidths=0)
            if len(values) > 1:
                ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="_", color="#243745", capsize=3, ms=12)
        ax.set_xticks(range(len(spec["methods"])), [names[m] for m in spec["methods"]], rotation=20, ha="right", rotation_mode="anchor")
        ax.set_xlim(-.5, len(spec["methods"])-.5)
        if spec["smoke"] and metric in {"success", "physical_failure", "success_weighted_geodesic_efficiency"}:
            ax.set_ylim(-.03, 1.03)
        ax.set_ylabel(labels[metric])
        ax.set_title(task.capitalize() + (" · SMOKE TEST — not paper results" if spec["smoke"] else " · fixed seed cohort"), fontsize=9)
        fig.supxlabel("Dots: seed means across fixed resets; " + ("no between-seed interval" if spec["smoke"] else "bars: mean ± 1 seed SD"), fontsize=8)
        fig.savefig(output / f"{task}_{metric}.pdf")
        fig.savefig(output / f"{task}_{metric}.svg")
        fig.savefig(output / f"{task}_{metric}.png", dpi=600)
        plt.close(fig)
    write_json(output / "manifest.json", dict(smoke=spec["smoke"], source_sha256=manifest["seed_means_sha256"],
        role="Seven-model maze comparison; every fixed training seed retained", alignment="Single plot per figure"))

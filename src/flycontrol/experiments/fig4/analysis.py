"""Rebuild panels B–G numerical inputs from the retained raw evidence."""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from fig6.paths import ROOT, read_json, write_json, sha256
from .statistics import contrast, holm

DATA = ROOT / "data/figure4"
CONFIG = read_json(ROOT / "configs/figure4/analysis.json")
SEEDS = read_json(ROOT / "run.json")["seeds"]
METHODS = ["connectome", "mlp"]
STARTS = CONFIG["ramp_window_starts_s"]
WIDTH = CONFIG["ramp_window_width_s"]
BASE = CONFIG["ramp_baseline_s"]
BOOT_SEED = CONFIG["bootstrap_seeds"]["ramp"]
ORDER = [
    "mechanosensory",
    "true_AN",
    "SA",
    "unknown_sensory",
    "gustatory",
    "visual",
    "hygrosensory",
    "thermosensory",
    "olfactory",
]


def behavior(output):
    from body_control.metrics import summarize

    rows = []
    cases = read_json(ROOT / "configs/figure4/evaluation.json")["cases"]
    for seed in SEEDS:
        for method in METHODS:
            values = []
            for case in cases:
                name = f"{case['kind']}_c{case['command_seed']}_p{case['physics_seed']}"
                path = DATA / "evaluation" / f"{method}_seed{seed}" / (name + ".json")
                meta = read_json(path)
                assert sha256(path.with_suffix(".npz")) == meta["trace_sha256"]
                with np.load(path.with_suffix(".npz")) as raw:
                    metrics = summarize(
                        raw, meta["command"], bool(meta["metrics"]["failure_rate"])
                    )
                for key, value in meta["metrics"].items():
                    if isinstance(value, (float, int)):
                        assert abs(float(value) - float(metrics[key])) < 1e-10, (
                            path,
                            key,
                        )
                values.append(metrics)
            keys = sorted(
                {k for v in values for k, x in v.items() if isinstance(x, (float, int))}
            )
            means = {k: float(np.mean([v[k] for v in values if k in v])) for k in keys}
            rows.append(
                dict(
                    phase="validation",
                    steps=15872,
                    seed=seed,
                    method=method,
                    **means,
                    episodes=len(values),
                    sine_complete_cases=sum("sine_gain_error" in v for v in values),
                    model="adult" if method == "connectome" else "mlp",
                )
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "short_budget_seed_metrics.csv", index=False)
    primary = []
    for metric in CONFIG["statistical_family"]:
        table = data.pivot(index="seed", columns="method", values=metric).loc[SEEDS]
        adult = table.connectome.to_numpy()
        mlp = table.mlp.to_numpy()
        primary.append(
            dict(
                steps=15872,
                cohort="all_20",
                metric=metric,
                adult_mean=adult.mean(),
                mlp_mean=mlp.mean(),
                adult_sd=adult.std(ddof=1),
                mlp_sd=mlp.std(ddof=1),
                adult_reduction_pct=100 * (mlp.mean() - adult.mean()) / mlp.mean(),
                **contrast(mlp - adult),
            )
        )
    for row, p in zip(primary, holm([v["raw_two_sided_signflip_p"] for v in primary])):
        row["holm_p_three_fixed_budget_endpoints"] = p
        row["adult_better_with_holm_p_lt_05"] = bool(
            row["difference_mean"] > 0 and p < 0.05
        )
    pd.DataFrame(primary).to_csv(
        output / "short_budget_primary_fixed_budget.csv", index=False
    )


def mechanisms(output, raw_root=DATA):
    inv = pd.read_csv(DATA / "anatomy/group_inventory.csv")
    classes = inv[inv.level == "input_class"]
    indices = {name: int(classes.index[classes.label == name][0]) for name in ORDER}
    local_rows = []
    signed_rows = []
    drives = []
    for seed in SEEDS:
        atlas = np.load(raw_root / "neurons" / str(seed) / "atlas.npz")
        local = atlas["local"]
        for ti, time in enumerate(atlas["time_ms"]):
            for pi, phase in enumerate(atlas["phases"]):
                for gi, row in classes.iterrows():
                    values = local[ti, pi, gi]
                    local_rows.append(
                        dict(
                            seed=seed,
                            phase=phase,
                            time_ms=time,
                            group=row.label,
                            n_cells=int(row.n_cells),
                            mean_squared_derivative=float(values[0]),
                            energy_fraction=float(values[1]),
                            enrichment=float(values[2]),
                        )
                    )
        signed = np.load(raw_root / "signed_replays" / str(seed) / "signed_neural.npz")[
            "values"
        ]
        for pi, phase in enumerate(["acceleration", "deceleration"]):
            for name in ORDER:
                for ti, time in enumerate(range(40, 1001, 40)):
                    signed_rows.append(
                        dict(
                            seed=seed,
                            phase=phase,
                            group=name,
                            time_ms=time,
                            signed_response=signed[ti, 0, pi, indices[name], 0],
                        )
                    )
        drives.append(
            pd.read_csv(raw_root / "events" / str(seed) / "drive_responses.csv")
        )
    pd.DataFrame(local_rows).to_csv(
        output / "latest_neuron_class_seed_times.csv", index=False
    )
    signed = pd.DataFrame(signed_rows)
    signed.to_csv(output / "latest_signed_class_seed_times.csv", index=False)
    signed = pd.read_csv(output / "latest_signed_class_seed_times.csv")
    heat = (
        signed.query('phase=="acceleration"')
        .pivot_table(index="group", columns="time_ms", values="signed_response")
        .loc[ORDER]
        * 1000
    )
    heat.to_csv(output / "acceleration_heatmap.csv")
    drives = pd.concat(drives, ignore_index=True)
    endpoint = (
        drives[drives.time_ms >= 800]
        .groupby(["seed", "method", "phase", "condition"])[
            ["signed_drive_change", "rms_drive_change"]
        ]
        .mean()
        .reset_index()
    )
    adult = endpoint.query('method=="connectome"')
    base = (
        adult[adult.condition == "hold_target"]
        .set_index(["seed", "phase"])
        .signed_drive_change
    )
    ablation = adult[adult.condition.str.startswith("hold_target_")].copy()
    ablation["source"] = ablation.condition.str.removeprefix("hold_target_")
    ablation["fraction_of_full_target_response"] = [
        r.signed_drive_change / base.loc[(r.seed, r.phase)]
        for r in ablation.itertuples()
    ]
    ablation.to_csv(output / "latest_event_class_ablation_seed.csv", index=False)
    grids = []
    for method in METHODS:
        values = []
        for seed in SEEDS:
            z = np.load(raw_root / "probes" / f"{method}_{seed}" / "surface.npz")
            values.append(z["metrics"])
        grids.append(np.stack(values))
    grids = np.stack(grids)
    np.savez_compressed(
        output / "latest_parameter_landscapes.npz",
        dose=z["dose"],
        metrics=grids,
        mean=grids.mean(1),
        seeds=SEEDS,
        methods=METHODS,
    )
    np.savez_compressed(
        output / "main_flatness_mean_grids.npz",
        dose=z["dose"],
        adult=grids[0, :, :, :, 0].mean(0),
        mlp=grids[1, :, :, :, 0].mean(0),
    )


def tracking(output):
    records = []
    sources = []
    for kind, case in [
        ("ramp", "ramps_c940001_p940201"),
        ("sine", "sine_c940101_p940201"),
    ]:
        for model, method in [("adult", "connectome"), ("mlp", "mlp")]:
            for seed in range(4401, 4421):
                p = DATA / "evaluation" / f"{method}_seed{seed}" / (case + ".npz")
                z = np.load(p)
                meta = json.loads(p.with_suffix(".json").read_text())
                assert meta["steps"] == 15872 and not z["failure"].any()
                onset = (
                    meta["command"]["ramps"][1]["start_s"] if kind == "ramp" else 0.0
                )
                t = z["elapsed_s"]
                x = t - onset
                keep = (
                    (x >= -0.6) & (x <= 1.8) if kind == "ramp" else (t >= 1) & (t <= 11)
                )
                records.append(
                    pd.DataFrame(
                        dict(
                            task=kind,
                            model=model,
                            seed=seed,
                            time_s=t[keep],
                            event_time_s=x[keep],
                            target_mm_s=z["target_applied_mm_s"][keep],
                            speed_mm_s=z["filtered_body_forward_velocity_mm_s"][keep],
                            raw_speed_mm_s=z["body_forward_velocity_mm_s"][keep],
                        )
                    )
                )
                sources.append(
                    dict(
                        task=kind,
                        model=model,
                        seed=seed,
                        path=str(p.relative_to(DATA)),
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        onset_s=onset,
                        command=meta["command"],
                    )
                )
    data = pd.concat(records, ignore_index=True)
    data.to_csv(output / "observed_tracking.csv", index=False)


def ramp_increments(output):
    d = pd.read_csv(output / "observed_tracking.csv")
    d = d[d.task == "ramp"].copy()
    base = (
        d[(d.event_time_s >= BASE[0]) & (d.event_time_s < BASE[1])]
        .groupby(["seed", "model"])
        .agg(
            baseline_speed_mm_s=("speed_mm_s", "mean"),
            baseline_target_mm_s=("target_mm_s", "mean"),
            baseline_n=("time_s", "size"),
        )
    )
    assert len(base) == 40 and base.baseline_n.eq(10).all()
    rows = []
    summaries = []
    paired = []
    idx = np.random.default_rng(BOOT_SEED).integers(0, 20, (20000, 20))
    for start in STARTS:
        win = (
            d[(d.event_time_s >= start) & (d.event_time_s < start + WIDTH)]
            .groupby(["seed", "model"])
            .agg(
                window_speed_mm_s=("speed_mm_s", "mean"),
                window_target_mm_s=("target_mm_s", "mean"),
                window_n=("time_s", "size"),
                first_time_s=("event_time_s", "min"),
                last_time_s=("event_time_s", "max"),
            )
        )
        assert len(win) == 40 and win.window_n.eq(10).all()
        w = win.join(base).reset_index()
        w["window_start_s"] = start
        w["window_end_s"] = start + WIDTH
        w["delta_speed_mm_s"] = w.window_speed_mm_s - w.baseline_speed_mm_s
        w["delta_target_mm_s"] = w.window_target_mm_s - w.baseline_target_mm_s
        rows.append(w)
        for model in ["adult", "mlp"]:
            v = w[w.model == model].sort_values("seed")
            assert np.array_equal(v.seed, np.arange(4401, 4421))
            lo, hi = np.quantile(
                v.delta_speed_mm_s.to_numpy()[idx].mean(1), [0.025, 0.975]
            )
            assert np.ptp(v.delta_target_mm_s.to_numpy()) < 1e-12
            summaries.append(
                dict(
                    model=model,
                    window_start_s=start,
                    window_end_s=start + WIDTH,
                    mean_delta_speed_mm_s=v.delta_speed_mm_s.mean(),
                    ci95_low=lo,
                    ci95_high=hi,
                    delta_target_mm_s=v.delta_target_mm_s.iloc[0],
                    n_seeds=20,
                )
            )
        p = w.pivot(
            index="seed", columns="model", values="delta_speed_mm_s"
        ).sort_index()
        delta = p.adult - p.mlp
        lo, hi = np.quantile(delta.to_numpy()[idx].mean(1), [0.025, 0.975])
        paired.append(
            dict(
                window_start_s=start,
                adult_minus_mlp_increment_mm_s=delta.mean(),
                ci95_low=lo,
                ci95_high=hi,
                n_pairs=20,
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(
        output / "ramp_increment_seeds.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(output / "ramp_increment_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(
        output / "ramp_increment_paired_summary.csv", index=False
    )

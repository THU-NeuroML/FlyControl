"""Fail-closed completeness audit and paired-seed summaries for the full run."""

from pathlib import Path
import json, hashlib, itertools, sys, time, platform
import numpy as np
import pandas as pd
from flycontrol.experiments.fig2.current.paths import DATA, OUTPUT as ROOT, REPO

HEIGHTS = [0.0, 0.2, 0.3, 0.45, 0.525]
METRICS = [
    "success",
    "crossed_within_corridor",
    "completion_time_with_failure_horizon_s",
    "max_lateral_mm",
    "physical_failure",
    "numerical_failure",
    "final_last_foot_x_mm",
    "final_body_x_mm",
    "final_up_z",
    "minimum_up_z",
    "final_stable_dwell_s",
    "residual_variation_per_observed_second",
    "residual_max_step_change_mean6_l1",
    "action_rms",
]


def exact_sign_flip(x):
    x = np.asarray(x)
    observed = abs(x.sum())
    if observed == 0:
        return 1.0
    count = 0
    total = 1 << len(x)
    for start in range(0, total, 32768):
        bits = (
            np.arange(start, min(start + 32768, total), dtype=np.uint32)[:, None]
            >> np.arange(len(x), dtype=np.uint32)
        ) & 1
        null = (2 * bits.astype(np.float64) - 1) @ x
        count += np.count_nonzero(np.abs(null) >= observed - 1e-12)
    return count / total


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def audit_and_summarize():
    manifest = json.loads((DATA / "evaluation_protocol.json").read_text())
    keys = set(
        itertools.product(
            manifest["training_seeds"],
            ["connectome", "mlp"],
            manifest["heights_mm"],
            manifest["phase_seeds"],
        )
    )
    rows = []
    actual = set()
    trace_checks = 0
    start = time.time()
    checkpoints = {
        (c["method"], c["seed"]): c["sha256"] for c in manifest["checkpoints"]
    }
    for p in sorted((DATA / "evaluation").glob("*/*/result.json")):
        r = json.loads(p.read_text())
        key = (r["training_seed"], r["method"], r["height"], r["phase_seed"])
        assert key not in actual, key
        actual.add(key)
        assert r["warmup_metadata"]["stable"] and not r["numerical_failure"], p
        assert r["checkpoint_sha256"] == checkpoints[(r["method"], r["training_seed"])]
        assert r["minimum_ground_edge_margin_mm"] > 1.0, (
            p,
            r["minimum_ground_edge_margin_mm"],
        )
        trace = p.parent / "trace.npz"
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == r["trace_sha256"]
        with np.load(trace, allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files), p
            t = z["time_s"]
            assert (
                np.all(np.diff(t) > 0)
                and abs(t[0]) < 1e-10
                and abs(t[-1] - r["duration_s"]) < 1e-8
            )
            assert (
                hashlib.sha256(z["qpos"][0].tobytes()).hexdigest()
                == r["initial"]["qpos_sha256"]
            )
            assert (
                hashlib.sha256(z["qvel"][0].tobytes()).hexdigest()
                == r["initial"]["qvel_sha256"]
            )
            assert (
                hashlib.sha256(z["observation"][0].tobytes()).hexdigest()
                == r["initial"]["observation_sha256"]
            )
            # Recompute the dwell success from each recorded 10 ms observation.
            cleared = (
                z["feet"][:, :, 0].min(axis=1) >= r["actual_obstacle_far_x_mm"]
            ) & (z["up"][:, 2] >= 0.8)
            dwell = 0.0
            ever = False
            for i in range(1, len(t)):
                dwell = dwell + float(t[i] - t[i - 1]) if cleared[i] else 0.0
                ever |= dwell >= 0.1 - 1e-9
            assert bool(ever) == r["success"], (p, ever, r["success"], dwell)
        trace_checks += 1
        rows.append(r)
    assert actual == keys, dict(
        expected=len(keys),
        found=len(actual),
        missing=sorted(keys - actual)[:10],
        unexpected=sorted(actual - keys)[:10],
    )
    df = pd.json_normalize(rows)
    for (h, phase), g in df.groupby(["height", "phase_seed"]):
        assert len(g) == 40
        for col in [
            "initial.qpos_sha256",
            "initial.qvel_sha256",
            "initial.observation_sha256",
        ]:
            assert g[col].nunique() == 1, (h, phase, col)
    df.to_csv(ROOT / "source_data/EPISODE_METRICS.csv", index=False)
    seed = (
        df.groupby(["training_seed", "height", "model"])[METRICS].mean().reset_index()
    )
    seed.to_csv(ROOT / "source_data/SEED_METRICS.csv", index=False)
    rng = np.random.default_rng(20261021)
    draw = rng.integers(0, 20, (50000, 20))
    summary = []
    for metric in METRICS:
        for h in HEIGHTS:
            p = (
                seed[seed.height == h]
                .pivot(index="training_seed", columns="model", values=metric)
                .sort_index()
            )
            assert len(p) == 20
            a = p.Adult.to_numpy()
            b = p.MLP.to_numpy()
            d = a - b
            bsa = a[draw].mean(axis=1)
            bsb = b[draw].mean(axis=1)
            bsd = d[draw].mean(axis=1)
            row = dict(
                metric=metric,
                height=h,
                n_seed_pairs=20,
                n_phases_per_model=64,
                adult_mean=a.mean(),
                mlp_mean=b.mean(),
                difference_adult_minus_mlp=d.mean(),
                adult_ci_low=np.quantile(bsa, 0.025),
                adult_ci_high=np.quantile(bsa, 0.975),
                mlp_ci_low=np.quantile(bsb, 0.025),
                mlp_ci_high=np.quantile(bsb, 0.975),
                difference_ci_low=np.quantile(bsd, 0.025),
                difference_ci_high=np.quantile(bsd, 0.975),
                exact_two_sided_p=exact_sign_flip(d),
                adult_higher_pairs=int((d > 0).sum()),
                equal_pairs=int((d == 0).sum()),
                adult_lower_pairs=int((d < 0).sum()),
            )
            summary.append(row)
    s = pd.DataFrame(summary)
    s["holm_p_within_metric_5_heights"] = np.nan
    for metric, idx in s.groupby("metric").groups.items():
        s.loc[idx, "holm_p_within_metric_5_heights"] = holm(
            s.loc[idx, "exact_two_sided_p"]
        )
    secondary = s.metric != "success"
    s["holm_p_all_secondary_comparisons"] = np.nan
    s.loc[secondary, "holm_p_all_secondary_comparisons"] = holm(
        s.loc[secondary, "exact_two_sided_p"]
    )
    s.to_csv(ROOT / "source_data/PAIRED_STATISTICS.csv", index=False)
    success = s[s.metric == "success"].copy()
    success.to_csv(ROOT / "source_data/SUCCESS_BY_HEIGHT.csv", index=False)
    # Increasing gap requires a direct within-pair contrast, not inspection of stars.
    p = seed.pivot(index="training_seed", columns=["height", "model"], values="success")
    contrast = (p[(0.525, "Adult")] - p[(0.525, "MLP")]) - (
        p[(0.3, "Adult")] - p[(0.3, "MLP")]
    )
    contrast_ci = np.quantile(contrast.to_numpy()[draw].mean(axis=1), [0.025, 0.975])
    audit = dict(
        complete=True,
        episodes=len(df),
        models=40,
        seed_pairs=20,
        heights=5,
        phases_per_model_height=64,
        all_trace_hashes_verified=trace_checks,
        success_recomputed_from_micro_frames=True,
        all_model_pairs_start_identically=True,
        all_initial_states_stable=True,
        minimum_ground_edge_margin_mm=float(df.minimum_ground_edge_margin_mm.min()),
        numerical_failures=int(df.numerical_failure.sum()),
        physical_failures=int(df.physical_failure.sum()),
        snapshot_validation=json.loads((DATA / "SNAPSHOT_VALIDATION.json").read_text()),
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        elapsed_s=time.time() - start,
    )
    (ROOT / "qa/COMPLETENESS_AUDIT.json").write_text(json.dumps(audit, indent=2))
    (ROOT / "qa/STATISTICAL_METHODS.json").write_text(
        json.dumps(
            dict(
                unit="paired training seed",
                n=20,
                within_model="64 paired phases per height",
                bootstrap="50,000 paired seed resamples, pointwise descriptive percentile 95% intervals, fixed phase panel conditional inference",
                test="exact two-sided sign-flip of 20 paired seed differences, all 2^20 signs; symmetry/exchangeability null",
                primary_family="all five height comparisons for ordinary success; Holm correction",
                secondary_family="all 65 secondary comparisons; Holm across the entire family, also within-metric columns supplied",
                selection="all requested seeds, phases and heights; evaluation protocol refined from two-seed pilot, results exploratory",
                training="frozen cold-trained 32,768-decision checkpoints, no retraining",
                initialization="physical steady start; recurrent policy state zero and four repeated current observations",
                gap_increase_0525_vs_030=dict(
                    mean=float(contrast.mean()),
                    ci_low=float(contrast_ci[0]),
                    ci_high=float(contrast_ci[1]),
                    p=exact_sign_flip(contrast),
                    interpretation="exploratory direct paired contrast",
                ),
            ),
            indent=2,
        )
    )
    print(
        success[
            [
                "height",
                "adult_mean",
                "mlp_mean",
                "difference_adult_minus_mlp",
                "holm_p_within_metric_5_heights",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return df, seed, s


if __name__ == "__main__":
    audit_and_summarize()

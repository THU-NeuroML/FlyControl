"""Seed-level summaries and an explicitly scoped paired comparison family."""

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import t

from flycontrol.experiments.shared.paths import ROOT

O = ROOT / "data/figure3/mechanisms/interface"
S = ROOT / "build/figure3_analysis/interface"
S.mkdir(parents=True, exist_ok=True)


def sign_sums(values):
    out = np.zeros(1)
    for v in values:
        out = np.concatenate([out + v, out - v])
    return out


def exact_signflip(values):
    """Count all 2**n sign assignments with meet-in-the-middle sorted sums."""
    v = np.asarray(values, float)
    observed = abs(v.sum())
    tol = 1e-12 * max(1.0, abs(v).sum())
    if observed <= tol:
        return 1.0
    left = sign_sums(v[: len(v) // 2])
    right = np.sort(sign_sums(v[len(v) // 2 :]))
    tail = np.sum(
        len(right) - np.searchsorted(right, observed - left - tol, side="left")
    )
    return min(1.0, float(2 * tail / (2 ** len(v))))


def ci(v):
    v = np.asarray(v, float)
    assert len(v) == 24 and np.isfinite(v).all()
    m = v.mean()
    delta = t.ppf(0.975, 23) * v.std(ddof=1) / np.sqrt(24)
    return float(m), float(m - delta), float(m + delta)


def main():
    # Independent brute-force check of exact counting, including ties.
    for v in [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6]), np.ones(8), np.zeros(4)]:
        z = np.array(list(itertools.product([-1, 1], repeat=len(v)))) @ v
        brute = np.mean(abs(z) >= abs(v.sum()) - 1e-12)
        assert abs(exact_signflip(v) - brute) < 1e-12
    data, audits = [], []
    for seed in range(4301, 4325):
        folder = O / f"models/seed{seed}"
        audit = json.loads((folder / "COMPLETE.json").read_text())
        assert (
            audit["matches_current_main_figure"]
            and audit["initial_readout_exactly_zero"]
        )
        data.append(pd.read_csv(folder / "seed_metrics.csv"))
        audits.append(audit)
    d = pd.concat(data, ignore_index=True)
    assert len(d) == 24 * 4 * 4 * 2 * 9
    assert d[d.C == 0].enrichment.isna().all()
    assert (d[d.C == 0].mean_squared_derivative == 0).all()
    assert np.isfinite(d[d.C == 1].enrichment).all()
    assert (
        d.groupby(["variant", "horizon_ms", "channel", "group"])
        .seed.nunique()
        .eq(24)
        .all()
    )
    d.to_csv(S / "exchange_seed_metrics.csv", index=False)
    summaries = []
    for keys, group in d.groupby(
        ["variant", "E", "C", "horizon_ms", "channel", "group"]
    ):
        base = dict(zip(["variant", "E", "C", "horizon_ms", "channel", "group"], keys))
        for metric in ["enrichment", "mean_squared_derivative"]:
            v = group[metric].to_numpy()
            if np.isnan(v).all():
                mean = low = high = sd = np.nan
            else:
                mean, low, high = ci(v)
                sd = float(v.std(ddof=1))
            summaries.append(
                dict(
                    **base,
                    metric=metric,
                    n_models=24,
                    mean=mean,
                    sd=sd,
                    ci95_low=low,
                    ci95_high=high,
                    undefined=bool(np.isnan(v).all()),
                )
            )
    pd.DataFrame(summaries).to_csv(S / "exchange_summary.csv", index=False)
    primary = d.query('horizon_ms == 100 and channel == "turn"')
    pairs, tests = [], []
    for group, rows in primary[primary.C == 1].groupby("group"):
        for metric in ["enrichment", "mean_squared_derivative"]:
            v = rows.pivot(index="seed", columns="E", values=metric).sort_index()
            assert v.shape == (24, 2) and (v > 0).all().all()
            effect = np.log2(v[1] / v[0]).to_numpy()
            m, low, high = ci(effect)
            dm, dl, dh = ci((v[1] - v[0]).to_numpy())
            tests.append(
                dict(
                    group=group,
                    metric=metric,
                    contrast="E1C1 / E0C1",
                    n=24,
                    mean_log2_ratio=m,
                    log2_ci95_low=low,
                    log2_ci95_high=high,
                    geometric_ratio=2**m,
                    ratio_ci95_low=2**low,
                    ratio_ci95_high=2**high,
                    raw_mean_difference=dm,
                    raw_ci95_low=dl,
                    raw_ci95_high=dh,
                    n_increased=int((effect > 0).sum()),
                    exact_p=exact_signflip(effect),
                )
            )
            for j, seed in enumerate(v.index):
                pairs.append(
                    dict(
                        seed=seed,
                        group=group,
                        metric=metric,
                        initial_E=v.loc[seed, 0],
                        trained_E=v.loc[seed, 1],
                        log2_ratio=effect[j],
                    )
                )
    tests = pd.DataFrame(tests)
    assert len(tests) == 18
    order = np.argsort(tests.exact_p.to_numpy())
    adjusted = np.empty(18)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(tests.exact_p.to_numpy()[order] * np.arange(18, 0, -1))
    )
    tests["holm18_p"] = adjusted
    tests.to_csv(S / "primary_paired_tests_all9classes.csv", index=False)
    pd.DataFrame(pairs).to_csv(S / "primary_seed_pairs.csv", index=False)
    summary = pd.DataFrame(summaries)
    an = summary.query('horizon_ms==100 and channel=="turn" and group=="true_AN"')
    an.to_csv(S / "AN_primary_summary.csv", index=False)
    qa = dict(
        models=24,
        variants_per_model=4,
        primary_windows_per_model=36,
        all_model_checkpoint_hashes_match_current_main_figure=True,
        all_24_initial_readouts_exactly_zero=True,
        undefined_relative_values_preserved=True,
        all_9_classes_retained=True,
        n_total_seed_metric_rows=len(d),
        test_family_size=18,
        independent_unit="training seed; one anatomy",
        exact_counting_bruteforce_checks_pass=True,
        new_training=False,
    )
    (S / "COMPLETE.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(an.to_string(index=False))
    print(tests[tests.group == "true_AN"].to_string(index=False))


if __name__ == "__main__":
    main()

"""Aggregate actual frozen-model probes; training seeds are independent units."""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from .paths import DATA, OUTPUT as R

S = R / "source_data"
from .behavior import exact_sign_flip, holm


def main():
    atlas = []
    probes = []
    matching = []
    receipts = []
    for seed in range(2011, 2031):
        p = DATA / "mechanisms/interventions" / str(seed)
        receipts.append(json.loads((p / "COMPLETE.json").read_text()))
        a = pd.read_csv(p / "atlas_40ms.csv")
        b = pd.read_csv(p / "interventions.csv")
        m = pd.read_csv(p / "matching.csv")
        assert (
            len(a) == 45
            and len(b) == 260
            and len(m) == 10
            and set(a.seed) == {seed}
            and set(b.seed) == {seed}
        )
        atlas.append(a)
        probes.append(b)
        matching.append(m)
    a = pd.concat(atlas, ignore_index=True)
    b = pd.concat(probes, ignore_index=True)
    m = pd.concat(matching, ignore_index=True)
    assert np.isfinite(a.enrichment).all() and np.isfinite(b.delta_power_mean).all()
    assert (
        b[b.dose == 0].delta_power_mean.eq(0).all()
        and b.baseline_power_mean.gt(0).all()
    )
    assert m.overlap_with_target.eq(0).all()
    a.to_csv(S / "EARLY_ATLAS_ALL.csv", index=False)
    b.to_csv(S / "CIRCUIT_ALL.csv", index=False)
    m.to_csv(S / "CIRCUIT_MATCHING.csv", index=False)
    boot = np.random.default_rng(20261022).integers(0, 20, (20000, 20))

    def summary(v):
        v = np.asarray(v)
        assert len(v) == 20 and np.isfinite(v).all()
        lo, hi = np.quantile(v[boot].mean(1), [0.025, 0.975])
        return dict(mean=float(v.mean()), ci_low=float(lo), ci_high=float(hi), n=20)

    rows = []
    for (group, height), q in a.groupby(["group", "height"]):
        rows.append(
            dict(
                group=group,
                height=height,
                n_cells=int(q.n_cells.iloc[0]),
                **summary(q.sort_values("seed").enrichment),
            )
        )
    pd.DataFrame(rows).to_csv(S / "EARLY_ATLAS_MEANS.csv", index=False)
    early = a.groupby(["seed", "group"]).enrichment.mean().reset_index()
    rows = []
    for group, q in early.groupby("group"):
        rows.append(dict(group=group, **summary(q.sort_values("seed").enrichment)))
    pd.DataFrame(rows).to_csv(S / "EARLY_ATLAS_POOLED.csv", index=False)
    masks = (
        b.groupby(["seed", "group", "kind", "replicate", "dose"])[
            ["delta_power_mean", "baseline_power_mean", "policy_kl"]
        ]
        .mean()
        .reset_index()
    )
    masks["output_change_percent"] = 100 * np.sqrt(
        masks.delta_power_mean / masks.baseline_power_mean
    )
    masks.to_csv(S / "CIRCUIT_MASK_SEEDS.csv", index=False)
    seeds = (
        masks.groupby(["seed", "group", "kind", "dose"])[
            ["output_change_percent", "policy_kl"]
        ]
        .mean()
        .reset_index()
    )
    seeds.to_csv(S / "CIRCUIT_SEEDS.csv", index=False)
    rows = []
    for (group, kind, dose), q in seeds.groupby(["group", "kind", "dose"]):
        rows.append(
            dict(
                group=group,
                kind=kind,
                dose=dose,
                **summary(q.sort_values("seed").output_change_percent),
            )
        )
    pd.DataFrame(rows).to_csv(S / "CIRCUIT_MEANS.csv", index=False)
    comparisons = []
    for group in ["mechanosensory", "olfactory"]:
        for dose in [0.25, 0.5, 1.0]:
            q = (
                seeds[(seeds.group == group) & (seeds.dose == dose)]
                .pivot(index="seed", columns="kind", values="output_change_percent")
                .sort_index()
            )
            delta = q.target - q["matched random"]
            comparisons.append(
                dict(
                    group=group,
                    dose=dose,
                    target_mean=float(q.target.mean()),
                    random_mean=float(q["matched random"].mean()),
                    **summary(delta),
                    p_exact=exact_sign_flip(delta),
                    n_positive=int((delta > 0).sum()),
                )
            )
    stats = pd.DataFrame(comparisons)
    stats["p_holm6"] = holm(stats.p_exact.to_numpy())
    stats.to_csv(S / "CIRCUIT_STATISTICS.csv", index=False)
    (R / "qa/CIRCUIT_COMPLETE.json").write_text(
        json.dumps(
            dict(
                complete=True,
                n_models=20,
                atlas_rows=len(a),
                intervention_rows=len(b),
                zero_dose_change_exactly_zero=True,
                matched_controls_zero_target_overlap=True,
                native_forward_max_error=max(r["forward_error"] for r in receipts),
                family="Two groups × three nonzero doses, Holm6; visual targeted responses descriptive without matched controls; original feasibility amendment retained",
            ),
            indent=2,
        )
    )
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

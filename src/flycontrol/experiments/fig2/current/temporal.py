import json
import numpy as np
import pandas as pd
from .paths import DATA, OUTPUT as R

S = R / "source_data"
Q = R / "qa"


def summarize(frame, keys, value, n, seed_key="seed"):
    boot = np.random.default_rng(20261023).integers(0, n, (20000, n))
    rows = []
    for key, q in frame.groupby(keys):
        v = q.sort_values(seed_key)[value].to_numpy()
        assert len(v) == n and np.isfinite(v).all()
        lo, hi = np.quantile(v[boot].mean(1), [0.025, 0.975])
        rows.append(
            dict(
                zip(keys, key if isinstance(key, tuple) else (key,)),
                mean=float(v.mean()),
                ci_low=float(lo),
                ci_high=float(hi),
                n=n,
            )
        )
    return pd.DataFrame(rows)


def dense():
    raw = []
    checks = []
    for seed in range(2011, 2031):
        p = DATA / "mechanisms/dense_neurons" / str(seed)
        checks.append(json.loads((p / "COMPLETE.json").read_text()))
        a = pd.read_csv(p / "sensitivity.csv")
        assert len(a) == 32 * 11 * 5 and set(a.seed) == {seed}
        raw.append(a)
    d = pd.concat(raw, ignore_index=True)
    d.to_csv(S / "DENSE_TIME_ALL.csv", index=False)
    pooled = (
        d.groupby(["seed", "horizon_ms", "group"])[
            ["population_enrichment", "partition_enrichment"]
        ]
        .mean()
        .reset_index()
    )
    pooled.to_csv(S / "DENSE_TIME_SEEDS.csv", index=False)
    stats = summarize(pooled, ["horizon_ms", "group"], "population_enrichment", 20)
    stats.to_csv(S / "DENSE_TIME_STATISTICS.csv", index=False)
    (Q / "DENSE_TIME_COMPLETE.json").write_text(
        json.dumps(
            dict(
                complete=True,
                n_models=20,
                rows=len(d),
                times_ms=list(range(10, 321, 10)),
                all_groups=11,
                actual_computation=True,
                interpolation=False,
                max_forward_error=max(c["forward_error"] for c in checks),
                max_old_timepoint_relative_error=max(
                    c["old_timepoint_max_relative_error"] for c in checks
                ),
                new_pointwise_tests="None: descriptive temporal sampling; original screened inferential results retained separately",
            ),
            indent=2,
        )
    )

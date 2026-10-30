"""Descriptive all24 analysis after an explicitly exploratory axis revision."""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from fig6.paths import ROOT

O = ROOT / "data/figure3/mechanisms/input_gain"
OUT = (
    Path(
        os.environ.get("FIGURE3_ANALYSIS_OUTPUT", str(ROOT / "build/figure3_analysis"))
    )
    / "input_gain"
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "qa").mkdir(exist_ok=True)
P = json.loads((O / "PROTOCOL.json").read_text())
MET = P["metrics"]
SEEDS = list(range(4301, 4325))
METHODS = ["mlp", "connectome"]

planes = []
summ = []
grid = []
qa = []
for m in METHODS:
    models = []
    for s in SEEDS:
        f = O / f"models/{m}_{s}"
        z = np.load(f / "surface.npz")
        v = z["metrics"]
        dose = z["dose"]
        assert v.shape == (21, 21, 3) and z["axes"].tolist() == ["Turn feedback", "CPG"]
        models.append(v)
        summ.append(dict(method=m, seed=s, **dict(zip(MET, v.mean((0, 1))))))
        for j, y in enumerate(dose):
            for i, x in enumerate(dose):
                grid.append(
                    dict(
                        method=m,
                        seed=s,
                        feedback_gain_change=x,
                        cpg_gain_change=y,
                        **dict(zip(MET, v[j, i])),
                    )
                )
        meta = json.loads((f / "SURFACE_COMPLETE.json").read_text())
        assert (
            meta["protocol_sha256"]
            == hashlib.sha256((O / "PROTOCOL.json").read_bytes()).hexdigest()
        )
        qa.append(meta)
    planes.append(models)
planes = np.array(planes)
np.savez_compressed(
    OUT / "cpg_input_gain_surfaces.npz",
    methods=METHODS,
    seeds=SEEDS,
    dose=dose,
    axes=["Turn feedback", "CPG"],
    metrics=planes,
    mean_all=planes.mean(1),
)
pd.DataFrame(grid).to_csv(OUT / "cpg_input_gain_grid.csv", index=False)
pd.DataFrame(summ).to_csv(OUT / "cpg_input_gain_seed_means.csv", index=False)
boot = np.random.default_rng(20261022).integers(0, 24, (20000, 24))
result = []
pairs = []
for k, metric in enumerate(MET):
    b = planes[0, :, :, :, k].mean((1, 2))
    a = planes[1, :, :, :, k].mean((1, 2))
    assert (a > 0).all() and (b > 0).all()
    logs = np.log2(a / b)
    lo, hi = np.quantile(np.exp2(logs[boot].mean(1)), [0.025, 0.975])
    ratio = a[boot].mean(1) / b[boot].mean(1)
    rlo, rhi = np.quantile(ratio, [0.025, 0.975])
    result.append(
        dict(
            metric=metric,
            n=24,
            adult_mean=float(a.mean()),
            mlp_mean=float(b.mean()),
            mean_reduction=float(1 - a.mean() / b.mean()),
            adult_lower=int((a < b).sum()),
            geometric_ratio=float(np.exp2(logs.mean())),
            geometric_ratio_ci_low=float(lo),
            geometric_ratio_ci_high=float(hi),
            mean_ratio_ci_low=float(rlo),
            mean_ratio_ci_high=float(rhi),
            role="Descriptive exploratory; no independent validation after revised axis choice; intervals conditional on chosen plane",
        )
    )
    for s, va, vb, lg in zip(SEEDS, a, b, logs):
        pairs.append(dict(seed=s, metric=metric, adult=va, mlp=vb, log2_ratio=lg))
pd.DataFrame(result).to_csv(OUT / "cpg_input_gain_summary.csv", index=False)
pd.DataFrame(pairs).to_csv(OUT / "cpg_input_gain_paired_effects.csv", index=False)
sections = []
axis_amplitude = []
for axis, values in [
    ("Feedback", planes[:, :, 10, :, 0]),
    ("CPG", planes[:, :, :, 10, 0]),
]:
    for mi, m in enumerate(METHODS):
        arr = values[mi]
        ci = np.quantile(arr[boot].mean(1), [0.025, 0.975], axis=0)
        for i, x in enumerate(dose):
            sections.append(
                dict(
                    axis=axis,
                    method=m,
                    dose=x,
                    mean=float(arr[:, i].mean()),
                    ci_low=float(ci[0, i]),
                    ci_high=float(ci[1, i]),
                )
            )
        axis_amplitude.append(
            dict(axis=axis, method=m, mean_endpoint_kl=float(arr[:, [0, 20]].mean()))
        )
pd.DataFrame(sections).to_csv(OUT / "cpg_input_gain_cross_sections.csv", index=False)
pd.DataFrame(axis_amplitude).to_csv(
    OUT / "cpg_input_gain_axis_amplitudes.csv", index=False
)
summary = dict(
    axes=["Turn feedback", "CPG"],
    n_seed_pairs=24,
    statistics=result,
    axis_amplitude=axis_amplitude,
    independent_validation=False,
    selection="Prior screening amplitude balance, exploratory",
)
(OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
audit = dict(
    passed=True,
    models=48,
    seed_pairs=24,
    computed_grid_cells=21168,
    all24_means_exact=True,
    max_native_error=max(q["native_error"] for q in qa),
    max_cpg_gain_equivalence_error=max(q["gain_equivalence_error"] for q in qa),
    max_origin_kl=float(planes[:, :, 10, 10, 0].max()),
    checkpoint_records=qa,
    no_new_training=True,
    axis_choice_exploratory=True,
)
(OUT / "qa/ANALYSIS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(pd.DataFrame(result).to_string(index=False))
print(pd.DataFrame(axis_amplitude).to_string(index=False))

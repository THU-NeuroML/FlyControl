"""Regenerate Figure 4g and its complete 45-comparison family from native inputs."""

import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from flycontrol.experiments.shared.paths import ROOT, read_json, sha256
from flycontrol.experiments.fig4.statistics import holm

DATA = ROOT / "data/figure4"
SEEDS = read_json(ROOT / "run.json")["seeds"]
METHODS = ["connectome", "mlp"]


def analyze(output):
    from flycontrol.experiments.fig4.mechanisms.study import checkpoint
    from body_control.policy import build_policy

    records = []
    projections = []
    blocks = dict(
        feet=(0, 18),
        contact=(18, 24),
        posture=(24, 28),
        cpg=(28, 46),
        speed=(46, 49),
        heading=(49, 51),
    )
    for seed in SEEDS:
        for method in METHODS:
            folder = DATA / "evaluation" / f"{method}_seed{seed}"
            for path in sorted(folder.glob("*_c*_p*.json")):
                d = read_json(path)
                assert d["checkpoint_sha256"] == sha256(checkpoint(method, seed))
                z = np.load(path.with_suffix(".npz"))
                assert d["trace_sha256"] == sha256(path.with_suffix(".npz"))
                row = dict(seed=seed, method=method)
                row.update(
                    {
                        k: v
                        for k, v in d["metrics"].items()
                        if isinstance(v, (float, int))
                    }
                )
                mask = z["elapsed_s"] >= 1
                error = (
                    z["filtered_body_forward_velocity_mm_s"][mask]
                    - z["target_applied_mm_s"][mask]
                )
                row.update(
                    signed_speed_bias=float(error.mean()),
                    speed_error_sd=float(error.std()),
                    mean_physical_drive=float(z["drive"][mask].mean()),
                    drive_sd=float(z["drive"][mask].std()),
                )
                records.append(row)
            initial = build_policy(method, seed, "cpu", 40).state_dict()
            final = torch.load(
                checkpoint(method, seed), map_location="cpu", weights_only=False
            )["policy_state"]
            key = (
                "actor.core.input_proj.weight"
                if method == "connectome"
                else "actor.net.0.weight"
            )
            for name, (lo, hi) in blocks.items():
                a = initial[key].numpy()[:, lo:hi]
                b = final[key].numpy()[:, lo:hi]
                projections.append(
                    dict(
                        seed=seed,
                        method=method,
                        block=name,
                        norm_final_over_initial=float(
                            np.linalg.norm(b) / np.linalg.norm(a)
                        ),
                    )
                )
    keep = [
        "speed_mae",
        "slope_error",
        "acceleration_slope_error",
        "deceleration_slope_error",
        "sine_gain",
        "sine_gain_error",
        "sine_lag_ms",
        "drive_tv_per_s",
        "heading_mean_deg",
        "signed_speed_bias",
        "speed_error_sd",
        "mean_physical_drive",
        "drive_sd",
        "failure_rate",
    ]
    assert len(records) == len(SEEDS) * 2 * 8
    summary = pd.DataFrame(records).groupby(["seed", "method"])[keep].mean()

    def combine(name):
        return pd.concat(
            [
                pd.read_csv(DATA / "probes" / f"{m}_{s}" / name)
                for s in SEEDS
                for m in METHODS
            ],
            ignore_index=True,
        )

    responses = combine("responses.csv")
    noise = combine("parameter_noise.csv")
    sensory = combine("sensory.csv")
    ns = (
        noise.groupby(["seed", "method", "radius"])[
            ["policy_kl", "unit_kl", "action_rms"]
        ]
        .mean()
        .reset_index()
    )
    ss = (
        sensory.groupby(["seed", "method", "block"])[
            ["policy_kl", "unit_kl", "action_rms"]
        ]
        .mean()
        .reset_index()
    )
    for inp in ["target", "measured"]:
        d = responses[responses.input == inp].set_index(["seed", "method"])
        summary[inp + "_response"] = d.drive_response
    d = ns[ns.radius == 0.1].set_index(["seed", "method"])
    summary["parameter_kl"] = d.policy_kl
    summary["parameter_action_rms"] = d.action_rms
    summary["unit_kl"] = d.unit_kl
    proj = pd.DataFrame(projections)
    d = proj[proj.block == "speed"].set_index(["seed", "method"])
    summary["speed_projection_gain"] = d.norm_final_over_initial
    for block in ss.block.unique():
        summary["sensitivity_" + block] = (
            ss[ss.block == block].set_index(["seed", "method"]).action_rms
        )
    summary.reset_index().to_csv(
        output / "scatter_MODEL_MECHANISM_SUMMARY.csv", index=False
    )
    # Within-method associations; no pooling architecture as a hidden confound.
    # Exact model performance has already been observed, so all these are exploratory.
    predictors = [
        "target_response",
        "measured_response",
        "parameter_kl",
        "parameter_action_rms",
        "speed_projection_gain",
    ]
    outcomes = ["speed_mae", "slope_error", "sine_gain_error"]
    corr = []
    rng = np.random.default_rng(20261010)
    perms = np.stack([rng.permutation(20) for _ in range(19999)])

    def permutation_spearman(x, y):
        xr = rankdata(x)
        yr = rankdata(y)
        xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
        yr = (yr - yr.mean()) / np.linalg.norm(yr - yr.mean())
        r = float(xr @ yr)
        p = float((1 + (np.abs(yr[perms] @ xr) >= abs(r) - 1e-12).sum()) / 20000)
        return r, p

    for method in METHODS:
        part = summary.xs(method, level="method").loc[SEEDS]
        for pred in predictors:
            for outcome in outcomes:
                rho, p = permutation_spearman(part[pred], part[outcome])
                corr.append(
                    dict(
                        cohort=method,
                        predictor=pred,
                        outcome=outcome,
                        n_seeds=20,
                        rho=rho,
                        p_permutation=p,
                    )
                )
    # Paired architecture differences also avoid a between-architecture mean confound.
    delta = (
        summary.xs("connectome", level="method").loc[SEEDS]
        - summary.xs("mlp", level="method").loc[SEEDS]
    )
    for pred in predictors:
        for outcome in outcomes:
            rho, p = permutation_spearman(delta[pred], delta[outcome])
            corr.append(
                dict(
                    cohort="adult_minus_mlp_pairs",
                    predictor=pred,
                    outcome=outcome,
                    n_seeds=20,
                    rho=rho,
                    p_permutation=p,
                )
            )
    corr = pd.DataFrame(corr)
    corr["p_holm_45"] = holm(corr.p_permutation)
    corr.to_csv(output / "scatter_performance_associations.csv", index=False)

"""Original episode metrics and complete registered Holm families, with relocated inputs."""

import json, hashlib, itertools
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from fig6.paths import ROOT
from .paths import DATA

METHODS = ["connectome", "mlp", "er"]
SECONDARY = {
    "angle": [
        "command_variation",
        "success",
        "failure",
        "speed",
        "relative_command_roughness",
        "late_heading_upright_fraction",
        "last_window_error",
        "worst_direction_error",
    ],
    "push": [
        "command_variation",
        "terminal_error",
        "failure",
        "speed",
        "relative_command_roughness",
        "late_heading_upright_fraction",
        "last_window_error",
        "recovery_auc",
    ],
}


def exact_p(diff):
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.sum())
    tol = 1e-12 * max(1.0, abs(diff).sum())
    if observed <= tol:
        return 1.0

    def sums(d):
        a = np.array([0.0])
        for v in d:
            a = np.r_[a + v, a - v]
        return a

    split = len(diff) // 2
    left = sums(diff[:split])
    right = np.sort(sums(diff[split:]))
    low = np.searchsorted(right, -observed + tol - left, side="right")
    high = len(right) - np.searchsorted(right, observed - tol - left, side="left")
    return float(np.sum(low + high, dtype=np.int64) / (len(left) * len(right)))


def self_check():
    for n in [3, 6, 12]:
        d = np.random.default_rng(771 + n).normal(size=n)
        sums = np.array(list(itertools.product([-1.0, 1.0], repeat=n))) @ d
        expected = np.mean(abs(sums) >= abs(d.sum()) - 1e-10)
        assert exact_p(d) == expected, (n, exact_p(d), expected)
    assert exact_p(np.zeros(24)) == 1
    assert exact_p(np.ones(24)) == 2 ** (-23)


def metrics(path):
    payload = path.read_bytes()
    r = json.loads(payload)
    tr = r["micro_trajectory"]
    times = np.array([x["elapsed_s"] for x in tr])
    grid = np.arange(1, 201) * 0.01
    valid = len(tr)
    if not np.isclose(times[-1], grid[valid - 1], atol=1e-9):
        valid -= 1
    error = np.full(200, 180.0)
    up = np.zeros(200)
    speed = np.zeros(200)
    assert np.allclose(times[:valid], grid[:valid], atol=1e-9)
    error[:valid] = np.rad2deg(np.abs([x["heading_error_rad"] for x in tr[:valid]]))
    up[:valid] = [x["up_z"] for x in tr[:valid]]
    speed[:valid] = [x["body_forward_speed_mm_s"] for x in tr[:valid]]
    macro = [x for x in r["trajectory"] if 0.4 + 1e-9 < x["elapsed_s"] <= 2 + 1e-9]
    rough = np.nan
    if len(macro) == 40 and not r["final"]["physical_failure"]:
        actions = np.array([x["action"] for x in macro])
        centered = actions - actions.mean(axis=0)
        den = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        rough = (
            float(np.sqrt(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=1))) / den)
            if den > 1e-12
            else 0.0
        )
    return dict(
        family=r["family"],
        seed=r["training_seed"] or -1,
        method=r["method"],
        phase=r["phase_seed"],
        angle=r["case"].get("angle_deg", 0.0),
        force=r["case"].get("force", 0.0),
        side=r["case"].get("side", 0.0),
        terminal_error=r["primary_angle_error_deg"],
        success=float(r["final"]["success"]),
        failure=float(r["final"]["physical_failure"]),
        command_variation=r["action_total_variation"],
        speed=float(speed.mean()),
        relative_command_roughness=rough,
        late_heading_upright_fraction=float(
            ((error[-50:] <= 10) & (up[-50:] >= 0.8)).mean()
        ),
        last_window_error=float(error[-50:].mean()),
        recovery_auc=r["primary_push_auc_rad_s"],
        source=str(path),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        initial_state=r["initial"]["initial_state_sha256"],
        initial_obs=r["initial"]["initial_observation_sha256"],
    )


def statistic(sm, family, metric, control, role):
    p = (
        sm[(sm.family == family) & sm.method.isin(["connectome", control])]
        .pivot(index="seed", columns="method", values=metric)
        .sort_index()
    )
    a = p["connectome"].to_numpy()
    b = p[control].to_numpy()
    diff = a - b
    n = len(diff)
    finite = np.isfinite(diff).all()
    half = (
        float(student_t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n))
        if finite
        else np.nan
    )
    return dict(
        family=family,
        metric=metric,
        control=control,
        role=role,
        n=n,
        adult=float(np.mean(a)),
        control_mean=float(np.mean(b)),
        adult_sd=float(np.std(a, ddof=1)),
        control_sd=float(np.std(b, ddof=1)),
        difference=float(np.mean(diff)),
        ci_low=float(np.mean(diff) - half),
        ci_high=float(np.mean(diff) + half),
        p_raw=exact_p(diff) if finite else 1.0,
        assessable=bool(finite),
        lower_seeds=int((diff < 0).sum()),
        higher_seeds=int((diff > 0).sum()),
        tied_seeds=int((diff == 0).sum()),
    )


def holm(rows):
    indices = np.argsort([r["p_raw"] for r in rows])
    last = 0.0
    for j, i in enumerate(indices):
        last = max(last, min(1.0, rows[i]["p_raw"] * (len(rows) - j)))
        rows[i]["p_holm"] = last


def analyze(root):
    self_check()
    plan = json.loads((ROOT / "configs/figure3/study_plan.json").read_text())
    rows = []
    for family in plan["families"]:
        for seed in plan["seeds"]:
            for method in METHODS:
                steps=json.loads((ROOT/'protocol.json').read_text())['ppo']['total_timesteps']
                folder = ROOT/f'evaluation/{method}_seed{seed}' if family=='angle' else ROOT/f'registered/evaluation/final/push_aligned_{method}_seed{seed}_step{steps}'
                episodes = sorted(folder.glob("phase*.json"))
                expected = len(plan["phase_seeds"]) * (6 if family == "angle" else 3)
                assert len(episodes) == expected, (folder, len(episodes), expected)
                rows.extend(metrics(p) for p in episodes)
    df = pd.DataFrame(rows)
    assert not df.duplicated(
        ["family", "seed", "method", "phase", "angle", "force", "side"]
    ).any()
    keys = ["family", "seed", "phase", "angle", "force", "side"]
    for column in ["initial_state", "initial_obs"]:
        p = df.pivot(index=keys, columns="method", values=column)
        assert p.nunique(axis=1).eq(1).all()
    out = root
    out.mkdir(exist_ok=True)
    df.to_csv(out / "episodes.csv", index=False)
    main = df[(df.family == "angle") | ((df.family == "push") & (df.force == 5))]
    metric_columns = [
        "terminal_error",
        "success",
        "failure",
        "command_variation",
        "speed",
        "relative_command_roughness",
        "late_heading_upright_fraction",
        "last_window_error",
        "recovery_auc",
    ]
    sm = main.groupby(["family", "seed", "method"])[metric_columns].mean().reset_index()
    # Do not silently drop failed/truncated episodes from the roughness endpoint.
    invalid = main.groupby(
        ["family", "seed", "method"]
    ).relative_command_roughness.apply(lambda x: x.isna().any())
    for idx, row in sm.iterrows():
        if invalid.loc[(row.family, row.seed, row.method)]:
            sm.loc[idx, "relative_command_roughness"] = np.nan
    if "angle" in plan["families"]:
        t = main[main.family == "angle"].assign(direction=lambda d: np.sign(d.angle))
        worst = (
            t.groupby(["family", "seed", "method", "direction"])
            .terminal_error.mean()
            .groupby(["family", "seed", "method"])
            .max()
            .rename("worst_direction_error")
            .reset_index()
        )
        sm = sm.merge(
            worst, on=["family", "seed", "method"], how="left", validate="one_to_one"
        )
    sm.to_csv(out / "seed_means.csv", index=False)
    primary = [
        statistic(sm, f, plan["primary"][f], c, "primary")
        for f in plan["families"]
        for c in ["mlp", "er"]
    ]
    secondary = [
        statistic(sm, f, m, c, "secondary")
        for f in plan["families"]
        for m in SECONDARY[f]
        for c in ["mlp", "er"]
    ]
    holm(primary)
    holm(secondary)
    stats = pd.DataFrame(primary + secondary)
    stats.to_csv(out / "statistics.csv", index=False)
    flat = (
        df[(df.family == "push") & (df.force == 0)]
        .groupby(["family", "seed", "method"])[metric_columns]
        .mean()
        .reset_index()
    )
    flat.to_csv(out / "flat_reference_seed_means.csv", index=False)
    (out / "statistics.json").write_text(
        stats.to_json(orient="records", indent=2) + "\n"
    )
    lines = [
        "# Three-way complete cohort review",
        "",
        f"Role: {plan['role']}. Independent training seeds: {len(plan['seeds'])}. No seed or physical episode excluded.",
        "",
        "Primary comparisons (Adult versus MLP and ER):",
        "",
        primary_table(stats),
        "",
        "Full secondary family, all individual seed values, failure rates and physical speeds are in statistics.csv / seed_means.csv.",
        "Pointwise paired-t confidence intervals are descriptive; significance uses exact two-sided paired sign flips and the specified complete Holm families.",
        "ER is a fixed-N/E uniform random control with matched rate scaling. Its degree and edge-weight distributions are not matched; no topology-only causal claim.",
        "Any undefined roughness from a failed/short episode invalidates that seed endpoint; its test is marked unassessable with P=1 in the unchanged correction family.",
    ]
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n")
    summary = dict(
        complete=True,
        training_seeds=len(plan["seeds"]),
        episodes=len(df),
        all_initial_states_paired=True,
        role=plan["role"],
        primary_tests=len(primary),
        secondary_tests=len(secondary),
    )
    (out / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def primary_table(stats):
    return stats[stats.role == "primary"][
        [
            "family",
            "metric",
            "control",
            "n",
            "adult",
            "control_mean",
            "difference",
            "p_raw",
            "p_holm",
        ]
    ].to_string(index=False)

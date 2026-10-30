"""Episode -> training-seed means -> paired-seed bootstrap, with full-grid validation."""

import csv
import math
from collections import defaultdict
import numpy as np
from fig4.statistics import signflip, holm


def summarize_condition_means(cells, config):
    """Unrounded D cells and their C−M differences, in displayed row order."""
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["body"], cell["condition"], cell["method"])].append(cell)
    means = []
    for body in config["bodies"]:
        seeds = set(config["training_seeds"][body])
        for condition in config["conditions"]:
            row = dict(body=body, condition=condition)
            for method in ("mlp", "connectome"):
                values = groups[(body, condition, method)]
                if (
                    len(values) != len(seeds)
                    or {v["training_seed"] for v in values} != seeds
                ):
                    raise ValueError("D/E condition means require every training seed")
                row[method] = float(np.mean([v["task_score"] for v in values]))
            row["difference_C_minus_M"] = row["connectome"] - row["mlp"]
            means.append(row)
    return means


def summarize(rows, config):
    # The display may summarize all conditions while retaining the original
    # four-condition endpoint in the frozen experiment protocol.
    effect_conditions = config.get("effect_conditions", config["primary_conditions"])
    if (
        not effect_conditions
        or len(set(effect_conditions)) != len(effect_conditions)
        or not set(effect_conditions) <= set(config["conditions"])
    ):
        raise ValueError("Invalid Figure 6 effect conditions")
    expected = {
        (b, s, m, e, c)
        for b in config["bodies"]
        for s in config["training_seeds"][b]
        for m in config["methods"]
        for e in config["evaluation_seeds"]
        for c in config["conditions"]
    }
    identities = [
        (
            r["body"],
            r["training_seed"],
            r["method"],
            r["evaluation_seed"],
            r["condition"],
        )
        for r in rows
    ]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError(
            "Episode grid is incomplete, duplicated, or contains unexpected observations"
        )
    groups = defaultdict(list)
    for r in rows:
        budget = 524288 if r["body"] in ("cassie", "go1") else 131072
        if (
            r["training_steps"] != budget
            or r["teacher_used"]
            or r["external_support_used"]
        ):
            raise ValueError("Episode violates the fixed-budget, reward-only protocol")
        if r["evaluation_device"] != "cpu":
            raise ValueError("Paper comparisons require CPU evaluation")
        if not all(
            math.isfinite(r[k])
            for k in ("duration_s", "horizon_s", "velocity_mps", "lateral_max_m")
        ):
            raise ValueError("Non-finite episode metrics")
        if not 0 <= r["duration_s"] <= r["horizon_s"] + 1e-6 or r["horizon_s"] <= 0:
            raise ValueError("Invalid episode duration")
        score = (
            min(1.0, r["duration_s"] / r["horizon_s"])
            if r["body"] == "cassie"
            else float(r["success"])
        )
        groups[(r["body"], r["training_seed"], r["method"], r["condition"])].append(
            dict(r, task_score=score)
        )
    cells, effects, cohorts = [], [], {}
    for body in config["bodies"]:
        body_cells = []
        for seed in config["training_seeds"][body]:
            for method in config["methods"]:
                for condition in config["conditions"]:
                    rr = groups[(body, seed, method, condition)]
                    cell = dict(
                        body=body,
                        training_seed=seed,
                        method=method,
                        condition=condition,
                        n_episodes=len(rr),
                    )
                    cell.update(
                        {
                            k: float(np.mean([r[k] for r in rr]))
                            for k in (
                                "task_score",
                                "success",
                                "duration_s",
                                "velocity_mps",
                                "lateral_max_m",
                            )
                        }
                    )
                    body_cells.append(cell)
        differences, method_scores = [], {m: [] for m in config["methods"]}
        for seed in config["training_seeds"][body]:
            scores = {
                m: float(
                    np.mean(
                        [
                            r["task_score"]
                            for r in body_cells
                            if r["training_seed"] == seed
                            and r["method"] == m
                            and r["condition"] in effect_conditions
                        ]
                    )
                )
                for m in config["methods"]
            }
            differences.append(scores["connectome"] - scores["mlp"])
            for method, score in scores.items():
                method_scores[method].append(score)
        n = len(differences)
        bs = (
            np.random.default_rng(config["bootstrap_seed"])
            .choice(differences, size=(config["bootstrap_resamples"], n), replace=True)
            .mean(axis=1)
        )
        lo, hi = np.quantile(bs, [0.025, 0.975])
        effects.append(
            dict(
                body=body,
                n_training_pairs=n,
                mean_delta=float(np.mean(differences)),
                ci95_low=float(lo),
                ci95_high=float(hi),
                paired_seed_differences=differences,
                training_seeds=list(config["training_seeds"][body]),
                effect_conditions=list(effect_conditions),
                mlp_mean=float(np.mean(method_scores["mlp"])),
                connectome_mean=float(np.mean(method_scores["connectome"])),
                p_raw=signflip(differences),
            )
        )
        cohorts[body] = list(config["training_seeds"][body])
        cells.extend(body_cells)
    condition_means = summarize_condition_means(cells, config)
    for effect, adjusted in zip(effects, holm(np.array([r["p_raw"] for r in effects]))):
        by_condition = {
            r["condition"]: r for r in condition_means if r["body"] == effect["body"]
        }
        differences = [
            by_condition[c]["difference_C_minus_M"] for c in effect_conditions
        ]
        mean_from_d = float(np.mean(differences))
        if not math.isclose(
            mean_from_d, effect["mean_delta"], abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("D condition means and E seed-based inference disagree")
        # Displayed points come directly from D. Inference still resamples whole
        # training-seed pairs, preserving their repeated-condition dependence.
        effect["condition_differences"] = differences
        effect["mean_delta"] = mean_from_d
        effect["p_holm_four_bodies"] = float(adjusted)
    return cells, effects, cohorts

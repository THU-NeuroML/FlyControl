"""Pool original per-frame energies before normalization, exactly as in panel f."""

import json
import numpy as np
import pandas as pd
from scipy.stats import t
from fig6.paths import sha256 as sha

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
SEEDS = np.arange(4301, 4325)


def extract(root, compact, output):
    inv = pd.read_csv(root / "source_data/analysis_inventory.csv")
    ids = [
        int(inv.index[(inv.level == "input_class") & (inv.label == g)][0])
        for g in ORDER
    ]
    partition = int(
        inv.index[(inv.level == "partition") & (inv.partition == "input")][0]
    )
    counts = inv.loc[ids, "n_cells"].to_numpy(int)
    assert counts.sum() == 18756
    energies, provenance = [], []
    for seed in SEEDS:
        path = root / f"models/seed{seed}/activity.npz"
        with np.load(path) as z:
            assert np.array_equal(z["group_id"], inv.group_id.to_numpy(str))
            assert np.array_equal(z["phases"], [790000, 790101, 790102])
            assert np.array_equal(z["angles"], [-60, 0, 60])
            assert np.all(
                z["lengths"] == 50
            ), "No truncated trials may be silently dropped."
            assert z["metrics"][3] == "delta_mean_square"
            time = z["time_s"]
            keep = (time >= 0) & (time < 1.7)
            assert keep.sum() == 42
            v = z["values"]
            assert np.all(
                v[:, 1, :, :, 3] == 0
            ), "Matched zero-control difference must be zero."
            energy = v[:, [0, 2]][:, :, keep][:, :, :, ids + [partition], 3]
            assert energy.shape == (3, 2, 42, 10)
            assert np.isfinite(energy).all() and (energy >= 0).all()
            energies.append(energy)
        complete = json.loads((path.parent / "COMPLETE.json").read_text())
        provenance.append(
            {
                "seed": int(seed),
                "raw_path": str(path.resolve()),
                "raw_sha256": sha(path),
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "original_replay_validation": complete,
            }
        )
    np.savez_compressed(
        compact,
        energy=np.stack(energies),
        seeds=SEEDS,
        phases=[790000, 790101, 790102],
        angles=[-60, 60],
        time_s=time[keep],
        groups=ORDER + ["input_partition"],
        n_cells=np.r_[counts, counts.sum()],
        axes=["seed", "phase", "angle", "time", "group"],
    )
    (output / "activity_full_turn_provenance.json").write_text(
        json.dumps(
            {
                "raw_metric": "Per-neuron mean squared tanh(hidden)-state difference from matched 0-degree trajectory",
                "compact_sha256": sha(compact),
                "records": provenance,
            },
            indent=2,
        )
        + "\n"
    )


def summarize(compact, output):
    z = np.load(compact)
    energy = z["energy"].astype(np.float64)
    counts = z["n_cells"][:9]
    assert energy.shape == (24, 3, 2, 42, 10)
    assert np.array_equal(z["seeds"], SEEDS)
    assert np.array_equal(z["groups"][:9], ORDER)
    assert np.isfinite(energy).all() and np.all(energy >= 0)
    # Independently reduced partition validates the exhaustive nine-class split.
    weighted = np.average(energy[..., :9], axis=-1, weights=counts)
    np.testing.assert_allclose(weighted, energy[..., 9], rtol=2e-5, atol=1e-9)
    # Pool energies, NOT precomputed enrichment ratios, with all 42 times equal.
    per_direction = energy[..., :9].mean(axis=(1, 3))  # seed, angle, class
    pooled = per_direction.mean(axis=1)  # seed, class
    denominator = np.average(pooled, axis=-1, weights=counts)
    ratios = pooled / denominator[:, None]
    np.testing.assert_allclose(np.average(ratios, axis=1, weights=counts), 1)
    direction_denominator = np.average(per_direction, axis=-1, weights=counts)
    direction_ratios = per_direction / direction_denominator[..., None]
    records, directional = [], []
    for s, seed in enumerate(SEEDS):
        for g, group in enumerate(ORDER):
            records.append(
                dict(
                    seed=int(seed),
                    group=group,
                    n_cells=int(counts[g]),
                    mean_square_change=pooled[s, g],
                    partition_mean_square_change=denominator[s],
                    enrichment=ratios[s, g],
                )
            )
            for a, angle in enumerate(z["angles"]):
                directional.append(
                    dict(
                        seed=int(seed),
                        angle=int(angle),
                        group=group,
                        mean_square_change=per_direction[s, a, g],
                        partition_mean_square_change=direction_denominator[s, a],
                        enrichment=direction_ratios[s, a, g],
                    )
                )
    pd.DataFrame(records).to_csv(
        output / "full_turn_activity_seed_values.csv", index=False
    )
    pd.DataFrame(directional).to_csv(
        output / "full_turn_activity_direction_seed_values.csv", index=False
    )
    mean = ratios.mean(axis=0)
    sd = ratios.std(axis=0, ddof=1)
    half = t.ppf(0.975, 23) * sd / np.sqrt(24)
    summary = pd.DataFrame(
        dict(
            group=ORDER,
            n_cells=counts,
            n_seeds=24,
            mean=mean,
            sd=sd,
            ci95_low=mean - half,
            ci95_high=mean + half,
            seeds_above_partition=(ratios > 1).sum(axis=0),
            seeds_highest_class=(ratios.argmax(axis=1)[:, None] == np.arange(9)).sum(
                axis=0
            ),
        )
    )
    summary.to_csv(output / "full_turn_activity_summary.csv", index=False)
    audit = dict(
        n_seeds=24,
        n_turning_trials=144,
        n_zero_controls=72,
        phases=z["phases"].tolist(),
        angles=z["angles"].tolist(),
        time_samples=42,
        sampled_cue_relative_range_s=z["time_s"][[0, -1]].tolist(),
        time_window_s=[0, 1.7],
        no_seed_or_trial_exclusions=True,
        partition_max_absolute_reconstruction_error=float(
            np.max(abs(weighted - energy[..., 9]))
        ),
        partition_max_relative_reconstruction_error=float(
            np.max(abs(weighted - energy[..., 9]) / np.maximum(energy[..., 9], 1e-30))
        ),
        compact_sha256=sha(compact),
        aggregation="Per-seed mean energy across phases, directions and all 42 time frames before count-weighted partition normalization; equal weight per seed.",
        confidence_interval="Pointwise Student t(23), two-sided 95%; seed is the independent unit.",
        AN=summary[summary.group == "true_AN"].iloc[0].to_dict(),
    )
    (output / "FULL_TURN_ACTIVITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(summary.to_string(index=False))

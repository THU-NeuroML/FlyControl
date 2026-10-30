# Ant flat-ground capacity control

`ant-flat` reproduces the updated Figure 5b–d and Supplementary Figure S7. It uses the Ant engine but has separate configurations and checkpoints from `maze`. Maze architecture, adaptation and navigation results are unchanged.

## Two reproducibility routes

```bash
# Reanalyze the paper observations; no model training or external workspace needed.
flycontrol ant-reference --output runs/ant-reference

# Fresh training and simulation; 20 seeds for each method.
python scripts/05_ant_flat.py --run-dir runs/ant-flat --device cuda:0

# Short software check: seed 0, 128 steps per training stage, one reset per condition.
python scripts/05_ant_flat.py --run-dir runs/ant-flat-smoke --smoke --device cpu
```

The [notebook](../notebooks/05_ant_flat.ipynb) runs the short workflow and the archived-data analysis in separate directories. Both sets of outputs are labelled. Full-budget training is a documented command, not part of Run All. Seeds train sequentially; `--workers` does not launch concurrent Ant training. Full training is computationally substantial.

The archived analysis reproduces the manuscript's recorded numerical results. Fresh training tests the same protocol; stochastic training and hardware-dependent arithmetic can produce different learned policies. The published comparison reuses the original Adult policies and newly trains the matched MLP. The public full workflow trains both methods anew and does not download paper checkpoints.

## Fixed protocol

| Setting | Value |
|---|---|
| Optimization seeds | 0–19, both methods; no unsuccessful-policy exclusions |
| MLP | 1378 → 1378 ReLU hidden layers |
| Mean-action actor parameters, base (105 inputs) | Adult 1,989,016; MLP 2,057,362 |
| Mean-action actor parameters, final (109 inputs) | Adult 2,064,040; MLP 2,062,874; difference 0.0565% |
| Parameter-count scope | Trainable mean-action actor; excludes critic, eight log-SD values and fixed graph buffers |
| Base PPO | Requested 3,000,000 steps; 2,998,272 complete rollout steps |
| Flat refinement PPO | Requested 200,000 steps; 198,656 complete rollout steps |
| PPO minibatches | Four per epoch, both methods and both stages |
| Refinement initialization | Base best checkpoint; append four zero observation features and transfer normalization |
| Evaluation | Deterministic actions, scale 0.9; native Ant-v5 termination; 1,000-step (50-s) horizon |
| Flat resets | 9010000–9010099 per training seed, friction 1.0 |
| S7 friction | 0.5, 0.75, 1.25, 1.5; 25 resets each, starting at 9030000 + 10000 × condition index |

The capacity match applies to the final 109-input actor. The base MLP is 3.44% larger. Both capacity and PPO minibatches changed relative to the earlier small MLP; this is not an isolated architecture-only intervention. Checkpoint selection uses the unchanged training-only rule. Evaluation states were reused, not a new blinded holdout.

Training uses the existing stabilization settings during refinement; evaluation disables stabilization termination. **Functional failure** means native termination or non-terminated completion with final torso tilt greater than 90°. An ordinary timeout is not itself a failure. Signed path efficiency is forward displacement divided by planar path length. Every episode, including early failures and backward motion, enters the seed average.

## Statistics and plots

Each of the 20 paired training seeds is one statistical unit; its 100 reset observations are averaged first. Three new endpoints use two-sided exact sign-flip tests over all 2²⁰ assignments. Their Holm correction retains the original 26 hypotheses and adds the three new comparisons (29 total). The original maze displays retain their original 26-test correction.

The original 26 raw P values are bundled and explicitly labelled as archived in `analysis/family_29.csv`. This preserves the manuscript's extension of the original family; it is not a newly simulated maze cohort. To recalculate the original family from newly trained maze policies, run the separate `maze` workflow. Do not substitute a three-test correction for the reported 29-test correction.

| Endpoint | Adult mean | Matched MLP mean | Raw P | Holm P (29) |
|---|---:|---:|---:|---:|
| Forward velocity (m/s) | 1.936390 | 1.235719 | 0.009851 | 0.059109 |
| Signed path efficiency | 0.884566 | 0.793281 | 0.006376 | 0.044634 |
| Functional failure | 0.386000 | 0.298500 | 0.054583 | 0.218330 |

Confidence intervals use 50,000 paired-seed bootstrap resamples (RNG seed 20261030). The efficiency CDF is evaluated at every observed threshold, retaining negative values. Displacement plots average all resets within each seed, hold terminal positions to 50 s and show ±1 SD across seed means. S7 uses all 20 pairs × four coefficients × 25 resets per method; its intervals are pointwise 95% bootstrap intervals, with no significance tests. Smoke runs have no inferential intervals or P values.

`figures/paper/` contains individual PDF/SVG/PNG quantitative panels and source-table hashes. These are reproducible data displays, not the manuscript's composite artwork. Figure 5b's earlier qualitative demo and Supplementary Video 6 remain illustrations from the earlier small-MLP cohort; they are not replay evidence for the new matched model. The failure plot has no `NS` annotation.

## Included source observations

[`configs/reference/ant_capacity_20261030.json`](../configs/reference/ant_capacity_20261030.json) contains:

- 4,000 native flat episodes and the corresponding 4,000 motion endpoint records, with method, optimization seed and reset seed.
- 4,000 S7 native episodes, with all four friction conditions and failure flags.
- Per-training-seed forward and absolute lateral displacement curves, each averaged over 100 resets at 1,001 time points. These are derived from the recorded trajectories; post-termination positions are held as above.
- All 26 original raw P values, evaluation settings and SHA-256 hashes of the original source files.

Workspace paths, model weights and unrelated motion diagnostics are omitted. The archive is included in source and wheel distributions. The analysis exports readable CSV files, verifies complete seed/reset grids, checks endpoint definitions, and records hashes. Trained-model runs additionally verify checkpoint and evaluation provenance. Parameter counts and completed training budgets are checked for each freshly trained stage.

See [validation](validation.md) for the checks actually executed on this release.

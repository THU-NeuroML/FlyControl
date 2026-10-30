# Validation record

Date: 2026-10-30. These checks validate software execution and numerical contracts. They do not establish the manuscript's scientific conclusions.

## Ant capacity-control update (2026-10-30)

The current source/configuration SHA-256 is `ec7a0a8e27306db0ced8bb688b0ef75d579aeb5120369af2ba3f31cac33f6374`. The checks below cover the new flat-ground workflow; the earlier release records follow separately.

| Check | Result |
|---|---|
| Automated tests | 37 passed, including the archived 20-pair numerical regression, complete reset grids, failure definitions and separation of flat/maze configurations |
| Published flat statistics | Recomputed from all 4,000 native and 4,000 motion endpoint records; means and Holm-29 P values reproduce the manuscript to numerical precision |
| Published curves and S7 | Seed displacement means/SD reproduce the original trajectory summaries; the entire empirical efficiency CDF and its intervals match the manuscript; all 4,000 S7 episodes reproduce every mean and 95% interval |
| Fresh flat workflow | Adult and matched MLP at seed 0, both base/refinement stages at 128 steps; all six public stages completed; native, motion and four-friction evaluations retained failures |
| Capacity audit | Actual base actors: 1,989,016 / 2,057,362; final actors: 2,064,040 / 2,062,874 (Adult / matched MLP); both new MLP YAML recipes exactly match the completed full-budget research experiment |
| Flat notebook | `05_ant_flat.ipynb` executed in 161.6 s, requesting CUDA 0 for training and CPU evaluation; nine executed code cells, eight embedded PNGs, no error outputs; archived analysis and fresh smoke outputs clearly separated |
| Figure/display checks | Six quantitative PDF/SVG/PNG exports; PDF collision audit: zero failures/warnings, minimum text size 8 pt; notebook HTML, tables and embedded figures inspected in Chrome |
| Independent installation | Final wheel installed separately, editable import hooks disabled; all 15 presets and archived observations present; source/configuration checksum matches; archived analysis reproduces the same three adjusted P values |
| Maze regression | `05_maze.ipynb` re-executed in 698.2 s with CUDA 1 training and CPU evaluation: all seven methods, 46 prerequisite/student training stages, 34 evaluation files and all six public stages completed; original 26-test analysis retained |
| Notebook integrity | All 15 notebook schemas and 121 executed code cells checked; 50 embedded PNGs; both updated notebooks have current stage/source hashes and images matching generated files byte for byte |
| Materials | Tables S1/S2 export the new base/refinement PPO and architecture settings alongside the original maze settings |

The updated notebook and README provide both archived-data reanalysis and full-budget training commands. **This validation did not repeat the full-budget 20-seed training experiment in the public repository.** It verifies the recorded scientific results and fresh short-budget execution separately. Original maze model YAML files and numerical engines remain unchanged. Earlier notebook outputs outside the two Ant notebooks retain their original validation records.

## Earlier executed checks

| Check | Result |
|---|---|
| Automated integrity/numerical tests | 32 passed, including canonical figure imports in an isolated worker |
| Public inputs | Checksum-verified FlyWire v783, BANC v626, larval matrix and simulator assets; rebuilt graph arrays match their fixed contracts, including the separate developmental Adult partition |
| Packaged installation | Wheel built and installed into a separate directory; imports the new workflows and lists all 14 presets with bundled JSON/YAML configurations |
| Public CLI | Fresh Adult + MLP Tripod policies, seed 861, 128 training steps each; all six stages completed |
| Numbered entries and notebooks | All 14 presets have matching scripts/notebooks; notebook schemas, saved-output integrity, executable cell syntax, root/notebook-directory setup, six-stage order, preset mapping, full-command documentation and script help checked; Figure 1 materials entry executed |
| Notebook execution | All 14 notebooks executed in Jupyter with fresh short-budget policies and every configured method retained: 112 code cells, 84 stages, 42 embedded PNG previews and numerical tables; no error outputs. Saved stage records match their recorded source/configuration checksum, and embedded images match generated files byte for byte |
| Notebook distribution | Source archive includes all 14 notebooks, their common presentation helper and numbered scripts; compact notebook tables and PNG previews are retained; checkpoints, complete run data and logs are excluded; helper rejects full or mismatched runs before launching a subprocess |
| Main terrain | Fresh Adult/MLP/ER policies, 128 decisions; a small physical grid, coarse/dense sensitivities, matched blockade, ER atlas, ten-axis screen, measured gain surfaces, analysis and plots completed |
| Main speed and S2 | Fresh Adult/MLP base training and continuation to 128 decisions; both 12-second neural-history cases, local parameter perturbations, neuronal sensitivity, signed responses, target holds, analysis and plots completed |
| Main turning | Fresh Adult/MLP/ER policies in separate angle/push cohorts, 128 decisions; signed-angle, push and zero-control evaluations, common histories, sensitivity, interface exchange, activity replay, all 22 closed-loop intervention/force conditions, gain screens/surfaces, analysis and plots completed; restored intact replay is bit-exact |
| Terrain sensitivity (S1) | Separate fresh Adult/MLP policies, 128 decisions; unit-state and activity-scaled derivatives, finite-difference checks, five-height natural-response histories, cue exchanges, encounter-aligned analysis and plots completed |
| Developmental S3 | Fresh Adult/larva terrain and direction/angle policies; small physical evaluation grids, seed-level analyses and plots completed |
| Developmental S4 | Fresh Adult/larva speed policies, 128 steps; eight physical cases per model, native-policy replay checks, analysis and plots completed |
| BANC S5 | Fresh Adult/BANC-DN/BANC-motor/ER policies, 128 steps; behavior evaluation, initial/final and hybrid sensitivities, native/finite-difference checks, aggregation and plots completed |
| BANC S6 | Fresh DN/motor policies, 128 steps; matched-state branches across all six modes and zero/nonzero forces, model/state identity, zero-force shams, first-action checks, fixed-input probes, regional/factorial/readout summaries and plots completed |
| Turning statistical regression | Re-read 2,112 original intervention records without modifying them; the new four-test specificity family reproduces the archived adjusted p values within 2 × 10⁻¹², and all ten intact comparisons are retained |
| Turning aggregate regression | Recomputed the complete 18-comparison input-learning family and 48 parameter surfaces from copied original numerical records; results match the archived calculations |
| Terrain aggregate regression | Dense temporal summaries and all six blockade comparisons match the archived calculations; the complete 45-comparison Adult–ER family executes on the original numerical records |
| Ant short workflow | Fresh shared Adult/MLP prerequisites and teacher-label generation where required; all seven architectures at seed 0 completed base/flat/U-Maze/Medium/Large stages, diagnostic evaluations, analysis and plots |
| Added maze architectures | All 25 GRU/LSTM/Transformer/Reservoir/Reservoir+RO stage configurations construct and execute; fused values/gradients match recurrent stepping with resets; frozen reservoir parameters stay unchanged after an optimizer step |
| Body short workflows | Cassie, Go1, Tripod and Hexapod each completed all six public stages with fresh Adult/MLP policies, 128 training steps, a physical flat-condition evaluation, analysis and plots |
| Native media | Tripod Adult/MLP clips, six turning clips, Adult/MLP sinusoidal-speed clips and an Ant U-Maze MLP clip exported through EGL/ffmpeg |
| Figure checks | 84 generated PDFs (panels, trajectories and final CLI diagnostics) checked for text size and collisions; a cue-exchange legend collision was corrected and rechecked; representative PNGs visually inspected |
| Original workspace preservation | All 541 recorded original source/configuration files still match their pre-import hashes; changes are confined to FlyControl |

A short public-CLI check can be repeated with:

```bash
python scripts/06_tripod.py --run-dir runs/check-tripod \
  --smoke --device cpu --workers 1
```

Component checks use real simulator episodes and newly trained short-budget models. They exercise the larger numerical protocols without pretending that 128-step policies are trained paper models. The main turning and speed checks retain the native physical horizons required by their neural assays. These component runs were used during integration; source/configuration guards intentionally require a new public run directory after a code change.

The saved notebooks were executed after the S1 preset rename to `terrain-sensitivity`, using source/configuration SHA-256 `eaab77b7c7f8b5c50e7015c4c1c15e9ce272e2e432760c88fcc2e24c7a48ea9f`. Seven notebooks used CPU settings and seven requested CUDA for training and neural probes; the physical simulations run on CPU. Public inputs were already cached and revalidated, while all policies were trained afresh. Stage durations and execution settings are preserved in each notebook. The figure-numbered notebooks and setup instructions are indexed [here](../notebooks/README.md).

The saved notebook outputs were published in [commit `2bf8f5f`](https://github.com/THU-NeuroML/FlyControl/commit/2bf8f5fee28156f518f5f7595bb84308a66da49f), before the source directories were reorganized by figure. Those outputs and their original checksums are retained unchanged. Earlier component records remain associated with their original source/configuration version. Choose a fresh run directory after code/configuration changes.

The automated tests cover code/configuration changes, incomplete cohorts, concurrent writers, download corruption/path traversal, subprocess import isolation, checkpoint tampering, statistical units, exact sign-flip calculations, missing-endpoint multiplicity slots, complete intervention families, neural-history selection, seven-condition body aggregation, recurrent resets/gradients, frozen reservoirs, maze prerequisites and stage-policy construction.

## Source organization validation

The figure-based source reorganization was checked with 32 automated tests, including imports in an isolated worker that rejects unqualified experiment package names. A source comparison retained the computation in 330 functions/classes after excluding imports, module identifiers, docstrings and the removed short-import-root injection. The three changed dispatch functions were reviewed separately. All 280 canonical internal import declarations resolve to packaged modules. Numerical engine files and fixed configurations are byte-identical to the preceding release.

Five fresh smoke workflows retained every configured method and completed the public six-stage pipeline:

| Preset | Result | Elapsed time |
|---|---|---|
| `maze` | All six stages passed | 15.5 min |
| `tripod` | All six stages passed | 1.4 min |
| `development-turning` | All six stages passed | 18.7 min |
| `banc-recovery` | All six stages passed | 19.1 min |
| `terrain-sensitivity` | All six stages passed | 13.9 min |

These runs used source/configuration SHA-256 `547db804fbbeb246b3e1b5462ab51101da6a6d516ef5b6a17fd340fceced9576`. The Figure 5 run includes all seven architectures and their prerequisite/base/flat/U-Maze/Medium/Large training stages. S1 includes its neural sensitivity assays, S3 both direction and angle tasks, and S6 regional branching and fixed-input neural probes. Inputs were cached and reverified; the policies were newly trained.

After whitespace cleanup, the final source/configuration checksum is `b3f6a1932b01d9988c4434b0b5f4a3dba16efc37c6c51f932b2e44679eddded1`. All 32 tests passed again, and a fresh CPU Tripod run completed all six stages in 83.6 seconds. The final wheel was installed into a separate directory and imported with editable-install hooks disabled; its source checksum matches the repository, all 14 presets are present, and obsolete experiment package paths are absent. The 14 saved notebook files remain byte-identical to the preceding release.

## Limits

The development environment is Linux with Python 3.10.20. The task-local virtual environment reuses installed scientific dependencies; the wheel was separately installed and checked. This is not a clean-machine installation certification or a test of every Python/CUDA/driver combination.

No complete full-budget manuscript cohort has been retrained for this release. In particular, the large additional-body tasks have not completed full-budget training, and the full seven-architecture Ant cohort has not been rerun. The full combined developmental/body campaigns and all formal-cohort statistical branches have not been executed end to end on fresh full-budget models. The regression against archived turning records checks the calculation, not a new replication of the effect.

Rendering checks do not cover every one of the 20 recipes. Obstacle rendering and the complete body/maze condition grids still need full-run validation. Exported PDF/SVG panels are quantitative displays; the original artistic page layouts and video montages are not reproduced. PNG previews are provided; submission TIFF export is not claimed.

See [paper coverage](paper-coverage.md) for the complete implemented workflow map and [reproduction](reproduction.md) for cohort and failure-handling rules.

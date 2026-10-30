# Validation record

Date: 2026-10-30. These checks validate software execution and numerical contracts. They do not establish the manuscript's scientific conclusions.

## Executed checks

| Check | Result |
|---|---|
| Automated integrity/numerical tests | 31 passed |
| Public inputs | Checksum-verified FlyWire v783, BANC v626, larval matrix and simulator assets; rebuilt graph arrays match their fixed contracts, including the separate developmental Adult partition |
| Packaged installation | Wheel built and installed into a separate directory; imports the new workflows and lists all 14 presets with bundled JSON/YAML configurations |
| Public CLI | Fresh Adult + MLP Tripod policies, seed 861, 128 training steps each; all six stages completed |
| Numbered entries and notebooks | All 14 presets have matching scripts/notebooks; notebook schemas, cleared outputs, executable cell syntax, root/notebook-directory setup, six-stage order, preset mapping, full-command documentation and script help checked; Figure 1 materials entry executed |
| Notebook execution | `06_tripod.ipynb` executed in a Jupyter kernel on CPU with fresh Adult + MLP policies; all eight code cells and six stages completed, three generated PNG previews displayed, no error outputs; approximately 71 seconds with checksum-verified public input downloads already cached |
| Notebook distribution | Source archive includes all 14 notebooks, their common presentation helper and numbered scripts; generated weights/data/figures are excluded; helper rejects full or mismatched runs before launching a subprocess |
| Main terrain | Fresh Adult/MLP/ER policies, 128 decisions; a small physical grid, coarse/dense sensitivities, matched blockade, ER atlas, ten-axis screen, measured gain surfaces, analysis and plots completed |
| Main speed and S2 | Fresh Adult/MLP base training and continuation to 128 decisions; both 12-second neural-history cases, local parameter perturbations, neuronal sensitivity, signed responses, target holds, analysis and plots completed |
| Main turning | Fresh Adult/MLP/ER policies in separate angle/push cohorts, 128 decisions; signed-angle, push and zero-control evaluations, common histories, sensitivity, interface exchange, activity replay, all 22 closed-loop intervention/force conditions, gain screens/surfaces, analysis and plots completed; restored intact replay is bit-exact |
| Historical S1 | Separate fresh Adult/MLP policies, 128 decisions; unit-state and activity-scaled derivatives, finite-difference checks, five-height natural-response histories, cue exchanges, encounter-aligned analysis and plots completed |
| Developmental S3 | Fresh Adult/larva terrain and direction/angle policies; small physical evaluation grids, seed-level analyses and plots completed |
| Developmental S4 | Fresh Adult/larva speed policies, 128 steps; eight physical cases per model, native-policy replay checks, analysis and plots completed |
| BANC S5 | Fresh Adult/BANC-DN/BANC-motor/ER policies, 128 steps; behavior evaluation, initial/final and hybrid sensitivities, native/finite-difference checks, aggregation and plots completed |
| BANC S6 | Fresh DN/motor policies, 128 steps; matched-state branches across all six modes and zero/nonzero forces, model/state identity, zero-force shams, first-action checks, fixed-input probes, regional/factorial/readout summaries and plots completed |
| Turning statistical regression | Re-read 2,112 original intervention records without modifying them; the new four-test specificity family reproduces the archived adjusted p values within 2 × 10⁻¹², and all ten intact comparisons are retained |
| Turning aggregate regression | Recomputed the complete 18-comparison input-learning family and 48 parameter surfaces from copied original numerical records; results match the archived calculations |
| Terrain aggregate regression | Dense temporal summaries and all six blockade comparisons match the archived calculations; the complete 45-comparison Adult–ER family executes on the original numerical records |
| Ant components | Fresh shared Adult/MLP prerequisites, teacher-label generation, MLP seed 0 through flat/U-Maze/Medium/Large, diagnostic evaluations and plots completed |
| Added maze architectures | All 25 GRU/LSTM/Transformer/Reservoir/Reservoir+RO stage configurations construct and execute; fused values/gradients match recurrent stepping with resets; frozen reservoir parameters stay unchanged after an optimizer step |
| Other body initialization | Cassie, Go1 and Hexapod pinned assets, reset and ten simulation steps completed with finite observations/rewards |
| Native media | Tripod Adult/MLP clips, six turning clips, Adult/MLP sinusoidal-speed clips and an Ant U-Maze MLP clip exported through EGL/ffmpeg |
| Figure checks | 84 generated PDFs (panels, trajectories and final CLI diagnostics) checked for text size and collisions; a cue-exchange legend collision was corrected and rechecked; representative PNGs visually inspected |
| Original workspace preservation | All 541 recorded original source/configuration files still match their pre-import hashes; changes are confined to FlyControl |

A short public-CLI check can be repeated with:

```bash
python scripts/06_tripod.py --run-dir runs/check-tripod \
  --smoke --device cpu --workers 1
```

Component checks use real simulator episodes and newly trained short-budget models. They exercise the larger numerical protocols without pretending that 128-step policies are trained paper models. The main turning and speed checks retain the native physical horizons required by their neural assays. These component runs were used during integration; source/configuration guards intentionally require a new public run directory after a code change.

The notebook organization change preserves the numerical source/configuration digest. All 14 notebook entry paths were checked, but only the Tripod notebook was executed end to end in Jupyter; the component checks above describe the other workflows' existing validation. The figure-numbered notebooks and setup instructions are indexed [here](../notebooks/README.md).

The automated tests cover code/configuration changes, incomplete cohorts, concurrent writers, download corruption/path traversal, subprocess import isolation, checkpoint tampering, statistical units, exact sign-flip calculations, missing-endpoint multiplicity slots, complete intervention families, neural-history selection, seven-condition body aggregation, recurrent resets/gradients, frozen reservoirs, maze prerequisites and stage-policy construction.

## Limits

The development environment is Linux with Python 3.10.20. The task-local virtual environment reuses installed scientific dependencies; the wheel was separately installed and checked. This is not a clean-machine installation certification or a test of every Python/CUDA/driver combination.

No complete full-budget manuscript cohort has been retrained for this release. In particular, the large additional-body tasks have not completed training, and the seven-architecture Ant cohort has not been rerun. The full combined developmental/body campaigns and all formal-cohort statistical branches have not been executed end to end on fresh full-budget models. The regression against archived turning records checks the calculation, not a new replication of the effect.

Rendering checks do not cover every one of the 20 recipes. Obstacle rendering and the complete body/maze condition grids still need full-run validation. Exported PDF/SVG panels are quantitative displays; the original artistic page layouts and video montages are not reproduced. PNG previews are provided; submission TIFF export is not claimed.

See [paper coverage](paper-coverage.md) for the complete implemented workflow map and [reproduction](reproduction.md) for cohort and failure-handling rules.

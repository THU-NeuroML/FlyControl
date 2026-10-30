# Validation record

Date: 2026-10-30. These checks validate software behavior, not the manuscript's scientific conclusions.

## Executed checks

| Check | Result |
|---|---|
| Automated integrity/numerical tests | 26 passed |
| Public-input preparation | FlyWire files downloaded with checksum verification; all three graph partitions rebuilt and matched against fixed array contracts |
| Packaged installation | Wheel built; separately installed wheel imports its bundled configurations and lists all eight presets |
| Tripod public CLI | Fresh Adult + MLP, seed 861, 128 training steps each; `prepare → train → evaluate → analyze → plot` completed |
| Turning components | Fresh MLP, seed 4301, 128 decisions; six signed-angle evaluations, metrics and plots completed |
| Speed components | Fresh MLP, seed 4401, 64-decision base continued to 128; fixed-history evaluation, metrics and plots completed |
| Terrain components | Fresh MLP, seed 2011, 128 decisions; one steady-state/height evaluation and exact native-state replay, metrics and plots completed |
| Ant components | Fresh shared Adult/MLP prerequisites, teacher-label generation, MLP seed 0 through flat/U-Maze/Medium/Large, diagnostic evaluations and plots completed |
| Added maze architectures | All 25 GRU/LSTM/Transformer/Reservoir/Reservoir+RO stage configurations construct and execute; fused sequence values and gradients match recurrent stepping with resets; frozen reservoir tensors stay unchanged after an optimizer step |
| Cassie, Go1 and Hexapod | Pinned assets downloaded; reset and ten simulation steps completed with finite observations/rewards |
| Plot QA | Final PDFs checked for minimum text size and collisions; PNGs visually inspected; all exported diagnostic figures contain one axes, so panel-alignment checks are not applicable |
| Source-workspace preservation | All 410 initially inspected/imported source/config files in the original curated repository still match their recorded hashes |

The final CLI smoke test can be repeated with:

```bash
flycontrol run --config tripod --run-dir runs/check-tripod \
  --smoke --device cpu --workers 1
```

Smoke mode includes small training-internal checkpoint evaluations as well as a small final evaluation grid. An earlier check exposed the unbounded internal evaluation; this was corrected before the successful final CLI run. Full-mode internal evaluation retains its original 20-second horizon and reset cohort.

The automated tests cover configuration/code changes, invalid and incomplete cohorts, concurrent writers, download corruption/path traversal, subprocess import isolation, statistical units, exact sign-flip calculations, multiplicity slots for undefined endpoints, the seven-condition body aggregate, recurrent resets/gradients, frozen reservoir parameters, the maze prerequisite graph and actual stage-policy construction.

## Limits

The development environment is Linux with Python 3.10.20. The task-local virtual environment reuses installed scientific dependencies; the wheel was also installed into a separate target directory. This is not a certification of a completely blank machine or all supported Python/CUDA/driver combinations.

The three large external-body tasks were checked through initialization and stepping, not complete training. The complete seven-architecture Ant cohort has not been trained. No full-budget 20/24-seed manuscript cohort has been rerun. Full-cohort statistical families and final manuscript panel composition therefore remain unvalidated end to end.

Diagnostic PNG export is intentional; submission TIFF export has not been claimed. Static plot-source checks report no failures, with advisory warnings for the PNG-only raster and the bootstrap RNG appearing in the combined Ant analysis/plot module. The latter resamples observed seed differences and does not generate observations.

See [paper coverage](paper-coverage.md) for the neural-mechanism, larval and BANC workflows still required before a reviewer release.

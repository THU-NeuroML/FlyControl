# Reproduction protocol

## Inputs and model initialization

The public interface is `flycontrol`. Each stage reads an explicitly selected run directory. It does not search for checkpoints in another checkout or fall back to historical result files.

Input URLs, exact revisions, sizes and SHA-256 hashes are recorded under `configs/inputs/`. A corrupt existing input is rejected. Downloads use temporary files and become visible only after validation. The loader verifies graph dimensions, edge order, weights and array hashes before training.

Three FlyWire v783 partitions are used:

| Graph | Input / internal / output cells | Use |
|---|---|---|
| `motor` | 18,756 / 119,719 / 109 | Ant and other robots |
| `descending` | 18,756 / 118,538 / 1,290 | Fly turning and speed |
| `terrain` | 2,279 / 135,015 / 1,290 | Fly obstacle traversal |

The terrain graph preserves the original descending-partition node order when extracting its ascending inputs, and preserves the motor graph's edge ordering. Re-sorting these arrays changes floating-point accumulation and the initialized interfaces; it is not an equivalent preparation step.

NeuroMechFly geometry, gait definitions and other required simulator assets come from the fixed public FlyGym revision in `nmf_assets.json`. Robot geometry comes from pinned MuJoCo Menagerie / One Policy to Run Them All revisions. No learned FlyControl policy is an input.

## Fixed training and evaluation

The root experiment JSON selects the cohort. Numerical settings live under `configs/internal/`; they are copied into each run before execution. The same public commands apply to every preset.

| Experiment | Training budget per main policy | Full evaluation |
|---|---|---|
| Terrain | 32,768 decisions | 5 heights × 64 common starting phases |
| Turning | 8,192 decisions | Six signed angles × fixed starting phases |
| Speed | 12,800 decisions, continued to 15,872 | Eight registered command/physics histories |
| Cassie, Go1 | 524,288 steps | 7 conditions × 20 evaluation seeds |
| Tripod, Hexapod | 131,072 steps | 7 conditions × 20 evaluation seeds |
| Ant | Stage-specific PPO and distillation budgets | 100 resets per main task and training seed; additional motion/friction checks |

Full runs are substantial experiments. Wall time depends on CPU workers, the accelerator and the number of graph policies. Short-run timing is not extrapolated into a claimed full-paper runtime. `--workers` controls evaluation concurrency; training vectorization is fixed by the numerical protocol. Multiple simultaneous writers to one run are rejected.

`--smoke` reduces training budgets and evaluation coverage, and uses one training seed by default. It retains real physics and real graph inputs. Its plots are explicitly labeled as smoke tests. It provides no inferential confidence interval or paper-level statistical claim. Method/seed subsets are useful for debugging but do not constitute a complete paper cohort.

## Ant dependencies

The complete chain is explicit in `src/flycontrol/ant.py` and `configs/internal/maze/`:

1. Train the shared Adult and MLP prerequisite policies through their five registered stages.
2. Train each architecture's base → flat → U-Maze chain.
3. Generate shared Medium teacher labels and train each Medium student from its specified source.
4. Generate Large waypoint labels; train the shared goal router from the MLP Medium source.
5. Insert that router while retaining each student's body policy and proprioceptive normalization; train Large students using the shared action-label corpus.
6. Evaluate the newly generated flat and maze checkpoints at the fixed reset seeds.

The historical source commonly labeled “v15” reused the v8 policy weights and changed the task interface. The new chain trains the actual v8 weight-generating stages and applies the destination interface configuration. It does not add a fictitious extra training stage.

The five added architectures are GRU, LSTM, Transformer, Reservoir and Reservoir+RO. Reservoir recurrence is a frozen buffer. Reservoir+RO also freezes its input projection. Shared prerequisite policies and label corpora are part of the experimental design, so the main seeds are not independent graph or teacher draws.

A pinned public GoalAnt teacher is downloaded to the run directory. It is an external experimental dependency used for distillation, not a FlyControl result checkpoint. Its repository and checksum are in `configs/inputs/teacher.json`.

## Outputs and interpretation

```text
runs/<experiment>/
  run.json                       Cohort, smoke flag and code/config digest
  configuration_manifest.json    Hashes of effective fixed configurations
  environment.json               Python/platform and installed numerical versions
  protocol.json                  Effective numerical protocol
  inputs/                        Verified downloads and generated graph
  assets/                        Downloaded simulator geometry
  training/                      Newly trained checkpoints and model hashes
  evaluation/                    New episode records and traces
  analysis/                      Episode tables, seed means and comparisons
  figures/                       PDF, SVG and PNG diagnostic plots
  status/                        Completed stage markers
```

The analysis stage rejects missing or duplicate evaluation cells and mismatched checkpoint/trace hashes. It first averages repeated evaluations within each training seed. Plotting verifies analysis-table and raw-evaluation hashes.

`paired_comparisons.csv` contains behavioral diagnostics. Its correction family is explicitly recorded in the manifest. For complete terrain and speed cohorts, `paper_behavior.csv` separately preserves the original fixed behavioral comparison families. Missing planned endpoint values retain their multiplicity slots and do not receive invented values. Mechanism families are not represented by these behavior-only tables.

`combine-bodies` requires all four full cohorts. It weights the seven conditions equally within each seed and applies Holm correction across the four primary body effects. Its detailed exploratory output retains the condition/window families and reports incomplete paired windows.

A completed stage can be rerun if its inputs still match. Interrupted body training resumes from its checkpoint. Some fly trainers intentionally reject a partially written training directory; use a fresh run directory instead of manually marking a stage complete. Ant stages keep explicit completion markers. No automatic deletion or replacement of an unfinished run occurs.

## Verification

```bash
python -m pytest -q
```

See [validation](validation.md) for actual execution evidence and its limits. Exact numerical equality across hardware, drivers and library versions is not assumed. The source/config guards prevent accidental mixing within a run; statistical replication still requires the complete registered cohorts.

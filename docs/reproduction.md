# Reproduction protocol

## Inputs and model initialization

The public interface is `flycontrol`. Each stage reads an explicitly selected run directory. It does not search for checkpoints in another checkout or fall back to historical result files.

Input URLs, exact revisions, sizes and SHA-256 hashes are recorded under `configs/inputs/`. A corrupt existing input is rejected. Downloads use temporary files and become visible only after validation. The loader verifies graph dimensions, edge order, weights and array hashes before training.

The main-task FlyWire v783 partitions are:

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

The updated flat-ground comparison uses the independent `ant-flat` preset (Fig. 5b–d and S7): base PPO → flat refinement → native flat/motion/friction evaluation. It does not require the maze teacher or routers. See [the fixed 20-seed protocol, archived-data command and 29-test correction](ant-flat.md).

### Original maze chain (Fig. 5e–j and S8)

The complete chain is explicit in [`experiments/fig5/workflow.py`](../src/flycontrol/experiments/fig5/workflow.py) and `configs/internal/maze/`:

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
  mechanisms/, data/             Generated neural histories and assay records
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

## Neural and supplementary stages

All runs now include `probe` between `evaluate` and `analyze`. Presets without a neural assay record that explicitly. Paired main-fly probes require Adult and MLP histories; a single-method smoke test skips that assay and records the reason. Full runs do not silently replace missing histories or pad terminated trajectories into complete neural windows.

Terrain uses the first phase in each of eight registered phase bins for its 80 common histories (five heights × eight phases × two controller origins). The full physical evaluation still uses 64 phases. ER preserves the signed weight multiset and node coordinates, not the degree sequence. Surface axes remain the manuscript's exploratory choice. Discovery/validation seed splits and the original coarser grid are retained for held-out inference; the displayed dense surface is directly measured.

Turning trains independent angle and push cohorts for all three architectures. Neural analyses regenerate the common seed-4301 bank, nine activity histories per Adult model, four initial/final interface exchanges and matched anatomical cuts. The input-learning family contains nine classes × two endpoints. Zero initial readouts are recorded as undefined relative sensitivities.

Turning pathway effects retain all ten comparisons against intact control and the separate four-comparison specificity family. Physical states are averaged within each training seed before paired tests and t-based intervals. No undisplayed intervention is dropped from multiplicity correction.

The speed workflow uses the same newly trained policies for tracking, fixed-history response, neuronal sensitivity, signed activity, input holds, parameter perturbation and the 45-association family. `--smoke` retains both required neural-history cases. Prespecified histories that terminate early are reported; their absence cannot be repaired by choosing another model after seeing the outcome.

The S1 six-model protocol has its own numerical engine and does not reuse the current terrain policies. Unit-state and activity-scaled derivatives use the original dense horizons and finite-difference checks. The same historical cohort also supplies the Methods' obstacle/flat, height-cue exchange and encounter-aligned response analysis. Its complete-window counts are retained.

Adult/larva speed, turning and terrain are independent cohorts with their original graph normalization and physics. The terrain configuration supplies the ten undisplayed tests in the joint 22-contrast developmental family. The separate speed family contains 20 contrasts. Use `combine-development` after both relevant complete analyses; the turning-only table reserves the ten missing slots conservatively and is not the final joint result.

## BANC and larval inputs

BANC uses **release 626**, matching the manuscript experiment, not release 888. Both readout partitions are reconstructed from public annotations and connections, and every graph array is checked against the frozen numerical contract. The current public v626 annotation file contains updated names for two cells; partition-defining annotations and the resulting graph arrays match the experiment. The exact downloaded annotation checksum is pinned.

The larval matrix and annotations come from a pinned revision of `brain-networks/larval-drosophila-connectome`. Speed uses its row-L1 graph; the turning configuration retains its distinct normalization. The additional developmental terrain Adult graph has 16,477 input, 120,817 internal and 1,290 output cells; it must not be replaced with the main-fly partition.

BANC sensitivity uses four source-model histories, initial/final input and readout combinations, two evaluation resets and the registered event windows. A history too short for a full window is marked as excluded with its reason. Regional recovery separately saves a complete physics, neural, CPG and RNG snapshot. Each intervention runs in a separate process, verifies the model/state identity, and checks zero-force shams and the first action against the reference. Region fractions are based on synapse location. Fixed-input assays and physical recovery are separate outputs.

## Rendering and configuration tables

`flycontrol materials --output runs/materials` exports configuration tables with source checksums and an editable architecture SVG. These are machine-readable records of the actual recipes; they do not create measured results.

`flycontrol replay` requires a completed evaluation and `ffmpeg`. Saved terrain/body states are rendered without stepping physics; turning, speed and Ant clips rerun the configured deterministic policy and initial state. Video recipes follow supplementary-caption numbering. Separate clips retain their method/condition labels and failure terminations; original cinematic compositions and outcome-selected examples are not assumed to recur after new training.

# FlyControl

From-scratch training, evaluation, neural analysis and plotting for connectome-based control.

This repository contains **code, fixed configurations and documentation**. Public anatomical inputs and simulator assets are downloaded with version and checksum checks. Policies are trained within each run; generated weights, data, figures and videos stay outside Git.

The main and supplementary workflows share one interface. [Paper coverage](docs/paper-coverage.md) maps the manuscript to commands and outputs. [Validation](docs/validation.md) distinguishes executed software checks from full-budget scientific replication, which has not yet been completed for this release.

## Install

Use Linux and Python 3.10:

```bash
git clone --depth 1 https://github.com/THU-NeuroML/FlyControl.git
cd FlyControl
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
flycontrol list
```

Numerical dependencies are pinned in `pyproject.toml`. Headless rendering needs EGL; alternatively install OSMesa and set `MUJOCO_GL=osmesa`. Native video export additionally requires `ffmpeg` on `PATH`. Training and probes accept `--device cpu` or an available CUDA device; physical evaluations use CPU.

## Reproduce an experiment

First check installation with a short, real-simulator run:

```bash
flycontrol run --config tripod --run-dir runs/tripod-smoke \
  --smoke --device cpu --workers 1
```

Smoke results are labelled and cannot establish the paper's conclusions. To execute a complete configured experiment:

```bash
flycontrol run --config terrain --run-dir runs/terrain \
  --device cuda:0 --workers 4
```

The equivalent stages are:

```bash
flycontrol prepare  --config terrain --run-dir runs/terrain
flycontrol train    --run-dir runs/terrain --device cuda:0
flycontrol evaluate --run-dir runs/terrain --workers 4
flycontrol probe    --run-dir runs/terrain --device cuda:0
flycontrol analyze  --run-dir runs/terrain
flycontrol plot     --run-dir runs/terrain
```

`prepare` verifies public inputs and builds the required graph and anatomy. `train` produces initial and final checkpoints. `evaluate` generates physical episodes, including failures. `probe` replays recorded inputs and runs the configured neural assays. `analyze` aggregates repeated observations within training models and applies the specified comparison families. `plot` exports editable PDF/SVG panels, PNG previews and their numerical sources.

Each run records effective configurations, source checksums, checkpoint checksums and the numerical environment. Use a new run directory after changing code or configurations. The original research workspace is not required.

## Manuscript workflows

| Preset | Results | Fixed cohort |
|---|---|---|
| `terrain` | Fig. 2 behavior, ER control, temporal sensitivity, blockade and gain planes | Adult / MLP / ER × 20 seeds |
| `turning` | Fig. 3 behavior, input learning, activity, AN–DN interventions and gain planes | Adult / MLP / ER × 24 seeds × angle/push cohorts |
| `speed` | Fig. 4 and S2: tracking, neural responses, target holds, parameter planes and associations | Adult / MLP × 20 seeds |
| `historical-terrain` | S1 unit-state and activity-scaled sensitivity | Separate Adult / MLP × 6 seeds |
| `development-turning` | S3 Adult/larva direction and angle control | Adult / larva × 6 seeds per task |
| `development-terrain` | Ten terrain contrasts in S3's joint statistical family | Adult / larva × 6 seeds |
| `development-speed` | S4 dynamic speed and perturbation protocols | Adult / larva × 6 seeds |
| `banc-sensitivity` | S5 initial/final readout sensitivity and matched controls | Adult / BANC-DN / BANC-motor / ER × 6 seeds |
| `banc-recovery` | S6 fixed-input message substitution and closed-loop recovery | BANC-DN / BANC-motor × 3 seeds |
| `maze` | Fig. 5, S7 friction and S8 Large maze | Seven architectures × 20 seeds |
| `cassie`, `go1`, `tripod`, `hexapod` | Fig. 6, S9 and S10 | Adult / MLP × 20 seeds per body |

Two comparisons combine experiments:

```bash
flycontrol combine-development --terrain runs/development-terrain \
  --turning runs/development-turning --output runs/development-combined
flycontrol combine-bodies --run-dirs runs/cassie runs/tripod runs/go1 runs/hexapod \
  --output runs/bodies-combined
```

These require complete cohorts. The developmental command preserves all 22 comparisons; the body command uses equal weights over seven conditions and corrects across four bodies.

To execute the whole fixed campaign, use `bash scripts/reproduce.sh cuda:0 4 runs/paper`. This launches full training and is computationally substantial. See [protocol details](docs/reproduction.md) for dependency chains, sample units and interpretation.

## Figures, tables and videos

Results are written to `runs/<experiment>/analysis/` and `figures/`. `figures/paper/` contains quantitative panels, with each panel's source values and aggregation recorded. These are regenerated scientific displays, not copies of the manuscript's finished page layouts.

```bash
flycontrol materials --output runs/materials
flycontrol replay --run-dir runs/turning --video 1 --output runs/videos/01
```

`materials` exports an editable controller overview and Tables S1–S4 configuration records. `replay` uses the 20 supplementary-caption recipes in `configs/videos.json` and writes native clips for each method/condition. Selection is fixed to the first configured training seed and registered initial state; the clips are illustrations, not population evidence. See [video mapping](docs/videos.md) for numbering and the existing published archive.

## Structure

```text
configs/                 Experiment presets, numerical recipes, public-input manifests
src/flycontrol/
  cli.py, pipeline.py    Commands and stage integrity
  inputs.py, anatomy.py Verified graphs and anatomical annotations
  engines/              Task-specific policies, trainers and physics
  experiments/          Numerical probes and metric definitions
  *analysis.py          Behavior and neural analyses
  plotting.py           Quantitative figures
  media.py, materials.py Native replays, overview and configuration tables
scripts/reproduce.sh    Full campaign and combined comparison families
tests/                   Integrity and numerical regression checks
docs/                    Protocol, coverage and executed validation
```

The separate engines preserve experiments with different physics, observation scaling or recurrent update rules. Common orchestration does not substitute one protocol for another.

## License and external inputs

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Downloaded anatomy, geometry and the public third-party Ant teacher retain their upstream terms. The teacher is a required distillation input, not a FlyControl trained result. Optional future model releases can be hosted separately on Hugging Face; no Hugging Face account or downloaded FlyControl checkpoint is required here.

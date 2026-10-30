# FlyControl

Code and fixed configurations for training, evaluating and analyzing connectome-based controllers.

**Status:** the core behavior workflows are implemented. The complete manuscript reproduction is still being assembled. See [paper coverage](docs/paper-coverage.md) for the remaining mechanism and supplementary experiments, and [validation](docs/validation.md) for what has actually been tested. A short smoke run checks the software; it does not reproduce a paper result.

## Install

Use Linux and Python 3.10. The simulator workflows are tested with MuJoCo 3.2.7. Use a separate environment:

```bash
git clone https://github.com/THU-NeuroML/FlyControl.git
cd FlyControl
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
flycontrol list
```

A headless machine needs a working EGL driver. For software rendering, install the system OSMesa library and set `MUJOCO_GL=osmesa`. Training can use `--device cpu` or an available CUDA device such as `--device cuda:0`; evaluation uses CPU. Dependencies are pinned in `pyproject.toml`.

## Run an experiment

Start with a small real-simulator check:

```bash
flycontrol run --config tripod --run-dir runs/tripod-smoke \
  --smoke --methods mlp --device cpu --workers 1
```

For the full fixed protocol, omit `--smoke` and the method/seed filters. For example:

```bash
flycontrol run --config turning --run-dir runs/turning --device cuda:0 --workers 4
```

The same workflow can be run one stage at a time:

```bash
flycontrol prepare  --config turning --run-dir runs/turning
flycontrol train    --run-dir runs/turning --device cuda:0
flycontrol evaluate --run-dir runs/turning --workers 4
flycontrol analyze  --run-dir runs/turning
flycontrol plot     --run-dir runs/turning
```

`prepare` downloads versioned public inputs and checks their SHA-256 hashes. It builds the graph from source annotations and connections. `train` creates the study's models from their specified initializations; dependent training stages are explicit. `evaluate` uses those newly trained checkpoints. `analyze` checks the evaluation grid and aggregates repeated evaluations within each training seed. `plot` exports editable PDF/SVG figures and PNG previews from the resulting tables.

Each run owns its downloaded inputs, effective configurations, checkpoints and outputs. No original research workspace is needed. Generated files stay under `runs/`, which Git ignores. Changing code or an effective configuration after initialization requires a new run directory.

## Configurations

| Configuration | Experiment | Fixed training cohort |
|---|---|---|
| `terrain` | Fly obstacle traversal | Adult and MLP, 20 seeds each |
| `turning` | Fly heading control | Adult and MLP, 24 seeds each |
| `speed` | Continuous speed control | Adult and MLP, 20 seeds each |
| `maze` | Ant flat, U-Maze, Medium and Large | Seven architectures, 20 seeds each |
| `cassie` | Biped balance | Adult and MLP, 20 seeds each |
| `go1` | Quadruped locomotion | Adult and MLP, 20 seeds each |
| `tripod` | Three-legged locomotion | Adult and MLP, 20 seeds each |
| `hexapod` | Six-legged locomotion | Adult and MLP, 20 seeds each |

After all four full body runs have been analyzed, compute the paper's equal-weight seven-condition comparison and four-body correction:

```bash
flycontrol combine-bodies \
  --run-dirs runs/cassie runs/tripod runs/go1 runs/hexapod \
  --output runs/four-bodies
```

The Ant experiment includes shared prerequisite policies and teacher-label generation. Its 20 seeds are optimization replicates conditional on those shared dependencies. The public third-party teacher is a required experimental input; paper-trained FlyControl weights are not downloaded. See [protocol details](docs/reproduction.md).

## Repository structure

```text
configs/                  Experiment presets and pinned input manifests
src/flycontrol/
  cli.py                  Public commands
  pipeline.py             Stage ordering and run integrity
  inputs.py               Downloads and graph construction
  engines/                Numerical policies, trainers and body simulators
  experiments/            Numerical evaluation and metric functions
  analysis.py             Complete-grid checks and seed-level diagnostics
  paper_statistics.py     Fixed behavioral comparison families
  plotting.py             Figures from newly generated tables
tests/                    Integrity and numerical regression checks
docs/                     Reproduction instructions and paper coverage
```

Only code, configuration and documentation are added to this repository. Checkpoints, downloaded datasets, simulation assets, logs and generated figures belong to local run directories. A future model release can be versioned separately on Hugging Face; no model upload or external model repository is required for these workflows.

## Supplementary videos

The previously published [20 supplementary videos at tag v1.0.0](https://raw.githubusercontent.com/THU-NeuroML/FlyControl/v1.0.0/Supplementary-Videos.zip) remain available. The archive remains in the published tag; this source branch does not carry the binary. It is not an input to training or analysis. Its SHA-256 is `acf91ab098a66b0af96a547cb10decfa0981b6bfc6c2d82754ba22e670e2e0e3`.

## License and sources

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Downloaded data and robot assets retain their upstream terms.

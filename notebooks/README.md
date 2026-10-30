# Short tests by figure

Each notebook includes **saved outputs from a real short test**: stage completion times, numerical tables and generated figures. Open a notebook on GitHub to inspect these results directly. The embedded previews are labelled **SMOKE TEST** and do not establish the paper's conclusions.

To regenerate the outputs, run the notebook through **prepare → train → evaluate → probe → analyze → plot** with the existing `--smoke` configuration. Use the matching numbered script for full reproduction.

From the repository root, activate the environment and install the optional notebook tools:

```bash
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name flycontrol --display-name "Python (FlyControl)"
jupyter lab notebooks/
```

Select **Python (FlyControl)**, open a notebook and run its cells in order. Start with `06_tripod.ipynb`. `DEVICE` defaults to `cpu`; CUDA can accelerate training and neural probes. Set `DEVICE` in the notebook or launch Jupyter with `FLYCONTROL_NOTEBOOK_DEVICE=cuda:0`. The saved outputs record the device actually used. Generated files go to `runs/notebooks/<preset>/`. Every notebook ends with the full command as Markdown, so **Run All only executes the short test**.

Keep the saved table and figure outputs when sharing these notebooks. Checkpoints, full numerical outputs and execution logs remain in the ignored run directory. Stage cells show concise completion records; complete logs are saved under `RUN_DIR/logs/notebook-<stage>.log`. The notebook metadata records the execution time, source/configuration checksum and run settings.

| Figure | Short-test notebook | Full-run script |
|---|---|---|
| Figure 2 | [`terrain`](02_terrain.ipynb) | [`02_terrain.py`](../scripts/02_terrain.py) |
| Figure 3 | [`turning`](03_turning.ipynb) | [`03_turning.py`](../scripts/03_turning.py) |
| Figure 4 · Figure S2 | [`speed`](04_speed.ipynb) | [`04_speed.py`](../scripts/04_speed.py) |
| Figure 5 · Figures S7–S8 | [`maze`](05_maze.ipynb) | [`05_maze.py`](../scripts/05_maze.py) |
| Figure 6 · Cassie | [`cassie`](06_cassie.ipynb) | [`06_cassie.py`](../scripts/06_cassie.py) |
| Figure 6 · Go1 | [`go1`](06_go1.ipynb) | [`06_go1.py`](../scripts/06_go1.py) |
| Figure 6 · Figure S9 | [`tripod`](06_tripod.ipynb) | [`06_tripod.py`](../scripts/06_tripod.py) |
| Figure 6 · Figure S10 | [`hexapod`](06_hexapod.ipynb) | [`06_hexapod.py`](../scripts/06_hexapod.py) |
| Figure S1 | [`terrain-sensitivity`](S01_terrain_sensitivity.ipynb) | [`S01_terrain_sensitivity.py`](../scripts/S01_terrain_sensitivity.py) |
| Figure S3 · terrain comparison family | [`development-terrain`](S03_development_terrain.ipynb) | [`S03_development_terrain.py`](../scripts/S03_development_terrain.py) |
| Figure S3 · turning | [`development-turning`](S03_development_turning.ipynb) | [`S03_development_turning.py`](../scripts/S03_development_turning.py) |
| Figure S4 | [`development-speed`](S04_development_speed.ipynb) | [`S04_development_speed.py`](../scripts/S04_development_speed.py) |
| Figure S5 | [`banc-sensitivity`](S05_banc_sensitivity.ipynb) | [`S05_banc_sensitivity.py`](../scripts/S05_banc_sensitivity.py) |
| Figure S6 | [`banc-recovery`](S06_banc_recovery.ipynb) | [`S06_banc_recovery.py`](../scripts/S06_banc_recovery.py) |

Figure S2 shares the speed workflow; S7–S8 share maze; S9 shares Tripod; S10 shares Hexapod. This avoids training the same cohort again under a different figure name. Figure 1 and Tables S1–S4 use `python scripts/01_materials.py --output runs/materials`; they do not train a model and need no separate smoke notebook.

“Short” means reduced training budgets and evaluation coverage. The first run downloads public inputs, and neural assays or maze prerequisites may still be slow. All configured methods are retained. Missing usable neural histories are recorded, not replaced with fabricated results. Smoke outputs are not evidence for the paper's claims.

The common `notebook_utils.py` helper records CLI logs, displays progress, stops child processes on interruption and displays generated files. Training, evaluation, analysis and plotting remain in the same package used by full runs. Use a fresh run directory after code/configuration changes. For a failed training stage, follow [resume rules](../docs/reproduction.md); notebooks do not add a separate resume mechanism.

For the complete campaign, run `bash scripts/00_reproduce_all.sh cuda:0 4 runs/paper` from the repository root. The combined developmental and body statistics require full cohorts and are included by that script; individual notebook endings also show these commands where applicable.

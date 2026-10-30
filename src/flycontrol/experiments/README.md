# Experiments by manuscript figure

Start with the figure below, then follow its fixed preset through `prepare → train → evaluate → probe → analyze → plot`. The public CLI, numbered scripts and notebooks all dispatch to these implementations.

| Figure | Source directory | What lives here | Public preset |
|---|---|---|---|
| 1; Tables S1–S4 | [`fig1/`](fig1/) | Editable controller overview and configuration-table export (`materials.py`) | `flycontrol materials` |
| 2 | [`fig2/`](fig2/) | Terrain workflow; behavior, ER controls, state sensitivity, circuit blockade and input-gain assays | `terrain` |
| 3 | [`fig3/`](fig3/) | Turning anatomy and workflow; registered push cohort; neural histories, activity, AN–DN interventions and gain assays | `turning` |
| 4; S2 | [`fig4/`](fig4/) | Speed analysis, target holds, local response, signed responses and neural associations | `speed` |
| 5; S7–S8 | [`fig5/`](fig5/) | Ant prerequisites, training and evaluation (`workflow.py`); maze/friction statistics and plots (`analysis.py`) | `maze` |
| 6; S9–S10 | [`fig6/`](fig6/) | Body training/evaluation, condition and time-window analyses, joint four-body comparison | `cassie`, `go1`, `tripod`, `hexapod` |
| S1 | [`supp_s1/`](supp_s1/) | Independent terrain cohort, state-sensitivity and activity-scaled sensitivity analyses | `terrain-sensitivity` |
| S3 | [`supp_s3/`](supp_s3/) | Adult/larva terrain and turning workflows; shared 22-comparison statistical family | `development-terrain`, `development-turning` |
| S4 | [`supp_s4/`](supp_s4/) | Adult/larva speed summary and the 20-comparison statistical family | `development-speed` |
| S5 | [`supp_s5/`](supp_s5/) | BANC readout sensitivity, initial/final parameter hybrids and aggregation | `banc-sensitivity` |
| S6 | [`supp_s6/`](supp_s6/) | BANC regional anatomy, same-state branching, stable replay, neural probing and recovery statistics | `banc-recovery` |

S2 extends the Figure 4 speed analysis. S7 (friction) and S8 (Large maze) use the Figure 5 policies. S9 (Tripod) and S10 (Hexapod) use the Figure 6 body runs. Their implementation is in the indicated main-figure directory: they do not require duplicate packages or new training cohorts.

[`shared/`](shared/) contains the common supplemental workflow dispatcher, paths, perturbation/statistical utilities and speed evaluation used by S4 and S5. S3, S4, S5 and S6 use the same supplemental trainer through `shared/workflow.py`; their specific assays and statistical families remain in their figure directories.

Shared statistical dispatch lives in [`../analysis/`](../analysis/); common panel and native replay rendering lives in [`../plotting/`](../plotting/). [`../engines/`](../engines/) holds isolated numerical policies, trainers and physics. Engine names and run-local data/configuration paths retain their registered protocol identities. For example, S1 uses the `historical` engine, while its public preset is `terrain-sensitivity` and its experiment code is `supp_s1`.

Use canonical imports such as `flycontrol.experiments.fig5.workflow`. Numerical workers add only their selected engine roots to `PYTHONPATH`; the `experiments/` directory is not a separate import root.

See the [script/notebook index](../../../README.md), [reproduction guide](../../../docs/reproduction.md) and [paper coverage](../../../docs/paper-coverage.md) for commands, complete cohorts and claim-to-output mapping.

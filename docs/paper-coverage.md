# Manuscript coverage

Reference: the author-confirmed **Fly_Locomotion (13).pdf**. The manuscript and historical research archive are not runtime inputs and are not copied into this repository.

This is an implementation checklist, not a claim that all paper results have already been reproduced. “Workflow implemented” means the public command connects fresh inputs to training/evaluation/analysis. Full-budget scientific replication and pixel-identical reconstruction of the manuscript panels have not been certified.

| Manuscript result | Current implementation | Remaining work |
|---|---|---|
| Fig. 1: fixed connectome with learned interfaces | Shared policies and three verified graph partitions | Editable overview figure and complete parameter tables |
| Fig. 2b–d: terrain behavior | `terrain`: fresh training, common-state evaluation, seed summaries and fixed behavior comparison families | Full 20-pair run; final paper panel composition |
| Fig. 2e: matched ER sensitivity | Matched ER construction/training kernel retained internally | Public control/probe orchestration and comparison family |
| Fig. 2f–g: temporal organization and transmission blocks | Not exposed in the public workflow | Fresh anatomical annotation, history collection, neural probes and complete intervention family |
| Fig. 2h–i: selected trajectory and input-gain surface | New evaluation traces are generated | Explicit selection rule, measured surface and rendering |
| Fig. 3b–d: heading behavior | `turning`: fresh training, six signed-angle evaluations, failure-aware metrics and diagnostic figures | Registered angle/push/ER control family and final directional inference |
| Fig. 3e–h: sensitivity, learned input, activity and AN→DN intervention | Not exposed | All 144 registered final models, initial/final probes and full statistical families |
| Fig. 3i–j: +60° example and gain surface | New heading trajectories are generated | Fresh common histories, axis-selection rule and measured surfaces |
| Fig. 4b–c: speed behavior | `speed`: base PPO → continuation → fixed histories; failure-aware metrics and three-endpoint family | Full 20-pair run and onset-response panel |
| Fig. 4d–g: pathway response, input hold and 45 associations | Not exposed | Anatomical preparation, neural collection, interventions and all 45 associations |
| Fig. 4h–i: actor-parameter perturbations | Not exposed | Fixed perturbation directions, new policy responses and surfaces |
| Fig. 5: Ant flat and maze behavior | `maze`: explicit prerequisite DAG, seven architectures, four tasks, additional motion checks, 26-slot Adult/MLP comparison family | Full cohorts; trajectory/path panels and paper composition |
| Fig. 6: four bodies × seven conditions | Four presets plus `combine-bodies`; primary seven-condition aggregate and detailed condition/window statistics | Full cohorts and paper figure composition |
| Fig. S1: historical six-model terrain neural cohort | Not migrated | Separate original cohort and probe protocol; do not substitute the current terrain cohort |
| Fig. S2: speed neural dynamics | Not migrated | Fresh neural histories and registered logarithmic-summary convention |
| Fig. S3–S4: Adult/larva turning and speed | Not migrated | Public larval input preparation, distinct training/evaluation protocols and focused panels |
| Fig. S5: BANC readout sensitivity | Not migrated | BANC graph preparation, both readouts, six-model initial/final probes |
| Fig. S6: BANC regional recovery | Not migrated | Anatomical regional masks and matched physics/neural/RNG branching |
| Fig. S7: Ant friction robustness | `maze` evaluates the full comparison cohort and exports the specified ten-seed display subset | Full run and supplementary composition |
| Fig. S8: seven-model Large maze | `maze` includes Large adaptation and evaluation | Full 7 × 20 cohort and supplementary composition |
| Fig. S9–S10: Tripod/Hexapod details | Body traces and condition/window metric extraction are implemented | Full cohorts and supplementary panel composition |
| Tables S1–S4 | Policy dimensions/counts are recorded by individual trainers | Unified tables, all supplemental model families and manuscript cross-check |
| Supplementary videos 1–20 | Previously published archive remains at tag v1.0.0 | Fresh rendering/selection workflows are not yet implemented |

## Release criterion

A reviewer release requires every row to have an executable from-scratch recipe, complete fixed cohorts, explicit statistical families, readable final figures/tables and a clean-environment run record. It must not depend on copying old CSVs, trained policies, manually prepared neural histories or local workstation paths.

The current branch provides the core execution system and behavior recipes. The missing rows above remain release blockers. Diagnostic plots are clearly distinguished from the manuscript's finished panels.

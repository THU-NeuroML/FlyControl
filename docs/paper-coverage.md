# Manuscript-to-code map

Reference: author-confirmed **Fly_Locomotion (13).pdf**. The PDF and research archive are not runtime inputs. Paths below are relative to an experiment's run directory. Execute the six stages shown in the README; `run` executes them in order.

**Latest-manuscript check for S1:** **Fly_Locomotion (14).pdf** retains this experiment in the main text (page 6), methods (page 29) and Figure S1 (page 32). Its public preset is `terrain-sensitivity`, covering unit-state and activity-scaled sensitivity in the separate six-model Adult cohort. This check updates the S1 mapping; the other experiments retain the coverage baseline above, except the Ant update below.

**Ant update (2026-10-30):** Figure 5b–d and S7 now use the capacity-matched 20-seed MLP cohort and the [updated statistical protocol](ant-flat.md). Maze results and architectures retain their original cohort.

| Results / claim | Preset and implementation | Generated evidence |
|---|---|---|
| Fig. 1: frozen anatomy, learned interfaces | Graph construction, fixed policy buffers; `materials` | Graph array contracts, initialization records, editable overview |
| Fig. 2b–d: smoothness, crossing and corridor retention | `terrain`; behavior metrics and `paper_statistics` | Episode/seed tables; five primary and 130 secondary slots |
| Fig. 2e: matched ER | `terrain`; independent ER graphs/training; `er_probe` | `analysis/neural/{atlas,er_atlas}.csv`; 45 Adult–ER contrasts |
| Fig. 2f–g: temporal sensitivity and transmission blockade | `terrain`; coarse/dense adjoints, anatomical and matched masks | All horizons/classes, native-gradient checks, six blockade contrasts |
| Fig. 2h–i: example and input perturbations | `terrain`; recorded trajectories, screen and measured planes | 21 × 21 grids; original 11 × 11 subset for held-out tests; video recipe 3 |
| Fig. 3b–d: turning and registered controls | `turning`; separate angle/push cohorts, ER and zero-control evaluation | 144 final models; main, directional and registered comparisons |
| Fig. 3e–h: sensitivity, learned inputs, activity, AN→DN | `turning`; shared histories, E/C exchange, nine activity histories, cuts | `analysis/neural/`; zero-readout enrichment remains undefined |
| Fig. 3i–j: example and gain plane | `turning`; fixed manuscript axes with regenerated exploratory screen | All 48 Adult/MLP planes; recipes 1–2 |
| Fig. 4b–c: tracking and onset response | `speed`; initial PPO and continuation | Eight histories/model, three behavioral endpoints, ramp increments |
| Fig. 4d–g: pathways, holds and associations | `speed`; local response, signed replay, events and associations | Per-class seed values and all 45 associations |
| Fig. 4h–i: example and actor perturbation | `speed`; row/tensor-relative parameter directions | Directly measured KL/action grids and auxiliary directions; recipes 4–5 |
| Fig. 5b–d: Ant flat-ground capacity control | `ant-flat`; fresh Adult and 1378×1378 MLP base/refinement | 20 seed pairs; archived observations or fresh episodes; three new endpoints corrected with the original 26 (29 total) |
| Fig. 5e–j: mazes | `maze`; prerequisite, teacher, router and seven student chains | Original maze policies and 26-test analysis; recipes 7–12 |
| Fig. 6: four bodies and perturbations | Four body presets plus `combine-bodies` | Seven-condition scores, four primary comparisons and full window families |
| S1: unit-state/activity-scaled sensitivity | `terrain-sensitivity`; separate original numerical engine | `analysis/{state_sensitivity.csv.gz,seed_sensitivity.csv}`; dense adjoints and finite-difference checks |
| S2: acceleration/deceleration dynamics | `speed` | Seed-level log2 enrichment; mean(log2) and log2(mean) both explicitly reported |
| S3: Adult/larva control | `development-turning`, `development-terrain`, `combine-development` | Twelve turning plus ten terrain endpoints; exact 924 allocations and Holm over 22 |
| S4: Adult/larva dynamic speed | `development-speed` | 30 profiles × 5 reset seeds/model; paired push controls and 20 contrasts |
| S5: BANC readout-dependent sensitivity | `banc-sensitivity`; BANC v626 and four initial/final E/C combinations | 16 BANC afferent groups, initial/final projections, neural energies and readout decomposition |
| S6: BANC regional recovery | `banc-recovery`; anatomy-weighted message replacement | Same-state branches, zero-force/replay audits, fixed-input probes, seed effects, readout/factorial contrasts and survivor counts |
| S7: friction | `ant-flat` | All 20 seed pairs × four coefficients × 25 resets; descriptive 95% bootstrap intervals |
| S8: Large maze | `maze` | Seven architectures × 20 models × 100 resets |
| S9–S10: Tripod/Hexapod | `tripod`, `hexapod` | Shared Fig. 6 episodes and window metrics, without counting them again as independent data |
| Tables S1–S4 | `materials` | Setting/value/configuration/checksum records from the actual training files |
| Videos 1–20 | `replay` with `configs/videos.json` | Native per-method clips, explicit selection and source/checkpoint checksums |

## Interpretation and completion

This release connects the numerical workflows. Full-budget cohorts have **not** all been rerun: [validation](validation.md) states exactly what was executed. Do not describe smoke plots as replicated manuscript results. The editable panels are deliberately separate files. The manuscript's artistic layout and original selected video montages are not copied into the source tree.

Fixed-input neural effects measure policy response under specified recorded histories. They are distinct from closed-loop behavioral interventions. ER coordinates carry no biological cell identity. Initial zero readouts do not have a defined relative enrichment. BANC summaries remain descriptive; no episode-level significance tests or additive causal percentages are implied. Failure-adjusted heading error includes an explicit π penalty after termination; measured survivor-only heading is separately labelled and counted.

Summative claims about adaptation and cross-body transfer require the corresponding complete cohorts across tasks. A model example, a small local sensitivity or a video cannot establish such a claim alone.

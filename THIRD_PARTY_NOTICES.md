# Sources and third-party material

The authors' controller, training and analysis code is distributed under the root Apache-2.0 license. Numerical routines have been extracted from the research implementation; filesystem orchestration and input preparation have been adapted for this standalone package.

- **FlyGym / NeuroMechFly:** Apache-2.0. The namespaced simulator dependency is under `src/flycontrol/engines/nmf/`, with its license retained there. It includes the CPG and gait interfaces. Required data assets are downloaded at the exact public revision and checksums in `configs/inputs/nmf_assets.json`.
- **DeepMind dm_control:** Apache-2.0. The procedural tripod follows the PyMJCF construction approach. The installed library supplies physics bindings.
- **Gymnasium and Gymnasium-Robotics:** MIT. The Ant/AntMaze tasks use the pinned installed versions. Their task definitions and reset conventions retain their upstream attribution.
- **[Farama-Antmaze-Dataset-Generation](https://github.com/LukaszSawala/Farama-Antmaze-Dataset-Generation):** MIT. The GoalAnt teacher is downloaded at commit `fb253c9efbe2fa5ec256a6fc3f983ba33192d706`. SHA-256: `1b835cc5185756c5af8cd0720dae0e4895efcec48ee577c450d4a5a633fb953e`. It is used only by the Ant distillation stages.
- **[One Policy to Run Them All](https://github.com/nico-bohlinger/one_policy_to_run_them_all):** code is MIT at the pinned revision. Its README excludes robot assets and XML from that license; the download includes upstream notices. Those assets are not relicensed here.
- **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie):** Go1 geometry is downloaded from a fixed revision together with its LICENSE and README. The asset-specific terms apply.
- **FlyWire FAFB v783:** source annotations and connections are downloaded from the official Codex static data service. Dataset use and citation follow FlyWire/Codex terms. Exact files, source URLs and checksums are in `configs/inputs/flywire783.json`.

Scientific results, trained FlyControl weights, research logs and manuscript figure files are not bundled as inputs. Previously published supplementary videos remain available at tag v1.0.0. This code license grants no additional rights to third-party datasets, meshes or videos.

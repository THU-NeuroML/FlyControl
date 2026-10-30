# Architecture and reuse

The public reading path is **figure entry → fixed configuration → six-stage pipeline → task engine → generated analyses and figures**. Numbered scripts only select an experiment and call the CLI. Notebooks call the same CLI with `--smoke`, using the kernel's interpreter. Neither layer implements a second trainer or analysis method.

## Four body experiments

The four bodies already share the graph-policy components, recurrent PPO base, training entry point and pipeline through `engines/common/fly_control/`. During preparation, common code and the selected body's adapter are copied into a run-local source snapshot and checksummed. Workers isolate the imports so one body's `body_control` module cannot leak into another experiment.

| Body | Separate implementation that the experiment requires |
|---|---|
| Cassie | Biped physics, balance curriculum and centered-balance objective |
| Go1 | Quadruped physics, path objective and its registered evaluation-seed cohort |
| Tripod | Three-legged physics and parameter-matched local feedback branches for both Adult and MLP actors |
| Hexapod | Six-legged physics, CPG/oscillator interfaces and evaluation metrics that integrate actual control duration |

These adapters are useful boundaries. In particular, Tripod's policy differences implement a comparison control, and Hexapod's physical-time accounting cannot be replaced by a fixed-step assumption. Full body summaries use one common aggregation command and the joint four-body correction family.

## Remaining duplication

The current engine tree contains 248 Python files. A byte-for-byte audit of files larger than 100 bytes found 43 duplicate groups, representing approximately 0.82 MB of extra copies out of 2.72 MB of engine Python source. This is a measure of source duplication, not a measurement of unnecessary scientific computation.

Most copies are in the historical, supplemental and developmental fly stacks. These stacks preserve different experiment versions; even identical files can resolve imports against different neighboring modules. Blindly replacing them with a single import tree would therefore need separate numerical regression checks.

There is also a small, concrete duplication in the bodies: Cassie and Tripod have identical checkpoint/evaluation hooks. Go1 has a similar hook but a different final evaluation-seed cohort; Hexapod has its own evaluator. A future cleanup can extract the common checkpoint hook with explicit protocol parameters and regression checks for seed selection, checkpoint contents and episode metrics. The present organization change leaves these verified numerical implementations intact.

The architecture is suitable for following and reproducing the experiment protocols. It is not yet a fully deduplicated simulation library. The useful next refactoring target is shared checkpoint bookkeeping, followed by individually verified simulator utilities; merging all body environments would erase meaningful protocol boundaries.

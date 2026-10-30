# Supplementary video recipes

Use `flycontrol replay --run-dir runs/<preset> --video N --output runs/videos/NN` after the corresponding evaluation. The recipe produces labelled native clips for each method and condition, with a source manifest. It does not choose successful episodes after inspecting results or conceal failure terminations. The first configured training seed and first registered evaluation state are used; full cohorts provide the requested condition grids. Some smoke subsets intentionally omit the requested video condition and will be rejected.

| Caption number | Preset | Content | Number in existing v1.0.0 ZIP |
|---|---|---|---|
| 1 | turning | Left turns, −30/−60/−90° | 2 |
| 2 | turning | Right turns, +30/+60/+90° | 3 |
| 3 | terrain | Four obstacle heights | 1 |
| 4 | speed | Ramps | 4 |
| 5 | speed | Sinusoid | 5 |
| 6 | maze | U-Maze, Adult/MLP | 10 |
| 7 | maze | U-Maze, seven architectures | 7 |
| 8 | maze | Medium, Adult/MLP | 11 |
| 9 | maze | Medium, seven architectures | 8 |
| 10 | maze | Large, Adult/MLP | 12 |
| 11 | maze | Large, seven architectures | 9 |
| 12 | maze | Flat Ant | 6 |
| 13 / 14 | cassie | Flat / lateral push | 13 / 14 |
| 15 / 16 | tripod | Flat / lateral push | 15 / 16 |
| 17 / 18 | go1 | Flat / lateral push | 17 / 18 |
| 19 / 20 | hexapod | Flat / lateral push | 19 / 20 |

The confirmed manuscript's running-text references to videos 1–3 disagree with its supplementary captions. The recipes follow the captions and content, as listed above. This source repository does not modify the manuscript.

The existing [v1.0.0 video archive](https://raw.githubusercontent.com/THU-NeuroML/FlyControl/v1.0.0/Supplementary-Videos.zip) remains available with SHA-256 `acf91ab098a66b0af96a547cb10decfa0981b6bfc6c2d82754ba22e670e2e0e3`. It is not a training or analysis input. Original montage timing, cameras and outcome-selected examples need not match fresh training. Native replay exports require `ffmpeg` and a working EGL/OSMesa renderer; see the validation record for exercised rendering paths.

Supplementary Video 6 and the earlier qualitative inset in Figure 5b use the original small-MLP cohort. The new `ant-flat` population statistics and S7 use the capacity-matched MLP; do not interpret the earlier movie as a replay of that model.

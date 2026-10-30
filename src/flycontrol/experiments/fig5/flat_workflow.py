"""Capacity-matched Ant locomotion; independent of the maze dependency chain."""

import numpy as np

from flycontrol.paths import read_json, write_json, sha256
from flycontrol.experiments.fig5.workflow import stage_dir, train_stage


def train(root, spec, method, seed, device):
    import torch
    import yaml
    from body_control.train import build_policy_from_config

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = read_json(root / "protocol.json")
    source = None
    for stage, obs_dim in (("base", 105), ("flat", 109)):
        run = stage_dir(root, method, seed, stage)
        name = "adult" if method == "connectome" else method
        source = train_stage(root, spec, f"{name}_{stage}", run, seed, device,
                             source=source, config_group="ant_flat")
        data = torch.load(source, map_location="cpu", weights_only=False)
        config = yaml.safe_load((run / "config.yaml").read_text())
        policy = build_policy_from_config(config, obs_dim, 8)
        policy.load_state_dict(data["policy_state"], strict=True)
        count = sum(p.numel() for p in policy.actor.parameters() if p.requires_grad)
        if count != protocol["expected_actor_parameters"][method][stage]:
            raise ValueError(f"Unexpected mean-action actor capacity: {method}/{stage}: {count}")
        summary = read_json(run / "summary.json")
        expected_steps = 128 if spec["smoke"] else protocol["actual_training_steps"][stage]
        if summary["completed_timesteps"] != expected_steps:
            raise ValueError(f"Incomplete training budget: {method}/{seed}/{stage}")
        write_json(run / "capacity_audit.json", dict(mean_action_actor_parameters=count,
            excludes="critic, log standard deviations and fixed graph buffers",
            completed_timesteps=summary["completed_timesteps"], checkpoint_sha256=sha256(source)))


def evaluate(root, spec, method, seed, device="cpu"):
    from body_control.evaluate import evaluate_run
    from body_control.motion_quality import evaluate_motion_quality

    protocol = read_json(root / "protocol.json")
    run = stage_dir(root, method, seed, "flat")
    cp = run / "best_checkpoint.pt"
    out = root / f"evaluation/{method}_seed{seed}"
    episodes = 1 if spec["smoke"] else protocol["episodes"]
    common = dict(device=device, checkpoint="best", action_scale=protocol["action_scale"])

    def save(role, records):
        path = out / f"{role}.json"
        write_json(path, records)
        write_json(path.with_suffix(".provenance.json"), dict(checkpoint=str(cp.relative_to(root)),
            checkpoint_sha256=sha256(cp), evaluation_sha256=sha256(path)))

    save("flat", evaluate_run(run, [1.], episodes=episodes, seed=protocol["flat_reset_seed"], **common))
    curves = []

    def capture(*, metrics, terminated, truncated, seed, **kwargs):
        if terminated or truncated:
            positions = np.vstack([metrics.initial_position, metrics.positions])
            displacement = positions[:, :2] - positions[0, :2]
            displacement[:, 1] = np.abs(displacement[:, 1])
            horizon = protocol["horizon_steps"] + 1
            if len(displacement) > horizon:
                raise ValueError("Episode exceeds the native Ant horizon")
            # All resets contribute at all times; hold the final position after termination.
            held = np.pad(displacement, ((0, horizon-len(displacement)), (0, 0)), mode="edge")
            curves.append(held)

    save("flat_motion", evaluate_motion_quality(run, friction=1., episodes=episodes,
        seed=protocol["flat_reset_seed"], frame_callback=capture, **common))
    if len(curves) != episodes:
        raise ValueError("Missing terminated/truncated trajectory")
    mean_curve = np.mean(curves, axis=0)
    save("timecourse", [dict(time_seconds=i*protocol["dt"], forward=float(x), absolute_lateral=float(y),
        episodes=episodes) for i, (x, y) in enumerate(mean_curve)])
    save("friction", evaluate_run(run, protocol["frictions"],
        episodes=1 if spec["smoke"] else protocol["friction_episodes"],
        seed=protocol["friction_reset_seed"], **common))

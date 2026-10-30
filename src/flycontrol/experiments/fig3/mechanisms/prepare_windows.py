"""Prepare the 48 original history windows used by current input-gain probes."""

import json, time
import numpy as np
import torch
from flycontrol.experiments.shared.paths import sha256 as sha
from flycontrol.experiments.fig3.paths import DATA, BANK, checkpoint as cp_path
from turning_control.policy import build_policy

S, I, D = 18756, 118538, 1290
N = S + I + D
STARTS = [8, 20, 32]
NAMES = [
    f"input_{m}_{a:+d}"
    for m in ["connectome", "mlp"]
    for a in [-90, -60, -30, 30, 60, 90]
]


def run(method, seed, device, output):
    started = time.time()
    folder = output
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = cp_path(method, seed)
    checkpoint_hash = sha(ckpt)
    policy = build_policy(
        DATA / "graph.npz", 64, 2, method, seed, device, "normalized", -1.0
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["obs_rms_state"] is None
    policy.load_state_dict(payload["policy_state"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(True)
    names = [
        f"input_{m}_{k:+d}.npz"
        for m in ["connectome", "mlp"]
        for k in [-90, -60, -30, 30, 60, 90]
    ]
    obs = np.stack([np.load(BANK / n)["observations"] for n in names])
    assert obs.shape == (12, 50, 64) and np.isfinite(obs).all()
    observations = torch.as_tensor(obs, device=device)
    starts = [8, 20, 32, 44]
    horizon = 4
    windows = []
    states = []
    manifest = []
    hidden = policy.init_state(12, device)
    with torch.inference_mode():
        for i in range(50):
            if i in starts:
                windows.append(observations[:, i : i + horizon].clone())
                states.append(hidden.clone())
                manifest.extend(
                    dict(
                        history=n,
                        start_index=i,
                        start_time_s=i * 0.04,
                        horizon_steps=horizon,
                    )
                    for n in names
                )
            _, hidden, _ = policy(observations[:, i], hidden)
    windows = torch.cat(windows)
    hidden = torch.cat(states)
    assert len(windows) == 48
    np.savez_compressed(
        folder / "fixed_windows.npz",
        observations=windows.cpu().numpy(),
        initial_hidden=hidden.cpu().numpy(),
    )
    (folder / "window_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = dict(
        method=method,
        seed=seed,
        checkpoint_sha256=checkpoint_hash,
        training_steps=int(payload['timesteps']),
        device=device,
        window_count=48,
        horizon_steps=4,
        clock_ms=40,
        input_bank_sha256={n: sha(BANK / n) for n in names},
        windows_sha256=sha(folder / "fixed_windows.npz"),
        role="Fixed histories used by the current CPG input-gain plane",
    )
    (folder / "COMPLETE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("COMPLETE", method, seed, "48 fixed history windows", flush=True)

"""Panel F: signed class activity under observed versus held target."""

import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from .study import (
    O,
    ANATOMY,
    METHODS,
    CASES,
    checkpoint,
    baseline_path,
    evaluation_path,
    sha,
    write_json,
)

S, I, D = 18756, 118538, 1290
N = S + I + D
PHASES = [
    "deceleration_early",
    "deceleration_middle",
    "acceleration_early",
    "acceleration_middle",
]
from .neurons import Kernel, csr
from .study import event_path


@torch.no_grad()
def run_signed(seed, device):
    dest = O / "signed_replays" / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    start_time = time.time()
    inv = pd.read_csv(ANATOMY / "group_inventory.csv")
    M = csr(sparse.load_npz(ANATOMY / "group_membership.npz"), device)
    counts = torch.as_tensor(inv.n_cells.to_numpy(), device=device)
    data = [np.load(baseline_path(m, seed, CASES[0])) for m in METHODS]
    obs = torch.as_tensor(np.stack([z["observations"] for z in data]), device=device)
    speeds = torch.as_tensor(
        np.stack([z["raw_measured_speed"] for z in data]),
        device=device,
        dtype=torch.float32,
    )
    meta = json.loads(
        evaluation_path("connectome", seed, CASES[0]).with_suffix(".json").read_text()
    )["command"]
    events = [r for r in meta["ramps"] if r["complete_planned"]]
    starts = [int(np.ceil(r["start_s"] / 0.04)) for r in events]
    stages = np.array(
        [
            "acceleration" if r["slope"] > 0 else "deceleration"
            for r in events
            for _ in METHODS
        ]
    )
    k = Kernel(seed, device)
    h = torch.zeros((2, N), device=device)
    stored = {}
    for i in range(max(starts) + 1):
        if i in starts:
            stored[i] = h.clone()
        for _ in range(4):
            h, _, _ = k.step(h, obs[:, i])
    x = torch.cat([obs[:, i : i + 25] for i in starts])
    v = torch.cat([speeds[:, i : i + 25] for i in starts])
    h = torch.cat([stored[i] for i in starts]).repeat(3, 1)
    onset = torch.as_tensor(
        [r["start_speed"] for r in events for _ in METHODS], device=device
    )[:, None].expand(-1, 25)
    held = x.clone()
    held[:, :, 46] = onset / 20
    held[:, :, 48] = torch.clamp((onset - v) / 20, -1, 1)
    speed_held = x.clone()
    speed_held[:, :, 47] = torch.clamp(v[:, :1] / 20, -1, 1)
    speed_held[:, :, 48] = torch.clamp(x[:, :, 46] - v[:, :1] / 20, -1, 1)
    neural = []
    actions = []
    err = 0.0
    for t in range(25):
        batch = torch.cat([x[:, t], held[:, t], speed_held[:, t]])
        for _ in range(4):
            h, _, _ = k.step(h, batch)
        rates = h.tanh().reshape(3, 8, N)
        act = k.output(h).reshape(3, 8, 2)
        truth = np.stack([np.tanh(data[0]["gaussian_actions"][s + t]) for s in starts])
        err = max(err, float(abs(act[0, 0::2].cpu().numpy() - truth).max()))
        for signal, ci in [("target", 1), ("measured", 2)]:
            delta = rates[0] - rates[ci]
            measures = []
            for values in [
                delta,
                delta.clamp_min(0),
                delta.clamp_max(0),
                delta.square(),
            ]:
                measures.append(torch.sparse.mm(M, values.T.contiguous()).T / counts)
            vals = torch.stack(measures, 2)
            for phase in ["acceleration", "deceleration"]:
                mask = stages == phase
                neural.append(vals[mask].mean(0).cpu().numpy())
                actions.append(float((act[0, mask, 0] - act[ci, mask, 0]).mean()))
    out = np.stack(neural).reshape(25, 2, 2, len(inv), 4)
    reference = np.load(event_path(seed))["values"][..., 0]
    energy_error = float(abs(out[..., 3] - reference).max())
    assert err < 3e-5 and energy_error < 2e-7, (err, energy_error)
    assert np.allclose(out[..., 0], out[..., 1] + out[..., 2], atol=1e-7)
    np.savez_compressed(
        dest / "signed_neural.npz",
        values=out,
        drive=np.array(actions).reshape(25, 2, 2),
        time_ms=np.arange(1, 26) * 40,
        signals=["target", "measured"],
        phases=["acceleration", "deceleration"],
        metrics=[
            "mean_signed_delta",
            "mean_positive_delta",
            "mean_negative_delta",
            "mean_squared_delta",
        ],
        group_ids=inv.group_id.to_numpy(str),
    )
    write_json(
        dest / "COMPLETE.json",
        dict(
            seed=seed,
            seconds=time.time() - start_time,
            native_baseline_max_error=err,
            original_squared_energy_max_error=energy_error,
            checkpoint_sha256=sha(checkpoint("connectome", seed)),
            output_sha256=sha(dest / "signed_neural.npz"),
            definition="tanh(state_actual)-tanh(state_held), mean across neurons in group, two events and two matched histories per phase; all 20 current Adult models",
        ),
    )
    print("SIGNED COMPLETE", seed, round(time.time() - start_time, 1), flush=True)

"""Panel G: observed versus class-specific target-hold responses."""

import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from flycontrol.experiments.fig4.mechanisms.study import O, ANATOMY, NODE_SOURCE, METHODS, CASES, checkpoint, load_policy, baseline_path, evaluation_path, sha, write_json

S, I, D = 18756, 118538, 1290
N = S + I + D
PHASES = [
    "deceleration_early",
    "deceleration_middle",
    "acceleration_early",
    "acceleration_middle",
]
from flycontrol.experiments.fig4.mechanisms.neurons import Kernel, csr


@torch.no_grad()
def event_run(seed, device):
    begin = time.time()
    dest = O / "events" / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    inv = pd.read_csv(ANATOMY / "group_inventory.csv")
    M = csr(sparse.load_npz(ANATOMY / "group_membership.npz"), device)
    nodes = pd.read_csv(NODE_SOURCE).fillna("unassigned")
    labels = inv[inv.level == "input_class"].label.tolist()
    classes = {
        name: np.flatnonzero(nodes.refined_group.iloc[:S].eq(name)) for name in labels
    }
    names = ["baseline", "hold_target", "hold_measured"] + [
        "hold_target_" + name for name in labels
    ]
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
    rows = []
    neural = []
    validation = {}
    for method in METHODS:
        p = load_policy(method, seed, device) if method == "mlp" else None
        k = Kernel(seed, device) if method == "connectome" else None
        h = torch.zeros((2, N), device=device) if k else p.init_state(2, device)
        stored = {}
        for i in range(max(starts) + 1):
            if i in starts:
                stored[i] = h.clone()
            if k:
                for _ in range(4):
                    h, _, _ = k.step(h, obs[:, i])
            else:
                _, h, _ = p(obs[:, i], h)
        x = torch.cat([obs[:, i : i + 25] for i in starts])
        v = torch.cat([speeds[:, i : i + 25] for i in starts])
        h = torch.cat([stored[i] for i in starts])
        onset = torch.as_tensor(
            [r["start_speed"] for r in events for _ in METHODS], device=device
        )[:, None].expand(-1, 25)
        hold_target = x.clone()
        hold_target[:, :, 46] = onset / 20
        hold_target[:, :, 48] = torch.clamp((onset - v) / 20, -1, 1)
        hold_measured = x.clone()
        hold_measured[:, :, 47] = torch.clamp(v[:, :1] / 20, -1, 1)
        hold_measured[:, :, 48] = torch.clamp(x[:, :, 46] - v[:, :1] / 20, -1, 1)
        conditions = names if k else names[:3]
        h = h.repeat(len(conditions), 1)
        for step in range(25):
            if k:
                baseline_external = x[:, step] @ k.E.T + k.eb
                target_external = hold_target[:, step] @ k.E.T + k.eb
                speed_external = hold_measured[:, step] @ k.E.T + k.eb
                externals = [baseline_external, target_external, speed_external]
                for label in labels:
                    e = baseline_external.clone()
                    e[:, classes[label]] = target_external[:, classes[label]]
                    externals.append(e)
                ext = torch.cat(externals)
                for substep in range(4):
                    recurrent = torch.sparse.mm(k.W, h.tanh().T.contiguous()).T
                    recurrent[:, :S] += ext
                    h = 0.5 * h + 0.5 * recurrent.tanh()
                action = k.output(h).reshape(len(conditions), 8, 2)
                rates = h.tanh().reshape(len(conditions), 8, N)
                for signal, ci in [("target", 1), ("measured", 2)]:
                    delta = (rates[0] - rates[ci]).square()
                    energy = torch.sparse.mm(M, delta.T.contiguous()).T
                    den = torch.stack(
                        [
                            delta[:, :S].sum(1),
                            delta[:, S : S + I].sum(1),
                            delta[:, -D:].sum(1),
                        ],
                        1,
                    )
                    part = torch.as_tensor(
                        inv.partition.map(
                            {"input": 0, "internal": 1, "output": 2}
                        ).to_numpy(),
                        device=device,
                    )
                    fractions = energy / den[:, part].clamp_min(1e-30)
                    en = fractions * torch.as_tensor(
                        (inv.denominator_cells / inv.n_cells).to_numpy(), device=device
                    )
                    for phase in ["acceleration", "deceleration"]:
                        mask = stages == phase
                        values = (
                            torch.stack(
                                [
                                    energy
                                    / torch.as_tensor(
                                        inv.n_cells.to_numpy(), device=device
                                    ),
                                    fractions,
                                    en,
                                ],
                                2,
                            )[mask]
                            .mean(0)
                            .cpu()
                            .numpy()
                        )
                        neural.append(values)
            else:
                mu, h, _ = p(
                    torch.cat(
                        [x[:, step], hold_target[:, step], hold_measured[:, step]]
                    ),
                    h,
                )
                action = mu.tanh().reshape(3, 8, 2)
            # Baseline must match archived actions for the architecture's own history.
            own = 0 if method == "connectome" else 1
            truth = np.stack(
                [
                    np.tanh(data[own]["gaussian_actions"][start + step])
                    for start in starts
                ]
            )
            err = float(abs(action[0, own::2].cpu().numpy() - truth).max())
            validation[method] = max(validation.get(method, 0), err)
            for ci, condition in enumerate(conditions):
                for phase in ["acceleration", "deceleration"]:
                    mask = stages == phase
                    delta = action[0, mask, 0] - action[ci, mask, 0]
                    rows.append(
                        dict(
                            seed=seed,
                            method=method,
                            phase=phase,
                            condition=condition,
                            time_ms=(step + 1) * 40,
                            baseline_drive=float(action[0, mask, 0].mean()),
                            counterfactual_drive=float(action[ci, mask, 0].mean()),
                            signed_drive_change=float(delta.mean()),
                            rms_drive_change=float(delta.square().mean().sqrt()),
                        )
                    )
        del k, p, h
    assert max(validation.values()) < 3e-5, validation
    pd.DataFrame(rows).to_csv(dest / "drive_responses.csv", index=False)
    arr = np.stack(neural).reshape(25, 2, 2, len(inv), 3)
    np.savez_compressed(
        dest / "neural_responses.npz",
        values=arr,
        time_ms=np.arange(1, 26) * 40,
        signals=np.array(["target", "measured"]),
        phases=np.array(["acceleration", "deceleration"]),
        group_ids=inv.group_id.to_numpy(str),
    )
    write_json(
        dest / "COMPLETE.json",
        dict(
            seed=seed,
            training_decisions=15872,
            events=4,
            seconds=time.time() - begin,
            native_baseline_errors=validation,
            definition="Observed ramp inputs versus counterfactual onset-target/onset-measured-speed holds; all other observations held to the same actual history. Category holds replace only input projected into that anatomical class.",
            checkpoint_hashes={m: sha(checkpoint(m, seed)) for m in METHODS},
            outputs_sha256={
                f.name: sha(f) for f in dest.iterdir() if f.name != "COMPLETE.json"
            },
        ),
    )
    print("COMPLETE EVENTS", seed, round(time.time() - begin, 1), flush=True)

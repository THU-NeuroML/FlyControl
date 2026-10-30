"""Panel E: phase-resolved local drive sensitivity in the frozen neural core."""

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
    load_policy,
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


def csr(a, device):
    a = a.tocsr()
    a.sort_indices()
    return torch.sparse_csr_tensor(
        torch.as_tensor(a.indptr.astype(np.int64)),
        torch.as_tensor(a.indices.astype(np.int64)),
        torch.as_tensor(a.data),
        size=a.shape,
        device=device,
    )


class Kernel:
    def __init__(self, seed, device):
        self.seed = seed
        self.device = device
        self.p = load_policy("connectome", seed, device)
        c = self.p.actor.core
        for par in self.p.parameters():
            par.requires_grad_(False)
        self.c = c
        self.E = c.input_proj.weight.detach()
        self.eb = c.input_proj.bias.detach()
        a = sparse.csr_matrix(
            (
                c.edge_weight.cpu().numpy(),
                (c.edge_post.cpu().numpy(), c.edge_pre.cpu().numpy()),
            ),
            shape=(N, N),
        )
        a.sort_indices()
        self.W = csr(a, device)
        self.WT = csr(a.T, device)
        self.gate = 1 - (1 - torch.clamp(c.dt / c.rate_tau, 0, 1)) ** (
            1 / c.internal_substeps
        )
        assert torch.max(abs(self.gate - 0.5)) < 1e-6
        assert (
            float(c._recurrent_gain) == 1
            and float(c._external_input_gain) == 1
            and float(c._neural_readout_gain) == 1
        )

    def output(self, h):
        return (3 * torch.tanh(self.c.output_layer(h[:, -D:].tanh()) / 3)).tanh()

    @torch.no_grad()
    def step(self, h, x):
        r = h.tanh()
        v = torch.sparse.mm(self.W, r.T.contiguous()).T
        v[:, :S] += x @ self.E.T + self.eb
        target = v.tanh()
        return (
            (1 - self.gate) * h + self.gate * target,
            1 - r * r,
            self.gate * (1 - target * target),
        )

    def readout_grad(self, h):
        with torch.enable_grad():
            hd = h.detach().requires_grad_(True)
            g = torch.autograd.grad(self.output(hd)[:, 0].sum(), hd)[0]
        return g.detach()


def run(seed, device):
    start = time.time()
    dest = O / "neurons" / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    inv = pd.read_csv(ANATOMY / "group_inventory.csv")
    M = csr(sparse.load_npz(ANATOMY / "group_membership.npz"), device)
    part_ids = torch.as_tensor(
        inv.partition.map({"input": 0, "internal": 1, "output": 2}).to_numpy(),
        device=device,
    )
    counts = torch.as_tensor(inv.n_cells.to_numpy(), device=device)
    den_counts = torch.as_tensor(inv.denominator_cells.to_numpy(), device=device)
    k = Kernel(seed, device)
    z = [np.load(baseline_path(m, seed, CASES[0])) for m in METHODS]
    obs = torch.as_tensor(np.stack([x["observations"] for x in z]), device=device)
    meta = json.loads(
        evaluation_path("connectome", seed, CASES[0]).with_suffix(".json").read_text()
    )["command"]
    events = []
    for ei, event in enumerate([r for r in meta["ramps"] if r["complete_planned"]]):
        direction = "acceleration" if event["slope"] > 0 else "deceleration"
        for phase, t in [
            ("early", event["start_s"] + 0.08),
            ("middle", (event["start_s"] + event["end_s"]) / 2),
        ]:
            index = int(np.ceil(t / 0.04))
            assert index + 8 <= 300
            events.append(
                dict(
                    event=ei,
                    phase=direction + "_" + phase,
                    index=index,
                    time_s=index * 0.04,
                    slope=event["slope"],
                )
            )
    hs = {}
    h = torch.zeros((2, N), device=device)
    replay = []
    with torch.no_grad():
        for t in range(300):
            if t in [r["index"] for r in events]:
                hs[t] = h.clone()
            for _ in range(4):
                h, _, _ = k.step(h, obs[:, t])
            replay.append(k.output(h).cpu().numpy())
    replay = np.stack(replay, 1)
    replay_error = float(abs(replay[0] - np.tanh(z[0]["gaussian_actions"])).max())
    assert replay_error < 3e-5, replay_error
    states = torch.cat([hs[r["index"]] for r in events])
    windows = torch.cat([obs[:, r["index"] : r["index"] + 8] for r in events])
    phase_index = np.array([PHASES.index(r["phase"]) for r in events for _ in METHODS])
    batch = len(states)
    write_json(dest / "EVENTS.json", events)
    # Validate the manually propagated adjoint against native autograd for each checkpoint.
    h0 = states[:1].clone().requires_grad_(True)
    x0 = windows[:1, 0]
    mu, hn, _ = k.p(x0, h0)
    native = torch.autograd.grad(mu[:, 0].tanh().sum(), h0)[0]
    h = states.clone()
    R = []
    B = []
    H = [h]
    for t in range(32):
        h, r, b = k.step(h, windows[:, t // 4])
        R.append(r)
        B.append(b)
        H.append(h)
    lam = k.readout_grad(H[4])
    with torch.no_grad():
        for t in range(3, -1, -1):
            lam = (
                0.5 * lam + torch.sparse.mm(k.WT, (lam * B[t]).T.contiguous()).T * R[t]
            )
    gradient_error = float((lam[:1] - native).norm() / native.norm().clamp_min(1e-20))
    assert gradient_error < 2e-4, gradient_error
    # Independent native 320-ms input-gradient check of the pulse-routing kernel.
    xx = windows[:1].clone().requires_grad_(True)
    hh = states[:1].detach()
    for step in range(8):
        mu, hh, _ = k.p(xx[:, step], hh)
    native_input = torch.autograd.grad(mu[:, 0].tanh().sum(), xx)[0].detach()
    lam = k.readout_grad(H[32][:1])
    manual_input = torch.zeros_like(xx)
    with torch.no_grad():
        for t in reversed(range(32)):
            gd = lam * B[t][:1]
            manual_input[:, t // 4] += gd[:, :S] @ k.E
            lam = 0.5 * lam + torch.sparse.mm(k.WT, gd.T.contiguous()).T * R[t][:1]
    pulse_gradient_error = float(
        (manual_input - native_input).norm() / native_input.norm().clamp_min(1e-20)
    )
    assert pulse_gradient_error < 2e-4, pulse_gradient_error

    def measure(values):
        energy = torch.sparse.mm(M, values.T.contiguous()).T
        den = torch.stack(
            [values[:, :S].sum(1), values[:, S : S + I].sum(1), values[:, -D:].sum(1)],
            1,
        )[:, part_ids]
        fractions = torch.where(den > 0, energy / den, torch.zeros_like(energy))
        arrays = torch.stack(
            [energy / counts, fractions, fractions * den_counts / counts], 2
        )
        return np.stack(
            [arrays[phase_index == pi].mean(0).cpu().numpy() for pi in range(4)]
        )

    local = []
    activity = []
    for horizon in range(1, 33):
        lam = k.readout_grad(H[horizon])
        with torch.no_grad():
            for t in reversed(range(horizon)):
                lam = (
                    0.5 * lam
                    + torch.sparse.mm(k.WT, (lam * B[t]).T.contiguous()).T * R[t]
                )
            local.append(measure(lam.square()))
            activity.append(measure((H[horizon].tanh() - states.tanh()).square()))
    np.savez_compressed(
        dest / "atlas.npz",
        local=np.stack(local),
        activity=np.stack(activity),
        phases=np.array(PHASES),
        time_ms=np.arange(1, 33) * 10,
        group_ids=inv.group_id.to_numpy(str),
    )
    write_json(
        dest / "COMPLETE.json",
        dict(
            seed=seed,
            training_decisions=15872,
            checkpoint_sha256=sha(checkpoint("connectome", seed)),
            groups=len(inv),
            events=8,
            windows=batch,
            model_horizon_ms=list(range(10, 321, 10)),
            phase_names=PHASES,
            native_replay_max_abs_error=replay_error,
            native_adjoint_relative_error=gradient_error,
            native_320ms_input_gradient_relative_error=pulse_gradient_error,
            script_sha256=sha(__file__),
            seconds=time.time() - start,
            outputs_sha256={
                f.name: sha(f) for f in dest.iterdir() if f.name != "COMPLETE.json"
            },
        ),
    )
    print("COMPLETE NEURONS", seed, round(time.time() - start, 1), flush=True)

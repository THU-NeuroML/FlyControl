"""Local common-history input-gain response; frozen checkpoint replay on CPU."""

import os

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"

from pathlib import Path
import sys, json, time, argparse, types, hashlib
import numpy as np
import pandas as pd
import torch
from flycontrol.experiments.shared.paths import ROOT as REPO
from flycontrol.experiments.fig2.runtime import activate

activate()
from terrain_control.context import configure_process

configure_process()
from terrain_control.policy import build_policy
from flycontrol.experiments.fig2.current.paths import DATA, OUTPUT as ROOT

GRID = json.loads((REPO / "evaluation/grid.json").read_text())
HEIGHTS = GRID["heights"]
STUDY = json.loads((REPO / "configs/figure2/study.json").read_text())
from flycontrol.mechanisms import terrain_history_phases
PHASES = terrain_history_phases(GRID, STUDY)
BLOCKS = {
    "Feet": list(range(0, 18)),
    "Contact": list(range(18, 24)) + list(range(55, 73)),
    "Orientation": list(range(24, 27)) + list(range(52, 55)),
    "Height": [27],
    "CPG": list(range(28, 46)),
    "Velocity": list(range(46, 52)),
    "Previous action": list(range(73, 79)),
    "Joint angle": list(range(79, 121)),
    "Joint velocity": list(range(121, 163)),
    "Task state": list(range(163, 172)),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_policy(method, seed):
    m = json.loads((DATA / "evaluation_protocol.json").read_text())
    entry = next(
        c for c in m["checkpoints"] if c["method"] == method and c["seed"] == seed
    )
    ck = torch.load(REPO / entry["path"], map_location="cpu", weights_only=False)
    assert ck.get("obs_rms_state") is None
    assert sha(REPO / entry["path"]) == entry["sha256"]
    policy = build_policy(
        DATA / "graph.npz", 688, 6, method, seed, "cpu", "normalized", -1.0
    )
    policy.load_state_dict(ck["policy_state"])
    policy.eval()
    return policy, entry


def install_csr(policy):
    c = policy.actor.core
    g = c.fixed_graph
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), g.f_lengths.cumsum(0)])
    matrix = torch.sparse_csr_tensor(ptr, g.f_source, g.f_weight, size=(c.N, c.N))

    def sparse(self, x):
        return torch.sparse.mm(matrix, x.T).T.contiguous()

    c.recurrent_drive = types.MethodType(sparse, c)


def histories(seed):
    xs = []
    meta = []
    for h in HEIGHTS:
        for phase in PHASES:
            for origin in ["connectome", "mlp"]:
                p = (
                    DATA
                    / "evaluation"
                    / f"{origin}_seed{seed}"
                    / f"h{h:g}_phase{phase}"
                    / "trace.npz"
                )
                with np.load(p) as z:
                    times = z["time_s"]
                    indices = [
                        int(np.argmin(abs(times - t))) for t in np.arange(6) * 0.04
                    ]
                    assert np.max(abs(times[indices] - np.arange(6) * 0.04)) < 1e-8
                    xs.append(z["observation"][indices])
                meta.append(
                    dict(
                        height=h,
                        phase=phase,
                        origin=origin,
                        source=str(p),
                        sha256=sha(p),
                    )
                )
    return torch.as_tensor(np.stack(xs), dtype=torch.float32), meta


class Probe:
    @torch.inference_mode()
    def __init__(self, method, seed, benchmark=False):
        tic = time.time()
        self.policy, self.checkpoint = load_policy(method, seed)
        self.method = method
        self.seed = seed
        self.x, self.meta = histories(seed)
        self.n = len(self.x)
        self.heights = np.array([r["height"] for r in self.meta])
        self.native_error = 0.0
        if method == "connectome":
            # Validate against the exact archived deterministic reduction before using CSR.
            sample = self.x[:2]
            state = self.policy.init_state(2, "cpu")
            native = []
            states = []
            for t in range(6):
                v, state, _ = self.policy(sample[:, t], state)
                native.append(v.clone())
                states.append(state.clone())
            install_csr(self.policy)
            state = self.policy.init_state(2, "cpu")
            errors = []
            for t in range(6):
                v, state, _ = self.policy(sample[:, t], state)
                errors.extend(
                    [
                        float((v - native[t]).abs().max()),
                        float((state - states[t]).abs().max()),
                    ]
                )
            self.native_error = max(errors)
            assert self.native_error < 2e-5, self.native_error
        state = self.policy.init_state(self.n, "cpu")
        for t in range(4):
            _, state, _ = self.policy(self.x[:, t], state)
        self.hidden = state.clone()
        self.windows = self.x[:, 4:6].clone()
        self.ref = self.forward(self.windows)
        self.var = self.policy.log_std.detach().exp().square()[None, None, :]
        self.preparation_s = time.time() - tic

    @torch.inference_mode()
    def forward(self, x):
        h = self.hidden.clone()
        out = []
        for t in range(x.shape[1]):
            v, h, _ = self.policy(x[:, t], h)
            out.append(v)
        return torch.stack(out, 1)

    @torch.inference_mode()
    def evaluate(self, changes):
        x = self.windows.clone().reshape(self.n, 2, 4, 172)
        for axis, dose in changes.items():
            x[:, :, :, BLOCKS[axis]] *= 1 + float(dose)
        y = self.forward(x.reshape(self.n, 2, 688))
        sq = (y - self.ref).square()
        values = (
            torch.stack(
                [
                    (0.5 * sq / self.var).sum(-1).mean(1),
                    (0.5 * sq).sum(-1).mean(1),
                    (y.tanh() - self.ref.tanh()).square().mean((1, 2)),
                ],
                1,
            )
            .cpu()
            .numpy()
        )
        out = []
        for h in HEIGHTS:
            v = values[self.heights == h].mean(0)
            out.append(
                dict(
                    height=h,
                    policy_kl=float(v[0]),
                    unit_kl=float(v[1]),
                    action_mse=float(v[2]),
                )
            )
        return out

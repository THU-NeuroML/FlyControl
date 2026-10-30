"""Exact 10-ms temporal sampling of all20 current Adult models; GPU adjoint."""

import os

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import sys, json, time, gc
import numpy as np
import pandas as pd
import torch
from .paths import DATA, OUTPUT as R
from .probe import Probe, sha

D = torch.device(os.environ.get("FLYCONTROL_DEVICE", "cpu"))
torch.backends.cuda.matmul.allow_tf32 = False
ANN = pd.read_csv(DATA / "anatomy/current_graph_annotations.csv")
GROUPS = {
    str(k): np.flatnonzero(ANN.super_class.eq(k)) for k in ANN.super_class.unique()
}
GROUPS["mechanosensory"] = np.flatnonzero(ANN["class"].eq("mechanosensory"))


def run(seed):
    out = R / "dense_neurons" / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        return
    tic = time.time()
    p = Probe("connectome", seed)
    c = p.policy.actor.core
    N, S, I = c.N, c.S, c.I
    g = c.fixed_graph
    n = p.n
    W = torch.sparse_csr_tensor(
        torch.cat([torch.zeros(1, dtype=torch.long), g.f_lengths.cumsum(0)]).to(D),
        g.f_source.to(D),
        g.f_weight.to(D),
        size=(N, N),
    )
    WT = W.transpose(0, 1).to_sparse_csr()
    # Copy only trainable projections; no CPU-inference tensor enters autograd.
    inp = torch.nn.Linear(172, S).to(D)
    inp.load_state_dict(c.input_proj.state_dict())
    readout = torch.nn.Sequential(
        torch.nn.LayerNorm(N - S - I, eps=1e-5), torch.nn.Linear(N - S - I, 6)
    ).to(D)
    readout.load_state_dict(c.output_layer.state_dict())
    for layer in [inp, readout]:
        for par in layer.parameters():
            par.requires_grad_(False)
    windows = (
        torch.cat([p.windows, p.windows[:, -1:].repeat(1, 6, 1)], 1)
        .reshape(n, 32, 172)
        .to(D)
    )
    h = p.hidden.to(D)
    rs = []
    bs = []
    states = []
    with torch.no_grad():
        for t in range(32):
            rr = h.tanh()
            v = torch.sparse.mm(W, rr.T.contiguous()).T
            v[:, :S] += inp(windows[:, t])
            u = v.tanh()
            rs.append(1 - rr.square())
            bs.append(0.5 * (1 - u.square()))
            h = 0.5 * h + 0.5 * u
            states.append(h[:, -(N - S - I) :].clone())
        mu = 3 * torch.tanh(readout(states[3].tanh()) / 3)
        forward_error = float((mu.cpu() - p.ref[:, 0]).abs().max())
        assert forward_error < 2e-5, forward_error
    ids = {group: torch.as_tensor(v, device=D) for group, v in GROUPS.items()}
    rows = []
    for horizon in range(1, 33):
        tail = states[horizon - 1].detach().requires_grad_(True)
        act = (3 * torch.tanh(readout(tail.tanh()) / 3)).tanh()
        lam = torch.zeros(6, n, N, device=D)
        for ch in range(6):
            lam[ch, :, -(N - S - I) :] = torch.autograd.grad(
                act[:, ch].sum(), tail, retain_graph=ch < 5
            )[0].detach()
        with torch.no_grad():
            for t in range(horizon - 1, -1, -1):
                gd = lam * bs[t][None]
                back = torch.sparse.mm(
                    WT, gd.reshape(6 * n, N).T.contiguous()
                ).T.reshape(6, n, N)
                lam = 0.5 * lam + back * rs[t][None]
            energy = lam.square().sum(0)
            total = energy.sum(1)
            assert bool((total > 0).all())
            for group, jj in ids.items():
                part = 0 if int(jj[0]) < S else 1 if int(jj[0]) < S + I else 2
                lo, hi = [(0, S), (S, S + I), (S + I, N)][part]
                fraction = energy[:, jj].sum(1) / total
                within = energy[:, jj].sum(1) / energy[:, lo:hi].sum(1)
                for height in np.unique(p.heights):
                    mask = torch.as_tensor(p.heights == height, device=D)
                    rows.append(
                        dict(
                            seed=seed,
                            horizon_ms=horizon * 10,
                            height=height,
                            group=group,
                            n_cells=len(jj),
                            population_enrichment=float(
                                (fraction[mask] * N / len(jj)).mean()
                            ),
                            partition_enrichment=float(
                                (within[mask] * (hi - lo) / len(jj)).mean()
                            ),
                        )
                    )
        pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
        print(seed, horizon * 10, round(time.time() - tic, 1), flush=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(
        DATA / "mechanisms/coarse_neurons" / str(seed) / "sensitivity.csv"
    )
    merged = old.merge(
        new,
        on=["seed", "horizon_ms", "height", "group"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    assert len(merged) == len(old)
    a = merged.population_enrichment_old.to_numpy()
    b = merged.population_enrichment_new.to_numpy()
    err = float(np.max(abs(a - b) / np.maximum(abs(a), 1e-12)))
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                checkpoint=p.checkpoint,
                histories=p.meta,
                forward_error=forward_error,
                old_timepoint_max_relative_error=err,
                horizons_ms=list(range(10, 321, 10)),
                all_groups=11,
                seconds=time.time() - tic,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                script_sha256=sha(__file__),
            ),
            indent=2,
        )
    )
    del p, lam, energy, states, rs, bs, W, WT, inp, readout
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    (R / "DENSE_TIME_PROTOCOL.json").write_text(
        json.dumps(
            dict(
                cohort="All20 current Adult models, same80histories and initial states as v10",
                times_ms=list(range(10, 321, 10)),
                method="Exact reverse-mode derivative at each of32 horizons, six output channels batched; no interpolated measurements",
                validation="All five original timepoints across all11 groups×5heights checked against CPU adjoint; native forward tolerance2e-5",
                input_extension="Actual first80ms then repeat final observed input window, same as original",
                inference="Descriptive 20-seed bootstrap bands; original five-time screen statistics retained separately; no added pointwise significance claims",
            ),
            indent=2,
        )
    )
    for seed in (
        [int(v) for v in sys.argv[1].split(",")]
        if len(sys.argv) > 1
        else range(2011, 2031)
    ):
        run(seed)

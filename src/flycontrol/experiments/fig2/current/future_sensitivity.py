"""Future-output Jacobian audit of all twenty frozen terrain Adult policies.

Same derivative concept as the speed figure, adapted to the current input
partition and all six residual outputs. No cell-class homology is assigned to MLP.
"""

import os

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "1"
from pathlib import Path
import sys, json, time, subprocess, concurrent.futures
import numpy as np
import pandas as pd
import torch
from flycontrol.experiments.fig2.current.paths import DATA, OUTPUT as ROOT
from flycontrol.experiments.fig2.current.probe import Probe, sha

HORIZONS = [1, 4, 8, 16, 32]
ANN = pd.read_csv(DATA / "anatomy/current_graph_annotations.csv")


def run(seed):
    dest = ROOT / "coarse_neurons" / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    started = time.time()
    p = Probe("connectome", seed)
    # Probe constructs an inference-only policy. Make ordinary detached copies
    # before the independent autograd validation; parameter values are unchanged.
    for module in p.policy.modules():
        for name, value in module._buffers.items():
            if value is not None:
                module._buffers[name] = value.detach().clone()
        for name, value in module._parameters.items():
            if value is not None:
                module._parameters[name] = torch.nn.Parameter(
                    value.detach().clone(), requires_grad=False
                )
    c = p.policy.actor.core
    for par in p.policy.parameters():
        par.requires_grad_(False)
    n, N = p.n, c.N
    S = c.S
    I = c.I
    assert (S, I, N) == (2279, 135015, 138584)
    g = c.fixed_graph
    WT = torch.sparse_csr_tensor(
        torch.cat([torch.zeros(1, dtype=torch.long), g.b_lengths.cumsum(0)]),
        g.b_source,
        g.b_weight,
        size=(N, N),
    )
    windows = (
        torch.cat([p.windows, p.windows[:, -1:].repeat(1, 6, 1)], 1)
        .reshape(n, 32, 172)
        .clone()
    )
    state = p.hidden.clone()
    rs = []
    bs = []
    hs = {}
    with torch.no_grad():
        for t in range(32):
            r = state.tanh()
            v = c.recurrent_drive(r)
            v[:, :S] += c.input_proj(windows[:, t])
            target = v.tanh()
            rs.append(1 - r * r)
            bs.append(0.5 * (1 - target * target))
            state = 0.5 * state + 0.5 * target
            if t + 1 in HORIZONS:
                hs[t + 1] = state.clone()
        raw = c.output_layer(hs[8][:, S + I :].tanh())
        mu = 3 * torch.tanh(raw / 3)
        forward_error = float((mu - p.ref[:, 1]).abs().max())
        assert forward_error < 2e-5
    # Independent automatic derivative through the native archived reducer.
    from terrain_control.policy import FixedRateCore

    saved = c.recurrent_drive
    c.recurrent_drive = FixedRateCore.recurrent_drive.__get__(c, type(c))
    h0 = p.hidden[:1].clone().requires_grad_(True)
    mean, _, _ = p.policy(p.windows[:1, 0].clone(), h0)
    native = torch.autograd.grad(mean[:, 0].tanh().sum(), h0)[0].detach()
    c.recurrent_drive = saved
    groups = {
        str(k): np.flatnonzero(ANN.super_class.eq(k)) for k in ANN.super_class.unique()
    }
    groups["mechanosensory"] = np.flatnonzero(ANN["class"].eq("mechanosensory"))
    rows = []
    gradient_error = None
    for horizon in HORIZONS:
        energy = torch.zeros((n, N))
        h = hs[horizon].clone().requires_grad_(True)
        act = (3 * torch.tanh(c.output_layer(h[:, S + I :].tanh()) / 3)).tanh()
        for ch in range(6):
            lam = torch.autograd.grad(act[:, ch].sum(), h, retain_graph=ch < 5)[
                0
            ].detach()
            with torch.no_grad():
                for t in range(horizon - 1, -1, -1):
                    gd = lam * bs[t]
                    lam = 0.5 * lam + torch.sparse.mm(WT, gd.T.contiguous()).T * rs[t]
                if horizon == 4 and ch == 0:
                    gradient_error = float(
                        (lam[:1] - native).norm() / native.norm().clamp_min(1e-20)
                    )
                    assert gradient_error < 2e-4, gradient_error
                energy += lam.square()
        with torch.no_grad():
            population = energy.sum(1)
            for group, ids in groups.items():
                assert len(ids) > 0
                parts = np.where(ids < S, 0, np.where(ids < S + I, 1, 2))
                assert len(np.unique(parts)) == 1
                part = int(parts[0])
                lo, hi = [(0, S), (S, S + I), (S + I, N)][part]
                group_energy = energy[:, ids].sum(1)
                denom = energy[:, lo:hi].sum(1)
                fraction = group_energy / population.clamp_min(1e-30)
                within = group_energy / denom.clamp_min(1e-30)
                for height in np.unique(p.heights):
                    mask = p.heights == height
                    rows.append(
                        dict(
                            seed=seed,
                            horizon_ms=horizon * 10,
                            height=height,
                            group=group,
                            n_cells=len(ids),
                            partition=["input", "internal", "output"][part],
                            partition_cells=hi - lo,
                            squared_derivative_per_cell=float(
                                (group_energy[mask] / len(ids)).mean()
                            ),
                            population_fraction=float(fraction[mask].mean()),
                            population_enrichment=float(
                                (fraction[mask] * N / len(ids)).mean()
                            ),
                            partition_fraction=float(within[mask].mean()),
                            partition_enrichment=float(
                                (within[mask] * (hi - lo) / len(ids)).mean()
                            ),
                        )
                    )
        pd.DataFrame(rows).to_csv(dest / "sensitivity.csv", index=False)
        print(seed, horizon * 10, round(time.time() - started, 1), flush=True)
    (dest / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                checkpoint=p.checkpoint,
                histories=p.meta,
                horizons_ms=[h * 10 for h in HORIZONS],
                native_forward_error=p.native_error,
                forward_error=forward_error,
                native_gradient_relative_error=gradient_error,
                script_sha256=sha(__file__),
                seconds=time.time() - started,
            ),
            indent=2,
        )
    )


def launch(seed):
    with (ROOT / "logs" / f"future_{seed}.log").open("w") as f:
        r = subprocess.run(
            [sys.executable, __file__, str(seed)], stdout=f, stderr=subprocess.STDOUT
        )
    return seed, r.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(int(sys.argv[1]))
    else:
        protocol = dict(
            cohort="All 20 frozen Adult policies, seeds 2011–2030, 32768 training decisions",
            histories="All 80 existing prespecified histories per seed: five heights × eight phases × two controller origins",
            start=".16 s; actual .16/.20 histories then repeat final history",
            horizons_ms=[h * 10 for h in HORIZONS],
            outcome="Sum of squared derivatives of all six tanh policy means with respect to initial model states; no physical onset gate",
            enrichment="Each history normalized separately; cell-count-normalized energy versus entire population and versus the correct current model partition; all 10 superclasses plus mechanosensory class retained",
            statistics="Seed is n=20; exploratory; entire 11-group × 5-horizon family corrected by Holm. Heights averaged within seed.",
            boundary="Local future-output influence, not activity response, physical crossing intervention, or real neural recordings. MLP units have no homologous cell labels.",
            data_selection="Written before new sensitivity outputs were inspected",
        )
        (ROOT / "FUTURE_SENSITIVITY_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2)
        )
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for r in pool.map(launch, range(2011, 2031)):
                print(r, flush=True)

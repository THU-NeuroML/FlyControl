"""40-ms sensory-class atlas and finite outgoing-transmission interventions."""

import os

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"
from pathlib import Path
import sys, json, time, subprocess, concurrent.futures
import numpy as np
import pandas as pd
import torch
from .paths import DATA, OUTPUT as R
from .probe import Probe, sha

ANN = pd.read_csv(DATA / "anatomy/current_graph_annotations.csv")
CLASSES = [
    "mechanosensory",
    "visual",
    "olfactory",
    "gustatory",
    "hygrosensory",
    "thermosensory",
    "unknown_sensory",
]
GROUPS = {s: np.flatnonzero(ANN["class"].eq(s)) for s in CLASSES}
GROUPS.update(
    {
        s: np.flatnonzero(ANN.super_class.eq(s))
        for s in ["ascending", "sensory_ascending"]
    }
)
TARGETS = ["mechanosensory", "visual", "olfactory"]
DOSES = [0.0, 0.25, 0.5, 1.0]


def run(seed):
    dest = R / "interventions" / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    tic = time.time()
    p = Probe("connectome", seed)
    for mod in p.policy.modules():
        for name, v in mod._buffers.items():
            if v is not None:
                mod._buffers[name] = v.detach().clone()
        for name, v in mod._parameters.items():
            if v is not None:
                mod._parameters[name] = torch.nn.Parameter(
                    v.detach().clone(), requires_grad=False
                )
    c = p.policy.actor.core
    g = c.fixed_graph
    n, N = p.n, c.N
    S, I = c.S, c.I
    WT = torch.sparse_csr_tensor(
        torch.cat([torch.zeros(1, dtype=torch.long), g.b_lengths.cumsum(0)]),
        g.b_source,
        g.b_weight,
        size=(N, N),
    )
    x = p.windows[:, 0].reshape(n, 4, 172).clone()
    h0 = p.hidden.clone()
    h = h0.clone()
    rs = []
    bs = []
    with torch.no_grad():
        for t in range(4):
            rr = h.tanh()
            v = c.recurrent_drive(rr)
            v[:, :S] += c.input_proj(x[:, t])
            target = v.tanh()
            rs.append(1 - rr * rr)
            bs.append(0.5 * (1 - target * target))
            h = 0.5 * h + 0.5 * target
        mean = 3 * torch.tanh(c.output_layer(h[:, S + I :].tanh()) / 3)
        ref = mean.tanh()
        forward_error = float((mean - p.ref[:, 0]).abs().max())
        assert forward_error < 2e-5
    h = h.detach().requires_grad_(True)
    act = (3 * torch.tanh(c.output_layer(h[:, S + I :].tanh()) / 3)).tanh()
    energy = torch.zeros_like(h)
    for ch in range(6):
        lam = torch.autograd.grad(act[:, ch].sum(), h, retain_graph=ch < 5)[0].detach()
        with torch.no_grad():
            for t in range(3, -1, -1):
                lam = (
                    0.5 * lam
                    + torch.sparse.mm(WT, (lam * bs[t]).T.contiguous()).T * rs[t]
                )
            energy += lam.square()
    den = energy.sum(1)
    rows = []
    for group, ids in GROUPS.items():
        en = energy[:, ids].sum(1)
        enrich = en / den * N / len(ids)
        for height in np.unique(p.heights):
            mask = p.heights == height
            rows.append(
                dict(
                    seed=seed,
                    group=group,
                    n_cells=len(ids),
                    height=height,
                    horizon_ms=40,
                    enrichment=float(enrich[mask].mean()),
                    fraction=float((en[mask] / den[mask]).mean()),
                )
            )
    atlas = pd.DataFrame(rows)
    previous = pd.read_csv(
        DATA / "mechanisms/coarse_neurons" / str(seed) / "sensitivity.csv"
    )
    a = (
        atlas[atlas.group == "mechanosensory"]
        .sort_values("height")
        .enrichment.to_numpy()
    )
    b = (
        previous[(previous.group == "mechanosensory") & (previous.horizon_ms == 40)]
        .sort_values("height")
        .population_enrichment.to_numpy()
    )
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-7)
    atlas.to_csv(dest / "atlas_40ms.csv", index=False)
    degree = np.bincount(g.f_source.numpy(), minlength=N) + g.f_lengths.numpy()
    # Degree bins and model partition are fixed before any intervention outcome.
    bins = pd.qcut(
        pd.Series(degree).rank(method="average"), 10, labels=False, duplicates="drop"
    ).to_numpy()
    part = np.where(np.arange(N) < S, 0, np.where(np.arange(N) < S + I, 1, 2))
    rng = np.random.default_rng(20261022 + seed)
    masks = {}
    matching = []
    for group in TARGETS:
        ids = GROUPS[group]
        assert np.all(part[ids] == 1)
        masks[(group, "target", 0)] = ids
        if group == "visual":
            continue  # Exact non-overlapping matching is infeasible (documented before outcomes).
        for rep in range(5):
            chosen = []
            for db in np.unique(bins[ids]):
                count = int((bins[ids] == db).sum())
                pool = np.setdiff1d(np.flatnonzero((part == 1) & (bins == db)), ids)
                assert len(pool) >= count, (group, db, len(pool), count)
                chosen.extend(rng.choice(pool, count, replace=False))
            selected = np.array(chosen)
            assert (
                len(np.unique(selected)) == len(ids)
                and not np.intersect1d(selected, ids).size
            )
            assert np.array_equal(np.bincount(bins[selected]), np.bincount(bins[ids]))
            masks[(group, "matched random", rep)] = selected
            matching.append(
                dict(
                    seed=seed,
                    group=group,
                    replicate=rep,
                    n_cells=len(ids),
                    overlap_with_target=0,
                    total_degree_mean=float(degree[selected].mean()),
                    target_total_degree_mean=float(degree[ids].mean()),
                )
            )
    np.savez_compressed(
        dest / "matched_masks.npz",
        **{
            f"{group}_{kind.replace(' ', '_')}_{rep}": ids
            for (group, kind, rep), ids in masks.items()
        },
    )
    pd.DataFrame(matching).to_csv(dest / "matching.csv", index=False)
    records = []
    with torch.no_grad():
        baseline_power = ref.square().mean(1)
        for (group, kind, rep), ids in masks.items():
            for dose in DOSES:
                state = h0.clone()
                for t in range(4):
                    rates = state.tanh()
                    rates[:, ids] *= 1 - dose
                    v = c.recurrent_drive(rates)
                    v[:, :S] += c.input_proj(x[:, t])
                    state = 0.5 * state + 0.5 * v.tanh()
                mu = 3 * torch.tanh(c.output_layer(state[:, S + I :].tanh()) / 3)
                residual = mu.tanh()
                delta_power = (residual - ref).square().mean(1)
                kl = (0.5 * (mu - mean).square() / p.var[:, 0, :]).sum(1)
                if dose == 0:
                    assert float(delta_power.max()) == 0.0
                for height in np.unique(p.heights):
                    mask = p.heights == height
                    records.append(
                        dict(
                            seed=seed,
                            group=group,
                            kind=kind,
                            replicate=rep,
                            dose=dose,
                            height=height,
                            delta_power_mean=float(delta_power[mask].mean()),
                            baseline_power_mean=float(baseline_power[mask].mean()),
                            policy_kl=float(kl[mask].mean()),
                        )
                    )
            print(seed, group, kind, rep, round(time.time() - tic, 1), flush=True)
        pd.DataFrame(records).to_csv(dest / "interventions.csv", index=False)
    (dest / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                checkpoint=p.checkpoint,
                histories=p.meta,
                native_forward_error=p.native_error,
                forward_error=forward_error,
                old_atlas_match=True,
                n_histories=p.n,
                target_groups=TARGETS,
                doses=DOSES,
                random_controls_per_group={
                    "mechanosensory": 5,
                    "olfactory": 5,
                    "visual": 0,
                },
                seconds=time.time() - tic,
                script_sha256=sha(__file__),
            ),
            indent=2,
        )
    )


def launch(seed):
    with (R / "logs" / f"circuit_{seed}.log").open("w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "fig2.current.circuit_probe", str(seed)],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return seed, r.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(int(sys.argv[1]))
    else:
        protocol = dict(
            cohort="All20 current frozen Adult models; original80 shared histories per seed",
            reason="Connect early cell-class enrichment to temporal influence and a finite intervention on the same class; replace the old contact-evoked excess-loss panel in the main mechanism chain",
            early_atlas="All seven sensory annotation classes plus AN and SA; 40ms after probe onset; whole-network per-cell sensitivity baseline; same derivative as v9; 40ms requested after previous temporal screen, descriptive table, not independent confirmation",
            intervention="Multiply outgoing tanh(state) of selected cells by 1-dose before recurrent propagation at each of the four10ms updates. Do not alter external inputs, readout, or frozen weights. Measure output at40ms.",
            targets=TARGETS,
            main_target="mechanosensory, named by the user before this intervention",
            doses=DOSES,
            random_controls="Five random masks for mechanosensory and olfactory per seed; visual targeted blockade retained descriptively because exact non-overlapping matching is infeasible; exact same cell count, internal partition, total-degree decile counts; zero overlap with target class; same masks reused across all doses",
            primary_outcome="100*sqrt(mean squared residual-output change / mean squared intact residual output), pooling all histories and six channels within seed. Random statistic averages five mask values within seed.",
            statistics="n20 paired training seeds; exact two-sided sign flips target-minus-random; Holm6 across two feasible matched groups × three nonzero doses; paired seed bootstrap20000. PolicyKL descriptive only.",
            boundary="Finite model-circuit intervention on fixed input histories; does not measure closed-loop crossing success or establish in-vivo biological necessity. Non-mechanosensory target results and old contact-pulse holds retained in report.",
            frozen_before_new_results=True,
            amendment_before_outcomes="Original non-overlap matching failed for visual degree bin0:8655 target cells versus6174 eligible controls. No intervention outcomes were computed. Retain visual targeted dose responses descriptively; do not weaken the matching rule; inferential family now six comparisons.",
        )
        (R / "CIRCUIT_PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for result in pool.map(launch, range(2011, 2031)):
                print(result, flush=True)

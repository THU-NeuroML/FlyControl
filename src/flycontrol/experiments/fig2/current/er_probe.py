"""Matched ER40-ms atlas on exactly the current Adult/MLP common histories."""

from pathlib import Path
import os, sys, json, time, types

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
from .paths import DATA, OUTPUT as R
from . import probe as f
import torch, numpy as np, pandas as pd
from terrain_control.policy import build_policy, FixedRateCore
from fig6.paths import ROOT as REPO

ANN = pd.read_csv(DATA / "anatomy/current_graph_annotations.csv")
GROUPS = {
    x: np.flatnonzero(ANN["class"].eq(x))
    for x in [
        "mechanosensory",
        "visual",
        "olfactory",
        "gustatory",
        "hygrosensory",
        "thermosensory",
        "unknown_sensory",
    ]
}
GROUPS.update(
    {
        x: np.flatnonzero(ANN.super_class.eq(x))
        for x in ["ascending", "sensory_ascending"]
    }
)
PILOT_CHECKPOINT = None


def load(method, seed):
    from flycontrol.pipeline import checkpoint
    from flycontrol.paths import read_json
    path = checkpoint(REPO, read_json(REPO / "run.json"), "er", seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert (ck["timesteps"] == read_json(REPO / "protocol.json")["ppo"]["total_timesteps"]) and ck.get(
        "obs_rms_state"
    ) is None
    p = build_policy(
        DATA / "er_inputs" / f"er_seed{seed}.npz",
        688,
        6,
        "connectome",
        seed,
        "cpu",
        "normalized",
        -1.0,
    )
    p.load_state_dict(ck["policy_state"])
    p.eval()
    return p, dict(method="er", seed=seed, path=str(path), sha256=f.sha(path))


def run(seed):
    out = R / "er_atlas" / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        return
    tic = time.time()
    f.load_policy = load
    p = f.Probe("connectome", seed)
    for mod in p.policy.modules():
        for key, v in mod._buffers.items():
            if v is not None:
                mod._buffers[key] = v.detach().clone()
        for key, v in mod._parameters.items():
            if v is not None:
                mod._parameters[key] = torch.nn.Parameter(
                    v.detach().clone(), requires_grad=False
                )
    c = p.policy.actor.core
    g = c.fixed_graph
    n, N = p.n, c.N
    S, I = c.S, c.I
    assert (S, I, N) == (2279, 135015, 138584)
    wt = torch.sparse_csr_tensor(
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
        error = float((mean - p.ref[:, 0]).abs().max())
        assert error < 2e-5
    # Independent native-autograd check of one output and history.
    saved = c.recurrent_drive
    c.recurrent_drive = FixedRateCore.recurrent_drive.__get__(c, type(c))
    hn = h0[:1].clone().requires_grad_(True)
    mu, _, _ = p.policy(p.windows[:1, 0].clone(), hn)
    native = torch.autograd.grad(mu[:, 0].tanh().sum(), hn)[0].detach()
    c.recurrent_drive = saved
    h = h.detach().requires_grad_(True)
    act = (3 * torch.tanh(c.output_layer(h[:, S + I :].tanh()) / 3)).tanh()
    energy = torch.zeros_like(h)
    for ch in range(6):
        lam = torch.autograd.grad(act[:, ch].sum(), h, retain_graph=ch < 5)[0].detach()
        with torch.no_grad():
            for t in range(3, -1, -1):
                lam = (
                    0.5 * lam
                    + torch.sparse.mm(wt, (lam * bs[t]).T.contiguous()).T * rs[t]
                )
            if ch == 0:
                gradient_error = float(
                    (lam[:1] - native).norm() / native.norm().clamp_min(1e-20)
                )
                assert gradient_error < 2e-4
            energy += lam.square()
    den = energy.sum(1)
    assert (den > 0).all()
    rows = []
    for group, ids in GROUPS.items():
        frac = energy[:, ids].sum(1) / den
        en = frac * N / len(ids)
        for height in f.HEIGHTS:
            mask = p.heights == height
            assert mask.sum() == 2 * len(f.PHASES)
            rows.append(
                dict(
                    seed=seed,
                    group=group,
                    n_cells=len(ids),
                    height=height,
                    horizon_ms=40,
                    enrichment=float(en[mask].mean()),
                    fraction=float(frac[mask].mean()),
                )
            )
    pd.DataFrame(rows).to_csv(out / "atlas_40ms.csv", index=False)
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                checkpoint=p.checkpoint,
                n_histories=n,
                histories=p.meta,
                native_forward_error=p.native_error,
                forward_error=error,
                native_gradient_relative_error=gradient_error,
                seconds=time.time() - tic,
            ),
            indent=2,
        )
    )
    print("ATLAS_COMPLETE", seed, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("--pilot-checkpoint")
    a = p.parse_args()
    PILOT_CHECKPOINT = a.pilot_checkpoint
    run(a.seed)

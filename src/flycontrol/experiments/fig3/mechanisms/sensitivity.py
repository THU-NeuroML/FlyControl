"""The original 36-window, 10--300 ms local output-sensitivity assay."""

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from fig6.paths import sha256 as sha, write_json
from fig3.paths import MECHANISMS, BANK, checkpoint as cp_path

S, I, D = 18756, 118538, 1290
N = S + I + D
STARTS = [8, 20, 32]
NAMES = [
    f"input_{m}_{a:+d}"
    for m in ["connectome", "mlp"]
    for a in [-90, -60, -30, 30, 60, 90]
]
from .kernel import Kernel


def mask_on_device(device):
    a = sparse.load_npz(MECHANISMS / "sensitivity/source_data/group_membership.npz")
    return torch.sparse_csr_tensor(
        torch.as_tensor(a.indptr, dtype=torch.long),
        torch.as_tensor(a.indices, dtype=torch.long),
        torch.as_tensor(a.data),
        size=a.shape,
        device=device,
    )


def measure(values, M, inv):
    # values is window/trial × neuron squared effect; mean of ratios as in stage1.
    e = torch.sparse.mm(M, values.T).T
    den = torch.stack(
        [values[:, :S].sum(1), values[:, S : S + I].sum(1), values[:, -D:].sum(1)], 1
    )
    part = torch.as_tensor(
        inv.partition.map({"input": 0, "internal": 1, "output": 2}).to_numpy(),
        device=values.device,
    )
    denom = den[:, part]
    frac = torch.where(denom > 0, e / denom, torch.zeros_like(e))
    n = torch.as_tensor(inv.n_cells.to_numpy(), device=values.device)
    width = torch.as_tensor(inv.denominator_cells.to_numpy(), device=values.device)
    return (
        torch.stack([(e / n).mean(0), frac.mean(0), frac.mean(0) * width / n], 1)
        .cpu()
        .numpy()
        .astype(float)
    )


def local(k, M, inv, dest):
    obs = torch.as_tensor(
        np.stack([np.load(BANK / f"{n}.npz")["observations"] for n in NAMES]),
        device=k.device,
    )
    with torch.no_grad():
        states, _ = k.replay(obs)
    windows = torch.cat([obs[:, s : s + 8] for s in STARTS])
    rows = []
    raw = []
    for horizon in range(1, 31):
        with torch.no_grad():
            h = states
            rs = []
            bs = []
            for t in range(horizon):
                h, r, b = k.substep(h, windows[:, t // 4], True)
                rs.append(r)
                bs.append(b)
        hd = h.detach().requires_grad_(True)
        act = k.output(hd)[1]
        lam = torch.autograd.grad(act[:, 1].sum(), hd)[0].detach()
        with torch.no_grad():
            for t in reversed(range(horizon)):
                lam = (
                    0.5 * lam
                    + torch.sparse.mm(k.WT, (lam * bs[t]).T.contiguous()).T * rs[t]
                )
            result = measure(lam.square(), M, inv)
        raw.append(result)
        for j, gid in enumerate(inv.group_id):
            rows.append(
                dict(
                    method=k.method,
                    seed=k.seed,
                    group_id=gid,
                    time_ms=10 * horizon,
                    mean_squared_derivative=result[j, 0],
                    rms_derivative=float(np.sqrt(result[j, 0])),
                    energy_fraction=result[j, 1],
                    enrichment=result[j, 2],
                )
            )
    x = pd.DataFrame(rows)
    x.to_csv(dest / "local.csv.gz", index=False)
    np.savez_compressed(
        dest / "local_arrays.npz",
        values=np.stack(raw),
        time_ms=np.arange(1, 31) * 10,
        group_ids=inv.group_id.to_numpy(str),
        metrics=np.array(["mean_squared_derivative", "energy_fraction", "enrichment"]),
    )
    # Independent native forward/state and finite-difference checks are kept.
    with torch.no_grad():
        states, means = k.replay(obs)
    checks = k.verify(obs, states, means)
    return dict(windows=36, timepoints=30, **checks)


def run(seed, device, output):
    output.mkdir(parents=True, exist_ok=True)
    k = Kernel("connectome", seed, device)
    inventory = pd.read_csv(MECHANISMS / "sensitivity/source_data/group_inventory.csv")
    qa = local(k, mask_on_device(device), inventory, output)
    qa.update(seed=seed, checkpoint_sha256=sha(cp_path("connectome", seed)))
    write_json(output / "COMPLETE.json", qa)

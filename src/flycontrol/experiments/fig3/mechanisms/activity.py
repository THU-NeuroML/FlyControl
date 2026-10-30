"""Original nine-history replay; cell-wise differences precede class aggregation."""

from pathlib import Path
import json, time, gc
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from flycontrol.experiments.shared.paths import sha256 as sha
from flycontrol.experiments.fig3.paths import MECHANISMS, checkpoint as cp_path, generated_path

S, I, D = 18756, 118538, 1290
N = S + I + D
STARTS = [8, 20, 32]
NAMES = [
    f"input_{m}_{a:+d}"
    for m in ["connectome", "mlp"]
    for a in [-90, -60, -30, 30, 60, 90]
]
from flycontrol.experiments.fig3.mechanisms.kernel import Kernel

PHASES = [790000, 790101, 790102]
ANGLES = [-60, 0, 60]
METRICS = [
    "raw_mean",
    "raw_mean_square",
    "delta_mean",
    "delta_mean_square",
    "positive_delta",
    "negative_delta",
    "baseline_mean_square",
]


@torch.no_grad()
def run(seed, device, output):
    dest = output
    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cohort = pd.read_csv(MECHANISMS / "activity/source_data/turn_cohort.csv")
    records = []
    for phase in PHASES:
        for angle in ANGLES:
            path = (
                MECHANISMS / f"activity/controls/seed{seed}_phase{phase}/trajectory.npz"
                if angle == 0
                else generated_path(
                    cohort[
                        (cohort.seed == seed)
                        & (cohort.phase == phase)
                        & (cohort.angle == angle)
                    ].path.iloc[0]
                )
            )
            if angle == 0:
                assert (path.parent / "COMPLETE.json").exists()
            with np.load(path) as z:
                record = {k: z[k] for k in z.files}
            record.update(
                seed=seed, phase=phase, angle=angle, path=str(path), sha256=sha(path)
            )
            records.append(record)
    lengths = np.array([len(v["observations"]) for v in records])
    T = 50
    x = np.zeros((9, T, 64), np.float32)
    for j, v in enumerate(records):
        x[j, : lengths[j]] = v["observations"]
    for j in range(0, 9, 3):
        assert np.array_equal(x[j, :8], x[j + 1, :8]) and np.array_equal(
            x[j + 1, :8], x[j + 2, :8]
        )
    x = torch.as_tensor(x, device=device)
    k = Kernel("connectome", seed, device)
    h = torch.zeros((9, N), device=device)
    rates = []
    means = []
    for ti in range(T):
        for _ in range(4):
            h = k.substep(h, x[:, ti])
        rates.append(h.tanh())
        means.append(k.output(h)[0])
    rates = torch.stack(rates, 1)
    means = torch.stack(means, 1).cpu().numpy()
    del h
    validations = []
    for j, v in enumerate(records):
        length = lengths[j]
        ae = float(np.max(np.abs(means[j, :length] - v["actions"])))
        if "selected_state_indices" in v:
            re = float(
                np.max(
                    np.abs(
                        rates[j, :length, v["selected_state_indices"]].cpu().numpy()
                        - v["rates"]
                    )
                )
            )
        else:
            with np.load(Path(v["path"]).parent / "reference_rates.npz") as native:
                nr = native["end_rates"]
            re = float(np.max(np.abs(rates[j, :length, :S].cpu().numpy() - nr)))
        assert ae < 2e-5 and re < 2e-5, (seed, j, ae, re)
        validations.append(
            {key: v[key] for key in ["seed", "phase", "angle", "path", "sha256"]}
            | dict(frames=int(length), action_max_error=ae, rate_max_error=re)
        )
    m = sparse.load_npz(
        MECHANISMS / "activity/source_data/analysis_membership.npz"
    ).astype(np.float32)
    inv = pd.read_csv(MECHANISMS / "activity/source_data/analysis_inventory.csv")
    m = sparse.diags(1 / inv.n_cells.to_numpy(np.float32)) @ m
    m = m.tocsr()
    m.sort_indices()
    M = torch.sparse_csr_tensor(
        torch.as_tensor(m.indptr, dtype=torch.int64),
        torch.as_tensor(m.indices, dtype=torch.int64),
        torch.as_tensor(m.data),
        size=m.shape,
        device=device,
    )

    def group(z):
        shape = z.shape[:-1]
        return (
            torch.sparse.mm(M, z.reshape(-1, N).T.contiguous())
            .T.reshape(*shape, m.shape[0])
            .cpu()
            .numpy()
        )

    # All arrays below have axes phase, angle, time, group, metric.
    r = rates.reshape(3, 3, T, N)
    base = r[:, :, 3:8].mean(2, keepdim=True)
    delta = r - r[:, 1:2]
    prefix = float(delta[:, :, :8].abs().max())
    assert prefix < 2e-6, prefix
    values = np.stack(
        [
            group(r),
            group(r.square()),
            group(delta),
            group(delta.square()),
            group(delta.clamp_min(0)),
            group(delta.clamp_max(0)),
            group((r - base).square()),
        ],
        -1,
    )
    lr = r[:, 2] - r[:, 0]
    direction = np.stack([group(lr), group(lr.square())], -1)
    # Independent float64 NumPy reductions: avoid relying on the same sparse group kernel.
    reduction_error = 0.0
    for gid in ["g0003", "g0006", "g0877", "g1350"]:
        gi = int(np.flatnonzero(inv.group_id.eq(gid))[0])
        ids = m.indices[m.indptr[gi] : m.indptr[gi + 1]]
        rr = (
            r[0, 0]
            .index_select(1, torch.as_tensor(ids, dtype=torch.long, device=device))
            .cpu()
            .numpy()
            .astype(float)
        )
        cc = (
            r[0, 1]
            .index_select(1, torch.as_tensor(ids, dtype=torch.long, device=device))
            .cpu()
            .numpy()
            .astype(float)
        )
        dd = rr - cc
        check = np.stack(
            [
                rr.mean(1),
                (rr**2).mean(1),
                dd.mean(1),
                (dd**2).mean(1),
                np.maximum(dd, 0).mean(1),
                np.minimum(dd, 0).mean(1),
            ],
            -1,
        )
        reduction_error = max(
            reduction_error, float(np.max(np.abs(check - values[0, 0, :, gi, :6])))
        )
    assert reduction_error < 2e-6, reduction_error
    for j, length in enumerate(lengths):
        p, a = divmod(j, 3)
        values[p, a, length:] = np.nan
        paired = min(length, lengths[p * 3 + 1])
        values[p, a, paired:, :, 2:6] = np.nan
    for p in range(3):
        direction[p, min(lengths[p * 3], lengths[p * 3 + 2]) :] = np.nan
    # GPU sparse averaging error is small; explicitly validate signed cancellation identity.
    err = float(np.nanmax(np.abs(values[..., 2] - values[..., 4] - values[..., 5])))
    assert err < 1e-5
    np.savez_compressed(
        dest / "activity.npz",
        values=values,
        direction=direction,
        metrics=np.array(METRICS),
        direction_metrics=np.array(["mean", "mean_square"]),
        phases=PHASES,
        angles=ANGLES,
        time_s=np.arange(T) * 0.04 - 0.3,
        group_id=inv.group_id.to_numpy(str),
        lengths=lengths.reshape(3, 3),
    )
    pd.DataFrame(validations).to_csv(dest / "replay_validation.csv", index=False)
    result = dict(
        seed=seed,
        seconds=time.time() - start,
        checkpoint_sha256=sha(cp_path("connectome", seed)),
        script_sha256=sha(__file__),
        traces=9,
        action_max_error=max(v["action_max_error"] for v in validations),
        rate_max_error=max(v["rate_max_error"] for v in validations),
        precommand_delta_max=prefix,
        signed_decomposition_error=err,
        numpy_reduction_max_error=reduction_error,
        frames=lengths.tolist(),
    )
    (dest / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "REPLAY",
        seed,
        round(result["seconds"], 1),
        result["action_max_error"],
        result["rate_max_error"],
        flush=True,
    )
    del k, r, rates, delta, values, lr, direction
    gc.collect()
    torch.cuda.empty_cache()

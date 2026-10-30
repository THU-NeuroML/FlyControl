"""Four-way component exchanges in the original 24 frozen turning policies."""

import os

for name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

from flycontrol.experiments.shared.paths import ROOT
from flycontrol.experiments.fig3.paths import DATA, MECHANISMS, BANK, checkpoint as cp_path
from flycontrol.experiments.fig3.runtime import activate

activate()
from flycontrol.experiments.fig3.mechanisms.kernel import Kernel, read_state, STARTS, NAMES, S, N, D

O = MECHANISMS / "interface"
B = MECHANISMS / "anatomy"
F = DATA

E_KEYS = ["input_proj.weight", "input_proj.bias"]
C_KEYS = [
    "output_layer.0.weight",
    "output_layer.0.bias",
    "output_layer.1.weight",
    "output_layer.1.bias",
]
ATTR = dict(zip(E_KEYS + C_KEYS, ["E", "eb", "gamma", "beta", "C", "cb"]))
SOURCES = {}


def record(path):
    path = Path(path).resolve()
    if str(path) not in SOURCES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        SOURCES[str(path)] = dict(
            path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size
        )
    return SOURCES[str(path)]["sha256"]


def save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def select(k, initial, final, e, c):
    k.st = dict(final)
    for key in E_KEYS:
        k.st["actor.core." + key] = (initial if e == 0 else final)["actor.core." + key]
    for key in C_KEYS:
        k.st["actor.core." + key] = (initial if c == 0 else final)["actor.core." + key]
    for key, attr in ATTR.items():
        setattr(k, attr, k.st["actor.core." + key].to(k.device))


def finite_difference(k, states, windows, ids):
    """Independent float64 central difference at the primary 100-ms horizon."""
    from scipy import sparse

    W = k.cpuW.astype(np.float64)
    E, eb, gamma, beta, C, cb = [
        getattr(k, x).detach().cpu().numpy().astype(np.float64)
        for x in ["E", "eb", "gamma", "beta", "C", "cb"]
    ]
    h0 = states[:1].cpu().numpy().astype(np.float64).T
    obs = windows[:1].cpu().numpy().astype(np.float64)
    direction = np.zeros((N, 1))
    direction[ids] = 1 / np.sqrt(len(ids))
    eps = 1e-3
    vals = []
    for sign in [-1, 1]:
        h = h0 + sign * eps * direction
        for t in range(10):
            v = W @ np.tanh(h)
            v[:S] += E @ obs[0, t // 4, :, None] + eb[:, None]
            h = 0.5 * h + 0.5 * np.tanh(v)
        r = np.tanh(h[-D:, 0])
        normalized = (r - r.mean()) / np.sqrt(r.var() + 1e-5)
        z = C @ (normalized * gamma + beta) + cb
        vals.append(float(np.tanh(3 * np.tanh(z[1] / 3))))
    finite = (vals[1] - vals[0]) / (2 * eps)
    grad, _, _ = k.gradient(states[:1], windows[:1], 10)
    analytic = float(grad[0, 1, ids].sum().cpu()) / np.sqrt(len(ids))
    relative = abs(finite - analytic) / max(abs(finite), 1e-7)
    assert relative < 0.01, (finite, analytic, relative)
    return dict(
        finite_difference_100ms=finite,
        analytic_100ms=analytic,
        relative_error_100ms=relative,
    )


def run(seed, device):
    (O / "source_data").mkdir(parents=True, exist_ok=True)
    dest = O / "models" / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "COMPLETE.json").exists():
        return
    started = time.time()
    final_path, initial_path = (
        cp_path("connectome", seed),
        cp_path("connectome", seed, True),
    )
    fh, ih = record(final_path), record(initial_path)
    atlas = json.loads(
        (MECHANISMS / f"sensitivity/models/connectome_{seed}/COMPLETE.json").read_text()
    )
    assert atlas["checkpoint_sha256"] == fh
    k = Kernel("connectome", seed, device)
    final, initial = dict(k.st), read_state("connectome", seed, True)
    assert torch.count_nonzero(initial["actor.core.output_layer.1.weight"]) == 0
    assert torch.count_nonzero(initial["actor.core.output_layer.1.bias"]) == 0
    changed = []
    for key in final:
        if key.startswith("actor.") and not torch.equal(final[key], initial[key]):
            changed.append(key)
            assert key.removeprefix("actor.core.") in E_KEYS + C_KEYS, key
    assert all(
        torch.equal(initial["actor.core." + key], final["actor.core." + key])
        for key in ["edge_pre", "edge_post", "edge_weight", "rate_tau"]
    )
    bank = [BANK / f"{name}.npz" for name in NAMES]
    obs = torch.as_tensor(
        np.stack([np.load(path)["observations"] for path in bank]), device=device
    )
    for path in bank:
        record(path)
    assert tuple(obs.shape) == (12, 50, 64) and torch.isfinite(obs).all()
    windows = torch.cat([obs[:, start : start + 8] for start in STARTS])
    mapping_path = B / "source_data/nodes.csv.gz"
    record(mapping_path)
    mapping = pd.read_csv(mapping_path).iloc[:S]
    assert np.array_equal(mapping.state_index, np.arange(S))
    groups = {
        str(g): x.state_index.to_numpy(int) for g, x in mapping.groupby("refined_group")
    }
    assert len(groups) == 9 and sum(map(len, groups.values())) == S
    rows, window_rows, qa = [], [], []
    for e in [0, 1]:
        select(k, initial, final, e, 0)
        states, _ = k.replay(obs)
        for c in [0, 1]:
            select(k, initial, final, e, c)
            variant = f"E{e}C{c}"
            if seed == 4301:
                _, means = k.replay(obs)
                checks = k.verify(obs, states, means)
                checks.update(finite_difference(k, states, windows, groups["true_AN"]))
                qa.append(dict(variant=variant, **checks))
            for horizon in [1, 4, 10, 30]:
                grad, _, act = k.gradient(states, windows, horizon)
                energy = grad[:, :, :S].double().square().cpu().numpy()
                totals = energy.sum(2)
                assert np.isfinite(energy).all()
                assert (totals == 0).all() if c == 0 else (totals > 0).all()
                if c == 0:
                    assert torch.count_nonzero(act) == 0
                for channel, ch in enumerate(["drive", "turn"]):
                    for group, ids in groups.items():
                        absolute = energy[:, channel, ids].mean(1)
                        # An exactly zero readout has undefined 0/0 enrichment.
                        fraction = (
                            energy[:, channel, ids].sum(1) / totals[:, channel]
                            if c
                            else np.full(36, np.nan)
                        )
                        enrichment = fraction * S / len(ids)
                        base = dict(
                            seed=seed,
                            E=e,
                            C=c,
                            variant=variant,
                            horizon_ms=horizon * 10,
                            channel=ch,
                            group=group,
                            n_cells=len(ids),
                        )
                        rows.append(
                            dict(
                                **base,
                                mean_squared_derivative=float(absolute.mean()),
                                energy_fraction=float(fraction.mean()),
                                enrichment=float(enrichment.mean()),
                            )
                        )
                        for j in range(36):
                            window_rows.append(
                                dict(
                                    **base,
                                    onset_step=STARTS[j // 12],
                                    history=NAMES[j % 12],
                                    mean_squared_derivative=float(absolute[j]),
                                    enrichment=float(enrichment[j]),
                                    energy_fraction=float(fraction[j]),
                                )
                            )
                del grad, energy
            print(f"seed {seed} {variant}: {time.time() - started:.1f} s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(dest / "seed_metrics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(dest / "window_metrics.csv.gz", index=False)
    save(dest / "QA.json", qa)
    pd.DataFrame(SOURCES.values()).to_csv(
        O / "source_data/source_hashes.csv", index=False
    )
    save(
        dest / "COMPLETE.json",
        dict(
            seed=seed,
            checkpoint_sha256=fh,
            initial_sha256=ih,
            matches_current_main_figure=True,
            frozen_graph_verified=True,
            changed_actor_keys=changed,
            initial_readout_exactly_zero=True,
            initial_readout_relative_enrichment="undefined (0/0)",
            variants=4,
            histories=12,
            windows=36,
            rows=len(table),
            seconds=time.time() - started,
            compute_script_sha256=record(__file__),
        ),
    )
    del k, initial, final, states, windows, obs
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record(O / "PROTOCOL.json")
    record(Path(__file__).with_name("kernel.py"))
    for seed in [args.seed] if args.seed else range(4301, 4325):
        run(seed, args.device)
    print("DONE", flush=True)

"""Frozen PT28 common-input, feature-group gain landscapes; no training."""

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
import sys, json, time, argparse, types, hashlib
import numpy as np
import pandas as pd
import torch
from flycontrol.experiments.shared.paths import ROOT
from flycontrol.experiments.fig3.paths import DATA, BANK, checkpoint
from flycontrol.experiments.fig3.runtime import activate

activate()
from turning_control.policy import build_policy

O = DATA / "mechanisms/input_gain"
O.mkdir(parents=True, exist_ok=True)
INPUT = DATA / "mechanisms/input_gain"
P = json.loads((INPUT / "PROTOCOL.json").read_text())
BLOCKS = P["candidate_axes"]
assert sorted(sum(BLOCKS.values(), [])) == list(range(64))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


class Probe:
    @torch.inference_mode()
    def __init__(self, method, seed, device):
        self.method, self.seed, self.device = method, seed, device
        cp = checkpoint(method, seed)
        meta = json.loads((BANK / f"{method}_{seed}/COMPLETE.json").read_text())
        assert sha(cp) == meta["checkpoint_sha256"]
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        assert ck["obs_rms_state"] is None
        self.policy = build_policy(
            DATA / "graph.npz", 64, 2, method, seed, device, "normalized", -1.0
        )
        self.policy.load_state_dict(ck["policy_state"])
        self.policy.eval()
        for par in self.policy.parameters():
            par.requires_grad_(False)
        path = BANK / f"{method}_{seed}/fixed_windows.npz"
        z = np.load(path)
        self.x = torch.as_tensor(z["observations"], device=device)
        self.hidden = torch.as_tensor(z["initial_hidden"], device=device)
        assert self.x.shape == (48, 4, 64)
        self.audit = dict(
            checkpoint=str(cp),
            checkpoint_sha256=sha(cp),
            windows=str(path),
            windows_sha256=sha(path),
            training_steps=int(ck['timesteps']),
        )
        ref0, h0 = self.forward(self.x[:2], self.hidden[:2], True)
        self.native_error = 0.0
        if method == "connectome":
            c = self.policy.actor.core
            g = c.fixed_graph
            ptr = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=device), g.f_lengths.cumsum(0)]
            )
            mat = torch.sparse_csr_tensor(
                ptr, g.f_source, g.f_weight, size=(c.N, c.N), device=device
            )

            def sparse(core, x):
                return torch.sparse.mm(mat, x.T.contiguous()).T.contiguous()

            c.recurrent_drive = types.MethodType(sparse, c)
            ref1, h1 = self.forward(self.x[:2], self.hidden[:2], True)
            self.native_error = max(
                float((ref1 - ref0).abs().max()), float((h1 - h0).abs().max())
            )
            assert self.native_error < 2e-5, self.native_error
        self.ref = self.forward(self.x, self.hidden)
        self.var = self.policy.log_std.exp().square()[None, None, :]
        # Independent equivalence check in the actual first affine layer.
        layer = (
            self.policy.actor.core.input_proj
            if method == "connectome"
            else self.policy.actor.net[0]
        )
        xx = self.x[:2].clone()
        xx[:, :, BLOCKS["CPG"]] *= 1.05
        input_mu = self.forward(xx, self.hidden[:2])
        w = layer.weight.detach().clone()
        layer.weight[:, BLOCKS["CPG"]] *= 1.05
        param_mu = self.forward(self.x[:2], self.hidden[:2])
        layer.weight.copy_(w)
        self.gain_error = float((input_mu - param_mu).abs().max())
        assert self.gain_error < 2e-5
        assert torch.equal(layer.weight, w)
        self.ref = self.forward(self.x.clone(), self.hidden)

    @torch.inference_mode()
    def forward(self, x, h, retstate=False):
        h = h.clone()
        v = []
        for t in range(x.shape[1]):
            mu, h, _ = self.policy(x[:, t], h)
            v.append(mu)
        out = torch.stack(v, 1)
        return (out, h) if retstate else out

    @torch.inference_mode()
    def evaluate(self, changes, windows=False):
        x = self.x.clone()
        for axis, dose in changes.items():
            x[:, :, BLOCKS[axis]] *= 1 + float(dose)
        mu = self.forward(x, self.hidden)
        sq = (mu - self.ref).square()
        vals = (
            torch.stack(
                [
                    (0.5 * sq / self.var).sum(-1).mean(1),
                    (0.5 * sq).sum(-1).mean(1),
                    (mu.tanh() - self.ref.tanh()).square().mean((1, 2)).sqrt(),
                ],
                1,
            )
            .cpu()
            .numpy()
        )
        if windows:
            return vals
        v = vals.mean(0)
        v[2] = np.sqrt(np.mean(vals[:, 2] ** 2))
        return v


def run(method, seed, device, stage):
    out = O / f"models/{method}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{stage.upper()}_COMPLETE.json").exists():
        return
    tic = time.time()
    p = Probe(method, seed, device)
    prep = time.time() - tic
    zero = p.evaluate({})
    assert max(zero[:2]) < 1e-12 and zero[2] < 2e-6, zero
    print(
        "READY",
        method,
        seed,
        round(prep, 2),
        "native_error",
        p.native_error,
        flush=True,
    )
    if stage == "benchmark":
        t = time.time()
        v = p.evaluate({"Velocity": 0.05})
        dt = time.time() - t
        save(
            out / "BENCHMARK.json",
            dict(
                preparation_s=prep,
                evaluation_s=dt,
                value=v.tolist(),
                device=device,
                **p.audit,
            ),
        )
        print("BENCHMARK", dt, v, flush=True)
        return
    if stage == "screen":
        rows = []
        for axis in BLOCKS:
            for dose in P["screen_doses"]:
                v = p.evaluate({axis: dose})
                rows.append(
                    dict(
                        method=method,
                        seed=seed,
                        axis=axis,
                        dose=dose,
                        **dict(zip(P["metrics"], v)),
                    )
                )
            pd.DataFrame(rows).to_csv(out / "screen.csv", index=False)
    else:
        selection = json.loads((INPUT / "AXIS_SELECTION.json").read_text())
        axes = selection["axes"]
        dose = np.linspace(-0.1, 0.1, 21)
        result = np.empty((21, 21, 3))
        bywin = np.empty((21, 21, 48, 3))
        for j, y in enumerate(dose):
            for i, x in enumerate(dose):
                val = p.evaluate({axes[0]: x, axes[1]: y}, windows=True)
                bywin[j, i] = val
                v = val.mean(0)
                v[2] = np.sqrt(np.mean(val[:, 2] ** 2))
                result[j, i] = v
            print("ROW", method, seed, j + 1, round(time.time() - tic, 1), flush=True)
        assert max(result[10, 10, :2]) < 1e-12 and result[10, 10, 2] < 2e-6
        assert np.isfinite(result).all() and np.min(result) >= 0
        np.savez_compressed(
            out / "surface.npz",
            dose=dose,
            metrics=result,
            window_metrics=bywin,
            axes=axes,
        )
    assert sha(p.audit["checkpoint"]) == p.audit["checkpoint_sha256"]
    save(
        out / f"{stage.upper()}_COMPLETE.json",
        dict(
            method=method,
            seed=seed,
            stage=stage,
            device=device,
            native_error=p.native_error,
            gain_equivalence_error=p.gain_error,
            origin_measured=zero.tolist(),
            seconds=time.time() - tic,
            protocol_sha256=sha(INPUT / "PROTOCOL.json"),
            **p.audit,
        ),
    )
    print("COMPLETE", method, seed, stage, round(time.time() - tic, 1), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("method")
    a.add_argument("seed", type=int)
    a.add_argument("stage")
    a.add_argument("--device", default="cpu")
    args = a.parse_args()
    run(args.method, args.seed, args.device, args.stage)

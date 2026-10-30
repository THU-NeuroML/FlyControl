"""Exact original sparse sensitivity kernel, validated against the native actor."""

import gc
import numpy as np
import torch
from scipy import sparse
from fig3.paths import DATA, MECHANISMS, checkpoint as cp_path
from turning_control.policy import build_policy

S, I, D = 18756, 118538, 1290
N = S + I + D
STARTS = [8, 20, 32]
NAMES = [
    f"input_{m}_{a:+d}"
    for m in ["connectome", "mlp"]
    for a in [-90, -60, -30, 30, 60, 90]
]


def read_state(method, seed, initial=False):
    p = torch.load(
        cp_path(method, seed, initial), map_location="cpu", weights_only=False
    )
    assert p["obs_rms_state"] is None
    return p["policy_state"]


class Kernel:
    def __init__(self, method, seed, device="cuda:7", hybrid=False):
        self.device = device
        self.method = method
        self.seed = seed
        st = read_state(method, seed)
        if hybrid:
            ini = read_state(method, seed, True)
            st["actor.core.input_proj.weight"] = ini["actor.core.input_proj.weight"]
            st["actor.core.input_proj.bias"] = ini["actor.core.input_proj.bias"]
        self.st = st
        pre = st["actor.core.edge_pre"].numpy()
        post = st["actor.core.edge_post"].numpy()
        w = st["actor.core.edge_weight"].numpy()
        a = sparse.csr_matrix((w, (post, pre)), shape=(N, N))
        a.sort_indices()
        self.cpuW = a

        def csr(a):
            a = a.tocsr()
            a.sort_indices()
            return torch.sparse_csr_tensor(
                torch.as_tensor(a.indptr.astype(np.int64)),
                torch.as_tensor(a.indices.astype(np.int64)),
                torch.as_tensor(a.data),
                size=a.shape,
                device=device,
            )

        self.W = csr(a)
        self.WT = csr(a.T)
        for key, name in [
            ("input_proj.weight", "E"),
            ("input_proj.bias", "eb"),
            ("output_layer.0.weight", "gamma"),
            ("output_layer.0.bias", "beta"),
            ("output_layer.1.weight", "C"),
            ("output_layer.1.bias", "cb"),
        ]:
            setattr(self, name, st["actor.core." + key].to(device))
        self.gate = float(
            1 - torch.pow(1 - torch.clamp(0.04 / st["actor.core.rate_tau"], 0, 1), 0.25)
        )
        assert abs(self.gate - 0.5) < 1e-6

    def output(self, h):
        r = h[:, -D:].tanh()
        n = torch.nn.functional.layer_norm(r, (D,), self.gamma, self.beta, 1e-5)
        z = n @ self.C.T + self.cb
        mu = 3 * torch.tanh(z / 3)
        return mu, mu.tanh()

    @torch.no_grad()
    def substep(self, h, x, trace=False):
        r = h.tanh()
        v = torch.sparse.mm(self.W, r.T.contiguous()).T
        v[:, :S] += x @ self.E.T + self.eb
        target = v.tanh()
        hn = (1 - self.gate) * h + self.gate * target
        return (hn, 1 - r * r, self.gate * (1 - target * target)) if trace else hn

    @torch.no_grad()
    def replay(self, obs, save_starts=STARTS):
        h = torch.zeros((len(obs), N), device=self.device)
        states = []
        means = []
        for t in range(obs.shape[1]):
            if t in save_starts:
                states.append(h.clone())
            for _ in range(4):
                h = self.substep(h, obs[:, t])
            means.append(self.output(h)[0])
        return torch.cat(states), torch.stack(means, 1)

    def gradient(self, h0, x, horizon):
        h = h0
        rs = []
        bs = []
        for t in range(horizon):
            h, r, b = self.substep(h, x[:, t // 4], True)
            rs.append(r)
            bs.append(b)
        hd = h.detach().requires_grad_(True)
        mu, act = self.output(hd)
        gs = []
        gxs = []
        for ch in [0, 1]:
            lam = torch.autograd.grad(act[:, ch].sum(), hd, retain_graph=ch == 0)[
                0
            ].detach()
            gx = torch.zeros((len(x), (horizon + 3) // 4, 64), device=self.device)
            for t in range(horizon - 1, -1, -1):
                gd = lam * bs[t]
                gx[:, t // 4] += gd[:, :S] @ self.E
                lam = (1 - self.gate) * lam + torch.sparse.mm(
                    self.WT, gd.T.contiguous()
                ).T * rs[t]
            gs.append(lam)
            gxs.append(gx)
        return torch.stack(gs, 1), torch.stack(gxs, 1), act.detach()

    def verify(self, obs, states, means):
        p = build_policy(
            DATA / "graph.npz",
            64,
            2,
            self.method,
            self.seed,
            self.device,
            "normalized",
            -1.0,
        )
        p.load_state_dict(self.st)
        p.eval()
        for par in p.parameters():
            par.requires_grad_(False)
        h = p.init_state(1, self.device)
        out = []
        saved = []
        with torch.no_grad():
            for t in range(50):
                if t in STARTS:
                    saved.append(h.clone())
                m, h, _ = p(obs[:1, t], h)
                out.append(m)
        forward = float((torch.stack(out, 1) - means[:1]).abs().max())
        state = float((torch.cat(saved) - states[[0, 12, 24]]).abs().max())
        h0 = states[:2].clone().requires_grad_(True)
        xx = obs[:2, 8:9].clone().requires_grad_(True)
        m, hn, _ = p(xx[:, 0], h0)
        native = torch.autograd.grad(m[:, 1].tanh().sum(), [h0, xx])
        gs, gx, _ = self.gradient(h0.detach(), xx.detach(), 4)
        ge = float((gs[:, 1] - native[0]).norm() / native[0].norm().clamp_min(1e-20))
        xe = float((gx[:, 1] - native[1]).norm() / native[1].norm().clamp_min(1e-20))
        assert max(forward, state) < 2e-5, (forward, state)
        assert max(ge, xe) < 2e-4, (ge, xe)
        # Float64 finite difference on exact native computation, normalized AN direction.
        p.double()
        base = states[:1].double()
        xx = obs[:1, 8:9].double()
        ids = np.load(MECHANISMS / "anatomy/source_data/masks.npz")["true_AN"]
        direction = torch.zeros_like(base)
        direction[:, ids] = 1 / np.sqrt(len(ids))
        eps = 0.001
        with torch.no_grad():
            plus = p(xx[:, 0], base + eps * direction)[0].tanh()[0, 1]
            minus = p(xx[:, 0], base - eps * direction)[0].tanh()[0, 1]
        finite = float((plus - minus) / (2 * eps))
        analytic = float(gs[0, 1, ids].sum() / np.sqrt(len(ids)))
        err = abs(finite - analytic) / max(abs(finite), 1e-7)
        assert err < 0.01, (finite, analytic, err)
        del p
        gc.collect()
        torch.cuda.empty_cache()
        return dict(
            native_forward_max_error=forward,
            native_state_max_error=state,
            native_state_gradient_relative_error=ge,
            native_input_gradient_relative_error=xe,
            finite_difference=finite,
            analytic=analytic,
            finite_difference_relative_error=err,
        )

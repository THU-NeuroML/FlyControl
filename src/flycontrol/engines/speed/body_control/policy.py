"""51-input/2-action adapter; graph dynamics and Gaussian PPO are shared."""

from .context import REPO
import torch
from fly_control.policy import build_policy as build_shared


def build_policy(method, seed, device, clock_ms=40):
    policy = build_shared(
        REPO / "data/figure4/graph.npz", 51, 2, method, seed, device, "normalized", -1.0
    )
    if method == "connectome":
        core = policy.actor.core
        repeat = int(clock_ms) // 10
        core.dt = clock_ms / 1000.0
        core.internal_substeps = repeat
        with torch.no_grad():
            core.rate_tau.fill_(core.dt / (1 - 0.5**repeat))
    return policy

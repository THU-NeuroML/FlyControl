"""Original graph-only ER control with the exact four-substep actor."""

from .common import DATA
from fly_control.policy import build_policy as build_shared
import torch


def graph_path(method, seed):
    return (
        DATA / "registered_graphs" / f"er_seed{seed}.npz"
        if method == "er"
        else DATA / "graph.npz"
    )


def build_policy(
    ignored_path,
    obs_dim,
    action_dim,
    method,
    seed,
    device,
    readout="normalized",
    log_std=-1.0,
):
    p = build_shared(
        graph_path(method, seed),
        obs_dim,
        action_dim,
        "connectome" if method == "er" else method,
        seed,
        device,
        readout,
        log_std,
    )
    if method in ["connectome", "er"]:
        c = p.actor.core
        c.dt = 0.04
        c.internal_substeps = 4
        with torch.no_grad():
            c.rate_tau.fill_(0.04 / 0.9375)
    return p

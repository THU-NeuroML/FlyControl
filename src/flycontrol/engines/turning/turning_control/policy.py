"""Original PT28 actor: four held-input substeps, normalized 1,290-DN readout."""

from .context import DATA
from fly_control.policy import build_policy as build_shared
import torch


def graph_path(method="connectome", seed=None):
    if method not in ("connectome", "mlp", "cpg"):
        raise ValueError(method)
    return DATA / "graph.npz"


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
    assert (obs_dim, action_dim, readout) == (64, 2, "normalized")
    policy = build_shared(
        graph_path(method, seed),
        obs_dim,
        action_dim,
        method,
        seed,
        device,
        readout,
        log_std,
    )
    if method == "connectome":
        core = policy.actor.core
        core.dt = 0.04
        core.internal_substeps = 4
        with torch.no_grad():
            core.rate_tau.fill_(0.04 / 0.9375)
    return policy

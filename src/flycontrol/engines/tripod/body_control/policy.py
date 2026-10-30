"""Tripod comparison gives both actors the same local feedback branch."""

from pathlib import Path
import torch
from torch import nn
from fly_control.actor_critic import ValueHead
from fly_control.policy import GraphActor, MLPActor, GaussianCoordinatePolicy


class SharedReflexMLPActor(MLPActor):
    """The same fast branch as GraphActor, plus a parameter-matched MLP core."""

    def __init__(self, obs_dim, action_dim, target_parameters, local_parameters):
        super().__init__(obs_dim, action_dim, target_parameters - local_parameters)
        self.reflex = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim)
        )

    def forward(self, obs, state):
        return self.net(obs) + self.reflex(obs), state


def build_policy(
    graph_path: Path,
    obs_dim: int,
    action_dim: int,
    method: str,
    seed: int,
    device: str,
    readout="linear",
    log_std=-1.0,
):
    torch.manual_seed(seed)
    reflex_parameters = (obs_dim + 1) * 64 + 65 * action_dim
    expected = (
        18756 * (obs_dim + 1)
        + (109 + 1) * action_dim
        + (218 if readout in {"normalized", "reflex"} else 0)
    )
    if readout == "reflex":
        expected += reflex_parameters
    if method == "connectome":
        actor = GraphActor(graph_path, obs_dim, action_dim, readout)
    elif method == "mlp":
        actor = (
            SharedReflexMLPActor(obs_dim, action_dim, expected, reflex_parameters)
            if readout == "reflex"
            else MLPActor(obs_dim, action_dim, expected)
        )
    else:
        raise ValueError((method, readout))
    final = (
        actor.core.output_layer
        if method == "connectome" and readout == "linear"
        else actor.core.output_layer[-1] if method == "connectome" else actor.net[-1]
    )
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    if readout == "reflex" and method in {"connectome", "mlp"}:
        # Identical local branch initialization for its explicit ablation.
        branch = actor.reflex
        with torch.random.fork_rng():
            torch.manual_seed(seed + 400000)
            branch[0].reset_parameters()
        nn.init.zeros_(branch[-1].weight)
        nn.init.zeros_(branch[-1].bias)
    actual = sum(v.numel() for v in actor.parameters() if v.requires_grad)
    assert abs(actual / expected - 1) < 0.005
    with torch.random.fork_rng():
        torch.manual_seed(seed + 200000)
        critic = ValueHead(obs_dim, [256, 256], activation="relu")
    return GaussianCoordinatePolicy(actor, critic, action_dim, log_std_init=log_std).to(
        device
    )

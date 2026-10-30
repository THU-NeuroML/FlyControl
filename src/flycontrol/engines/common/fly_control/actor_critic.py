"""Critic and actor-critic interface used by the Gaussian policy."""

from __future__ import annotations
from typing import Sequence, Tuple
import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    """Resolve the configured activation; the Figure 6 critic uses ReLU."""
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
    }
    try:
        return activations[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {name!r}; choose from {sorted(activations)}"
        ) from exc


def mlp(
    in_dim: int, hidden: list[int], out_dim: int, act: type[nn.Module] = nn.Tanh
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ValueHead(nn.Module):
    """State-less MLP critic on a flat observation vector."""

    def __init__(
        self, obs_dim: int, hidden: list[int], activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1, act=_activation(activation))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class Policy(nn.Module):
    """Stochastic actor-critic with optional persistent hidden state."""

    def __init__(
        self,
        actor: nn.Module,
        critic: ValueHead,
        action_dim: int,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    @property
    def has_state(self) -> bool:
        return self.actor.has_state

    @property
    def state_dim(self) -> int:
        return self.actor.state_dim

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.actor.init_state(batch, device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mean, hidden_next, value)."""
        (mean, h_next) = self.actor(obs, hidden)
        value = self.critic(obs)
        return (mean, h_next, value)

    def distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

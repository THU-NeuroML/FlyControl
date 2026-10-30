"""Filter-relative actor perturbations; frozen graph and critic stay unchanged."""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterable, Iterator
import torch
from fly_control.actor_critic import Policy


def actor_filter_relative_direction(
    policy: Policy,
    *,
    seed: int,
    minimum_scale: float = 1e-6,
) -> ParameterDirection:
    """Draw a Li-et-al.-style filter-normalized actor direction.

    Rows of matrix-valued parameters are treated as filters and matched to the
    corresponding trained row norm.  Vector-valued parameters are normalized
    per tensor.  This removes a major scale confound in 2-D parameter-space
    visualizations, but does not by itself make sharpness invariant to all
    neural-network reparameterizations.
    """
    generator = torch.Generator(device=next(policy.parameters()).device)
    generator.manual_seed(int(seed))
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in policy.actor.named_parameters():
        if not parameter.requires_grad:
            continue
        raw = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        detached = parameter.detach()
        if parameter.ndim >= 2:
            raw_rows = raw.reshape(raw.shape[0], -1)
            parameter_rows = detached.reshape(detached.shape[0], -1)
            raw_norm = torch.linalg.vector_norm(
                raw_rows, dim=1, keepdim=True
            ).clamp_min(1e-30)
            floor = minimum_scale * max(raw_rows.shape[1] ** 0.5, 1.0)
            parameter_norm = torch.linalg.vector_norm(
                parameter_rows, dim=1, keepdim=True
            ).clamp_min(floor)
            tensors[name] = (raw_rows / raw_norm * parameter_norm).reshape_as(parameter)
        else:
            raw_norm = torch.linalg.vector_norm(raw).clamp_min(1e-30)
            floor = minimum_scale * max(parameter.numel() ** 0.5, 1.0)
            parameter_norm = torch.linalg.vector_norm(detached).clamp_min(floor)
            tensors[name] = raw / raw_norm * parameter_norm
    if not tensors:
        raise ValueError("policy actor has no trainable parameters")
    return ParameterDirection(tensors)


@contextmanager
def multiply_perturbed_actor(
    policy: Policy,
    perturbations: Iterable[tuple[ParameterDirection, float]],
) -> Iterator[Policy]:
    """Apply several actor directions atomically and restore them on exit."""
    perturbations = tuple(perturbations)
    parameters = dict(policy.actor.named_parameters())
    for direction, _ in perturbations:
        unknown = set(direction.tensors) - set(parameters)
        if unknown:
            raise KeyError(
                f"direction contains unknown actor parameters: {sorted(unknown)}"
            )
    affected = {name for direction, _ in perturbations for name in direction.tensors}
    originals = {name: parameters[name].detach().clone() for name in affected}
    with torch.no_grad():
        for direction, radius in perturbations:
            for name, delta in direction.tensors.items():
                parameters[name].add_(delta, alpha=float(radius))
    try:
        yield policy
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                parameters[name].copy_(original)


def policy_mean_on_windows(
    policy: Policy,
    observation_windows: torch.Tensor,
    initial_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Replay fixed observation windows while evolving the policy state."""
    if observation_windows.ndim != 3:
        raise ValueError(
            "observation_windows must have shape (batch, horizon, obs_dim)"
        )
    if observation_windows.shape[1] < 1:
        raise ValueError("observation windows must contain at least one step")
    if initial_hidden_states.shape[0] != observation_windows.shape[0]:
        raise ValueError("one initial hidden state is required per window")
    hidden = initial_hidden_states
    means = []
    with torch.no_grad():
        for step in range(observation_windows.shape[1]):
            mean, hidden, _ = policy(observation_windows[:, step], hidden)
            means.append(mean)
    return torch.stack(means, dim=1)


@dataclass(frozen=True)
class ParameterDirection:
    tensors: dict[str, torch.Tensor]


def direction_cosine(first: ParameterDirection, second: ParameterDirection) -> float:
    if set(first.tensors) != set(second.tensors):
        raise ValueError("directions must contain identical actor parameters")
    numerator = sum(
        (first.tensors[name] * second.tensors[name]).sum() for name in first.tensors
    )
    first_norm = torch.sqrt(
        sum(value.square().sum() for value in first.tensors.values())
    )
    second_norm = torch.sqrt(
        sum(value.square().sum() for value in second.tensors.values())
    )
    denominator = first_norm * second_norm
    if float(denominator) == 0.0:
        raise FloatingPointError("cannot compare a zero parameter direction")
    return float(numerator / denominator)

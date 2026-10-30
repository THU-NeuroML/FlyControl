"""Candidate C1: fixed synaptic graph with bounded group-level dynamics.

An engineered, connectome-constrained model: learned time constants and gains
are not experimentally measured physiology. Exact/null must share groups,
parameterization, initialization, and training opportunity.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .models import ConnectomeRateCore


def load_dynamics_groups(data, config):
    """Validate external group provenance and exact node order before loading."""
    config = dict(config)
    unknown = set(config) - {"groups_path", "groups_sha256", "tau_bounds", "gain_bounds", "bias_bound"}
    if unknown:
        raise ValueError(f"unknown group_dynamics keys: {sorted(unknown)}")
    path = Path(config.pop("groups_path")).resolve()
    expected_hash = config.pop("groups_sha256")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("dynamics group file checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        ids, groups, labels = (arrays[k] for k in ("root_ids", "group_ids", "group_labels"))
    ordered_ids = np.asarray((*data.sensory_ids, *data.internal_ids, *data.output_ids))
    if not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, ordered_ids):
        raise ValueError("dynamics root IDs must exactly match the graph partition order")
    if not np.issubdtype(groups.dtype, np.integer) or groups.shape != (data.N,):
        raise ValueError("dynamics groups must contain one integer per graph node")
    if labels.ndim != 1 or not np.array_equal(np.unique(groups), np.arange(len(labels))):
        raise ValueError("dynamics labels must cover all contiguous groups")
    if len(set(map(str, labels))) != len(labels):
        raise ValueError("dynamics group labels must be unique")
    manifest = {"groups_path": str(path), "groups_sha256": actual_hash,
                "num_groups": len(labels), "group_labels": list(map(str, labels)),
                "group_counts": np.bincount(groups).tolist()}
    return {**config, "group_ids": groups.copy()}, manifest


class GroupAdaptiveRateCore(ConnectomeRateCore):
    """Three parameters per predeclared annotation group, without edge learning."""

    def __init__(self, data, input_dim, output_dim, *, group_ids: Sequence[int],
                 tau_bounds=(.0125, .2), gain_bounds=(.5, 1.5), bias_bound=.5,
                 **kwargs):
        super().__init__(data, input_dim, output_dim, **kwargs)
        groups = torch.as_tensor(group_ids, dtype=torch.long)
        if groups.shape != (data.N,) or torch.any(groups < 0):
            raise ValueError("group_ids must give one nonnegative group per node")
        unique = torch.unique(groups)
        if not torch.equal(unique, torch.arange(len(unique))):
            raise ValueError("group IDs must be contiguous from zero")
        self.register_buffer("dynamics_group_ids", groups)
        self.tau_bounds = tuple(float(x) for x in tau_bounds)
        self.gain_bounds = tuple(float(x) for x in gain_bounds)
        self.bias_bound = float(bias_bound)
        if not (self.dt < self.tau_bounds[0] < float(self.rate_tau) < self.tau_bounds[1]):
            raise ValueError("tau bounds must exceed dt and contain initial tau")
        if not (0 < self.gain_bounds[0] < 1. < self.gain_bounds[1]):
            raise ValueError("gain bounds must be positive and contain native gain one")
        if not math.isfinite(self.bias_bound) or self.bias_bound <= 0:
            raise ValueError("bias bound must be finite and positive")
        if not all(math.isfinite(x) for x in (*self.tau_bounds, *self.gain_bounds)):
            raise ValueError("bounds must be finite")

        def initial_logit(value, bounds):
            fraction = (value - bounds[0]) / (bounds[1] - bounds[0])
            return math.log(fraction / (1. - fraction))

        self.group_tau_logit = nn.Parameter(torch.full(
            (len(unique),), initial_logit(float(self.rate_tau), self.tau_bounds)))
        self.group_gain_logit = nn.Parameter(torch.full(
            (len(unique),), initial_logit(1., self.gain_bounds)))
        self.group_bias_raw = nn.Parameter(torch.zeros(len(unique)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "dynamics_group_ids")
        if saved is not None and not torch.equal(saved.cpu(), self.dynamics_group_ids.cpu()):
            error_msgs.append("checkpoint dynamics groups differ from the validated group file")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def effective_dynamics(self):
        tau = self.tau_bounds[0] + (self.tau_bounds[1] - self.tau_bounds[0]) * self.group_tau_logit.sigmoid()
        gain = self.gain_bounds[0] + (self.gain_bounds[1] - self.gain_bounds[0]) * self.group_gain_logit.sigmoid()
        bias = self.bias_bound * self.group_bias_raw.tanh()
        return tuple(v[self.dynamics_group_ids] for v in (tau, gain, bias))

    def step(self, state, inputs):
        external = torch.zeros_like(state)
        projected = self.input_proj(inputs)
        if self.input_mixing is not None:
            projected = projected @ self.input_mixing
        external[:, self.input_start:self.input_start + self.input_width] = self._external_input_gain * projected
        tau, gain, bias = self.effective_dynamics()
        full_gate = torch.clamp(self.dt / tau, 0., 1.)
        gate = 1. - torch.pow(1. - full_gate, 1. / self.internal_substeps)
        next_state = state
        for _ in range(self.internal_substeps):
            recurrent = self.recurrent_drive(torch.tanh(next_state))
            target = torch.tanh(self._recurrent_gain * gain * recurrent + external + bias)
            next_state = (1. - gate) * next_state + gate * target
        output_rates = self._neural_readout_gain * torch.tanh(next_state)[:, self.S + self.I:]
        output = (output_rates @ self.fixed_output_readout if self.fixed_output_readout is not None
                  else self.output_layer(output_rates))
        return next_state, output

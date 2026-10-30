"""Shared fixed adult-connectome rate dynamics for all five bodies."""

from __future__ import annotations
import torch
from torch import nn
from .graph import ConnectomeData


class ConnectomeRateCore(nn.Module):
    """Pure sparse fixed-time-constant rate dynamics for an adult graph."""

    mode = "rate"

    def __init__(
        self,
        data: ConnectomeData,
        input_dim: int,
        output_dim: int,
        *,
        dt: float,
        tau: float,
        input_target: str,
        readout_hidden: int | None = None,
        internal_substeps: int = 1,
        initial_state_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not data.is_sparse:
            raise ValueError("connectome_rate requires a sparse connectome")
        if dt <= 0 or tau <= 0:
            raise ValueError("dt and tau must be positive")
        if not isinstance(internal_substeps, int) or internal_substeps <= 0:
            raise ValueError("internal_substeps must be a positive integer")
        if initial_state_std < 0.0:
            raise ValueError("initial_state_std must be non-negative")
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        self.N = data.N
        self.S = data.sensory_dim
        self.I = data.internal_dim
        self.O = data.output_dim
        if input_target == "sensory":
            (self.input_start, self.input_width) = (0, self.S)
        elif input_target == "internal":
            (self.input_start, self.input_width) = (self.S, self.I)
        elif input_target == "all":
            (self.input_start, self.input_width) = (0, self.N)
        else:
            raise ValueError("input_target must be sensory, internal, or all")
        self.register_buffer(
            "edge_pre", torch.as_tensor(data.edge_pre, dtype=torch.long)
        )
        self.register_buffer(
            "edge_post", torch.as_tensor(data.edge_post, dtype=torch.long)
        )
        self.register_buffer(
            "edge_weight", torch.as_tensor(data.edge_weight, dtype=torch.float32)
        )
        self.register_buffer("rate_tau", torch.tensor(float(tau), dtype=torch.float32))
        self.dt = float(dt)
        self.internal_substeps = internal_substeps
        self.initial_state_std = float(initial_state_std)
        self._recurrent_gain = 1.0
        self._external_input_gain = 1.0
        self._neural_readout_gain = 1.0
        self.input_proj = nn.Linear(input_dim, self.input_width)
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        if readout_hidden is None:
            self.output_layer = nn.Linear(self.O, output_dim)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
            nn.init.zeros_(self.output_layer.bias)
        else:
            if readout_hidden <= 0:
                raise ValueError("readout_hidden must be positive")
            self.output_layer = nn.Sequential(
                nn.Linear(self.O, int(readout_hidden)),
                nn.ReLU(),
                nn.Linear(int(readout_hidden), output_dim),
            )
            for module in self.output_layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.initial_state_std == 0.0:
            return torch.zeros(batch, self.N, device=device)
        return self.initial_state_std * torch.randn(batch, self.N, device=device)

    def step(
        self, state: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        external = torch.zeros_like(state)
        start = self.input_start
        projected = self.input_proj(inputs)
        external[:, start : start + self.input_width] = (
            self._external_input_gain * projected
        )
        full_step_gate = torch.clamp(self.dt / self.rate_tau, 0.0, 1.0)
        gate = 1.0 - torch.pow(1.0 - full_step_gate, 1.0 / self.internal_substeps)
        next_state = state
        for _ in range(self.internal_substeps):
            rates = torch.tanh(next_state)
            recurrent = self.recurrent_drive(rates)
            target = torch.tanh(self._recurrent_gain * recurrent + external)
            next_state = (1.0 - gate) * next_state + gate * target
        output_rates = (
            self._neural_readout_gain * torch.tanh(next_state)[:, self.S + self.I :]
        )
        assert self.output_layer is not None
        output = self.output_layer(output_rates)
        return (next_state, output)

    def recurrent_drive(self, rates: torch.Tensor) -> torch.Tensor:
        messages = rates.index_select(1, self.edge_pre) * self.edge_weight
        drive = torch.zeros_like(rates)
        drive.index_add_(1, self.edge_post, messages)
        return drive

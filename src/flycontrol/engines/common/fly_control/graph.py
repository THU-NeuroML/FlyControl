"""SIO-ordered sparse adult-connectome data used by the controller."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConnectomeData:
    W: np.ndarray | None  # Dense (N, N) SIO-ordered adjacency
    sensory_dim: int
    internal_dim: int
    output_dim: int
    sensory_ids: tuple
    internal_ids: tuple
    output_ids: tuple
    edge_pre: np.ndarray | None = None
    edge_post: np.ndarray | None = None
    edge_weight: np.ndarray | None = None
    metadata: dict | None = None

    @property
    def N(self) -> int:
        return self.sensory_dim + self.internal_dim + self.output_dim

    @property
    def is_sparse(self) -> bool:
        return self.edge_pre is not None

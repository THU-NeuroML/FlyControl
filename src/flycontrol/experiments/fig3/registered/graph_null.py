from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ConnectomeData:
    W: np.ndarray | None
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


def _presynaptic_signs(data: ConnectomeData) -> np.ndarray:
    """Infer a Dale-like sign for every presynaptic neuron.

    Adult and BANC edge signs are assigned from presynaptic transmitter
    identity before row-L1 normalization.  A few rows have no retained
    outgoing edge; those rows receive the majority sign so an ER draw that
    happens to use them still has a well-defined rate-network sign.
    """
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_pre is not None
        assert data.edge_weight is not None
        row_sum = np.bincount(
            data.edge_pre, weights=data.edge_weight.astype(np.float64), minlength=data.N
        )
    else:
        row_sum = data.W.astype(np.float64).sum(axis=1)
    signs = np.sign(row_sum).astype(np.float32)
    nonzero = signs != 0
    majority = 1.0
    if nonzero.any() and float(signs[nonzero].sum()) < 0:
        majority = -1.0
    signs[~nonzero] = majority
    return signs


def _row_l1_weights(
    edge_pre: np.ndarray, row_sign: np.ndarray, n_rows: int, global_gain: float
) -> np.ndarray:
    """Return equal-magnitude, Dale-like weights with exact row-L1 gain."""
    degree = np.bincount(edge_pre, minlength=n_rows)
    denom = degree[edge_pre].astype(np.float32)
    return (
        row_sign[edge_pre]
        * np.float32(global_gain)
        / np.maximum(denom, np.float32(1.0))
    ).astype(np.float32)


def size_density_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Exact size-and-density random rate graph.

    This is the deliberately weak ER-SD null used in the NMI baseline ladder.
    It preserves N, E and the S/I/O node partition.  The same S and O indices
    therefore remain the policy input/output gateways.  Recurrent edges are
    sampled uniformly without duplicates, and weights are Dale-like with the
    same row-L1/global-gain operating scale as the biological graph.

    It does *not* preserve node degrees or S/I/O block counts; those are the
    purpose of :func:`strict_block_degree_rewire`.
    """
    rng = np.random.default_rng(seed)
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_weight is not None
        edge_count = int(data.edge_weight.size)
    else:
        edge_count = int(np.count_nonzero(data.W))
    (edge_pre, edge_post) = _sample_unique_pairs(rng, data.N, data.N, edge_count)
    row_sign = _presynaptic_signs(data)
    gain = float((data.metadata or {}).get("global_gain", 1.0))
    edge_weight = _row_l1_weights(edge_pre, row_sign, data.N, global_gain=gain)
    metadata = _randomized_metadata(data, "size_density_random", seed)
    metadata.update(
        {
            "null_family": "ER-SD",
            "preserves_exact_N": True,
            "preserves_exact_E": True,
            "preserves_sio_node_partition": True,
            "preserves_node_degrees": False,
            "weight_scaling": "presynaptic_sign_equal_weight_row_l1",
        }
    )
    if data.W is None:
        return _replace(
            data,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_weight=edge_weight,
            metadata=metadata,
        )
    W = np.zeros_like(data.W)
    W[edge_pre, edge_post] = edge_weight
    return _replace(data, W=W, metadata=metadata)


def _require_sparse_edges(data: ConnectomeData) -> None:
    if data.edge_pre is None or data.edge_post is None or data.edge_weight is None:
        raise ValueError("sparse connectome requires edge_pre/post/weight")


def _sample_unique_pairs(
    rng: np.random.Generator, n_pre: int, n_post: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    capacity = int(n_pre) * int(n_post)
    if count > capacity:
        raise ValueError("cannot sample more unique edges than block capacity")
    if count == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    keys = np.array([], dtype=np.int64)
    while keys.size < count:
        missing = count - keys.size
        batch = max(missing * 2, 16)
        pre = rng.integers(0, n_pre, size=batch, dtype=np.int64)
        post = rng.integers(0, n_post, size=batch, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, pre * int(n_post) + post]))
    if keys.size > count:
        keys = rng.choice(keys, size=count, replace=False)
    keys.sort()
    return (
        (keys // int(n_post)).astype(np.int64),
        (keys % int(n_post)).astype(np.int64),
    )


def _randomized_metadata(
    data: ConnectomeData, kind: str, seed: int, **details: object
) -> dict:
    metadata = dict(data.metadata or {})
    metadata.update(randomization_kind=kind, randomize_seed=int(seed), **details)
    metadata["kind"] = kind
    return metadata


def _replace(data: ConnectomeData, **kwargs) -> ConnectomeData:
    """dataclass replacement helper (frozen dataclass)."""
    base = dict(
        W=data.W,
        sensory_dim=data.sensory_dim,
        internal_dim=data.internal_dim,
        output_dim=data.output_dim,
        sensory_ids=data.sensory_ids,
        internal_ids=data.internal_ids,
        output_ids=data.output_ids,
        edge_pre=data.edge_pre,
        edge_post=data.edge_post,
        edge_weight=data.edge_weight,
        metadata=data.metadata,
    )
    base.update(kwargs)
    return ConnectomeData(**base)

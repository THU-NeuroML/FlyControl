"""Serializable diagnostics for dense and sparse connectomes."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    from .connectome import ConnectomeData


def _sparse_matrix(data: ConnectomeData) -> sp.csr_matrix:
    if data.is_sparse:
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        matrix = sp.coo_matrix(
            (data.edge_weight, (data.edge_pre, data.edge_post)),
            shape=(data.N, data.N),
        ).tocsr()
        matrix.sum_duplicates()
        return matrix
    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    return sp.csr_matrix(data.W)


def estimate_spectral_radius(
    data: ConnectomeData, *, tolerance: float = 1e-7, max_iterations: int = 5000
) -> float:
    """Estimate ``max(abs(eigenvalue))`` without densifying sparse graphs."""
    matrix = _sparse_matrix(data).astype(np.float64)
    if matrix.nnz == 0 or data.N == 0:
        return 0.0
    if data.N == 1:
        return abs(float(matrix[0, 0]))
    if data.N == 2:
        a = float(matrix[0, 0])
        b = float(matrix[0, 1])
        c = float(matrix[1, 0])
        d = float(matrix[1, 1])
        values = np.roots([1.0, -(a + d), a * d - b * c])
        return float(np.max(np.abs(values)))

    start = np.full(data.N, 1.0 / np.sqrt(data.N), dtype=np.float64)
    try:
        values = spla.eigs(
            matrix,
            k=1,
            which="LM",
            v0=start,
            tol=tolerance,
            maxiter=max_iterations,
            return_eigenvectors=False,
        )
    except spla.ArpackNoConvergence as error:
        if error.eigenvalues is None or len(error.eigenvalues) == 0:
            raise RuntimeError("spectral-radius estimation did not converge") from error
        values = error.eigenvalues
    return float(np.max(np.abs(values)))


def _digest(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.sort(values))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _degree_summary(values: np.ndarray) -> dict:
    unique, counts = np.unique(values, return_counts=True)
    return {
        "min": int(values.min()) if values.size else 0,
        "max": int(values.max()) if values.size else 0,
        "mean": float(values.mean()) if values.size else 0.0,
        "histogram": [
            [int(degree), int(count)] for degree, count in zip(unique, counts)
        ],
        "sorted_multiset_sha256": _digest(values.astype(np.int64, copy=False)),
    }


def _weight_summary(values: np.ndarray) -> dict:
    signs, counts = np.unique(np.sign(values), return_counts=True)
    sign_counts = {"negative": 0, "zero": 0, "positive": 0}
    names = {-1.0: "negative", 0.0: "zero", 1.0: "positive"}
    for sign, count in zip(signs, counts):
        sign_counts[names[float(sign)]] = int(count)
    quantiles = (
        np.quantile(values.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
        if values.size
        else np.zeros(5)
    )
    return {
        "count": int(values.size),
        "dtype": str(values.dtype),
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "quantiles": {
            name: float(value)
            for name, value in zip(("q0", "q25", "q50", "q75", "q100"), quantiles)
        },
        "sign_counts": sign_counts,
        "sorted_multiset_sha256": _digest(values),
    }


def diagnose_connectome(
    data: ConnectomeData, reference: ConnectomeData | None = None
) -> dict:
    """Return exact structural summaries and sparse spectral diagnostics."""
    matrix = _sparse_matrix(data)
    coo = matrix.tocoo()
    pre = coo.row.astype(np.int64, copy=False)
    post = coo.col.astype(np.int64, copy=False)
    weights = coo.data
    out_degree = np.bincount(pre, minlength=data.N)
    in_degree = np.bincount(post, minlength=data.N)

    bounds = np.array(
        [0, data.sensory_dim, data.sensory_dim + data.internal_dim, data.N],
        dtype=np.int64,
    )
    pre_part = np.searchsorted(bounds[1:], pre, side="right")
    post_part = np.searchsorted(bounds[1:], post, side="right")
    partition_edges = np.zeros((3, 3), dtype=np.int64)
    np.add.at(partition_edges, (pre_part, post_part), 1)
    radius = estimate_spectral_radius(data)

    result = {
        "node_count": int(data.N),
        "edge_count": int(matrix.nnz),
        "storage": "sparse_edges" if data.is_sparse else "dense",
        "partitions": {
            "sensory": int(data.sensory_dim),
            "internal": int(data.internal_dim),
            "output": int(data.output_dim),
        },
        "partition_edge_counts": partition_edges.tolist(),
        "in_degree": _degree_summary(in_degree),
        "out_degree": _degree_summary(out_degree),
        "weights": _weight_summary(weights),
        "spectral_radius_estimate": radius,
    }
    if reference is not None:
        reference_radius = estimate_spectral_radius(reference)
        absolute_error = abs(radius - reference_radius)
        result["spectral_radius_match"] = {
            "reference": reference_radius,
            "absolute_error": absolute_error,
            "relative_error": (
                absolute_error / reference_radius
                if reference_radius > np.finfo(float).eps
                else absolute_error
            ),
        }
    return result

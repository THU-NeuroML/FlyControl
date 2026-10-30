"""Exact paired sign-flip counting without allocating all 2**n sign vectors."""

import numpy as np


def signflip(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite vector")
    threshold = abs(values.sum()) - 1e-12
    if threshold <= 0:
        return 1.0

    def sums(v):
        result = np.zeros(1)
        for value in v:
            result = np.concatenate([result + value, result - value])
        return result

    left = sums(values[: len(values) // 2])
    right = np.sort(sums(values[len(values) // 2 :]))
    lower = np.searchsorted(right, -threshold - left, side="right").sum()
    upper = (len(right) - np.searchsorted(right, threshold - left, side="left")).sum()
    return float(lower + upper) / (1 << len(values))

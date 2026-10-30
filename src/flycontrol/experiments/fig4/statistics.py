"""Original exact paired sign-flip and seed-bootstrap statistics."""

import numpy as np


def signflip(x):
    x = np.asarray(x, dtype=float)
    threshold = abs(x.sum()) - 1e-12
    count = 0
    total = 2 ** len(x)
    for start in range(0, total, 32768):
        bits = (
            np.arange(start, min(total, start + 32768), dtype=np.uint32)[:, None]
            >> np.arange(len(x), dtype=np.uint32)
        ) & 1
        sums = (2 * bits.astype(float) - 1) @ x
        count += int((np.abs(sums) >= threshold).sum())
    return count / total


def holm(p):
    order = np.argsort(p)
    out = np.empty(len(p))
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(p) - rank) * p[i]))
        out[i] = previous
    return out


def contrast(x, random_seed=20261020):
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all()
    indices = np.random.default_rng(random_seed).integers(0, len(x), (20000, len(x)))
    lo, hi = np.quantile(x[indices].mean(axis=1), [0.025, 0.975])
    sd = float(x.std(ddof=1))
    return dict(
        difference_mean=float(x.mean()),
        difference_ci95_low=float(lo),
        difference_ci95_high=float(hi),
        positive_pairs=int((x > 0).sum()),
        negative_pairs=int((x < 0).sum()),
        tied_pairs=int((x == 0).sum()),
        n_pairs=len(x),
        paired_standardized_difference=float(x.mean() / sd) if sd > 0 else None,
        raw_two_sided_signflip_p=signflip(x),
    )

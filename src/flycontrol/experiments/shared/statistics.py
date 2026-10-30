"""Exact paired sign-flip counting without allocating all 2**n sign vectors."""

import itertools
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


def inference(a,l,rng):
    assert len(a)==len(l)==6 and np.isfinite(a).all() and np.isfinite(l).all()
    pool=np.r_[a,l];inds=np.array(list(itertools.combinations(range(12),6)));tot=pool[inds].sum(1);delta=l.mean()-a.mean();null=tot/6-(pool.sum()-tot)/6
    boot=l[rng.integers(0,6,(20000,6))].mean(1)-a[rng.integers(0,6,(20000,6))].mean(1);ci=np.quantile(boot,[.025,.975])
    return dict(adult_mean=a.mean(),adult_sd=a.std(ddof=1),larva_mean=l.mean(),larva_sd=l.std(ddof=1),difference=delta,ci_low=ci[0],ci_high=ci[1],p_raw=np.mean(abs(null)>=abs(delta)-1e-12),n_adult=6,n_larva=6)

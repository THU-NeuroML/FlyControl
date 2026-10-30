import numpy as np
import pandas as pd
from .paths import DATA, OUTPUT

BOOT = np.random.default_rng(2026102101).integers(0, 20, (20000, 20))


def summary(v):
    v = np.asarray(v)
    assert len(v) == 20 and np.isfinite(v).all()
    low, high = np.quantile(v[BOOT].mean(1), [0.025, 0.975])
    return dict(mean=float(v.mean()), ci_low=float(low), ci_high=float(high), n=20)


def summarize_groups(d, keys, value):
    rows = []
    for k, q in d.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(keys, k), **summary(q.sort_values("seed")[value])))
    return pd.DataFrame(rows)


def main():
    frames = []
    for seed in range(2011, 2031):
        folder = DATA / "mechanisms/er_atlas" / str(seed)
        if not folder.exists():
            folder = DATA / "mechanisms/er_atlas" / f"er_seed{seed}"
        frame = pd.read_csv(folder / "atlas_40ms.csv")
        assert len(frame) == 45 and set(frame.seed) == {seed}
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTPUT / "source_data/MATCHED_ER_ATLAS_ALL.csv", index=False)
    summarize_groups(data, ["group", "height"], "enrichment").to_csv(
        OUTPUT / "source_data/MATCHED_ER_ATLAS_MEANS.csv", index=False
    )

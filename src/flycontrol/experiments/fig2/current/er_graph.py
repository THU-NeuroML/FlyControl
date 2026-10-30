"""Regenerate the frozen directed fixed-edge-count ER null, without self edges."""

from pathlib import Path
import argparse, hashlib, json
import numpy as np
from flycontrol.experiments.fig2.current.paths import DATA


def generate(seed, output):
    if seed not in range(2011, 2031):
        raise ValueError("Expected current training seed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "graph.npz") as z:
        original = {k: z[k] for k in z.files}
    n = len(original["root_ids"])
    weights = original["edge_weight"]
    m = len(weights)
    rng = np.random.default_rng(202610210000 + seed)
    codes = rng.choice(n * (n - 1), size=m, replace=False, shuffle=True)
    pre, off = codes // (n - 1), codes % (n - 1)
    post = off + (off >= pre)
    w = weights[rng.permutation(m)]
    order = np.lexsort((pre, post))
    result = dict(
        original, edge_pre=pre[order], edge_post=post[order], edge_weight=w[order]
    )
    assert len(np.unique(codes)) == m and np.all(pre != post)
    np.testing.assert_array_equal(np.sort(w), np.sort(weights))
    dest = output / f"er_seed{seed}.npz"
    np.savez_compressed(dest, **result)
    print(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    generate(a.seed, a.output)

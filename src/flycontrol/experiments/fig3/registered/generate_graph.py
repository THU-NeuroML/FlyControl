"""Regenerate the registered ER-SD graph without changing its node gateways."""

import argparse
from pathlib import Path
import numpy as np
from .common import DATA
from .graph_null import ConnectomeData, size_density_random


def generate(seed, output):
    z = np.load(DATA / "graph.npz")
    dims = [int(z[k]) for k in ["sensory_dim", "internal_dim", "output_dim"]]
    graph = ConnectomeData(
        None,
        *dims,
        (),
        (),
        (),
        edge_pre=z["edge_pre"],
        edge_post=z["edge_post"],
        edge_weight=z["edge_weight"],
    )
    result = size_density_random(graph, 830000 + seed)
    arrays = {k: getattr(result, k) for k in ('edge_pre', 'edge_post', 'edge_weight', 'sensory_dim', 'internal_dim', 'output_dim')}
    arrays['root_ids'] = z['root_ids']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Generated registered graph seed {seed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=range(4301, 4325), default=4301)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    generate(a.seed, a.output)

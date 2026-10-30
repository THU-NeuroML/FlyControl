"""Load the shared FlyWire v783 graph; reject a different graph configuration."""

from pathlib import Path
import hashlib, os
import numpy as np
from fly_control.graph import ConnectomeData


def build_connectome(cfg=None):
    from flycontrol.paths import run_root, read_json, CONFIGS
    ROOT = run_root()

    cfg = cfg or {}
    expected = read_json(CONFIGS / "inputs/motor.json")
    if (
        cfg.get("species", "adult") != "adult"
        or cfg.get("kind", "original") != "original"
    ):
        raise ValueError("The published experiments use the original adult graph")
    for key, value in expected["loader"].items():
        if key in cfg and cfg[key] != value:
            raise ValueError("Graph configuration differs: " + key)
    path = ROOT / "inputs/graph.npz"
    if not path.exists():
        raise FileNotFoundError("Run flycontrol prepare first")
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in ("edge_pre", "edge_post", "edge_weight")}
    for key, value in arrays.items():
        if (
            hashlib.sha256(value.tobytes()).hexdigest()
            != expected["arrays"][key]["sha256"]
        ):
            raise ValueError("Graph checksum differs: " + key)
    s, i, o = expected["dimensions"]
    return ConnectomeData(None, s, i, o, (), (), (), **arrays)


def diagnose_connectome(graph):
    weights = graph.edge_weight
    return dict(
        node_count=graph.N,
        edge_count=len(weights),
        storage="sparse",
        partitions=dict(
            sensory=graph.sensory_dim,
            internal=graph.internal_dim,
            output=graph.output_dim,
        ),
        weights=dict(
            sorted_multiset_sha256=hashlib.sha256(
                np.sort(weights).tobytes()
            ).hexdigest()
        ),
        spectral_radius_estimate=None,
    )

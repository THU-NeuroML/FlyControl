"""Download pinned public inputs and build the experiment's sparse graph."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import os
import shutil
import urllib.request
import time

from .paths import CONFIGS, ENGINES, read_json, write_json, sha256


def fetch(record, destination):
    destination = Path(destination).resolve()
    target = destination / record["path"]
    if not target.resolve().is_relative_to(destination):
        raise ValueError("Input path escapes destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != record["sha256"]:
            raise ValueError(f"Input checksum mismatch: {target}")
        return target
    temporary = target.with_name(target.name + f".{os.getpid()}.part")
    try:
        for attempt in range(3):
            try:
                request = urllib.request.Request(record["url"], headers={"User-Agent": "FlyControl/0.1"})
                with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, 1024 * 1024)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if temporary.stat().st_size != record["bytes"] or sha256(temporary) != record["sha256"]:
            raise ValueError(f"Downloaded input failed checksum: {record['url']}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download(records, root):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda record: fetch(record, root), records))


def build_graph(root, name):
    import numpy as np
    from .connectomes.adult import load_adult

    cfg = read_json(CONFIGS / "inputs" / f"{name}.json")
    output = root / "inputs/graph.npz"
    graph = load_adult(root / "inputs/flywire783", **cfg["loader"])
    ids = np.asarray(graph.sensory_ids + graph.internal_ids + graph.output_ids, dtype=np.int64)
    arrays = {key: getattr(graph, key) for key in ("edge_pre", "edge_post", "edge_weight")}
    if name == "terrain":
        # Terrain preserves the motor graph's edge order while repartitioning
        # the nodes. Summation order is part of the numerical protocol.
        motor_cfg = read_json(CONFIGS / "inputs/motor.json")
        motor = load_adult(root / "inputs/flywire783", **motor_cfg["loader"])
        # The terrain input partition was extracted from the descending
        # partition, preserving that partition's order (not globally sorting
        # the new internal nodes). Its edges retain the motor input order.
        sensory = set(graph.sensory_ids)
        descending = set(graph.output_ids)
        full = set(map(int, ids))
        prior = list(motor.sensory_ids) + sorted(full - set(motor.sensory_ids) - descending) + list(graph.output_ids)
        ids = np.asarray([x for x in prior if x in sensory] +
                         [x for x in prior if x not in sensory and x not in descending] +
                         [x for x in prior if x in descending], dtype=np.int64)
        index = {int(node): i for i, node in enumerate(ids)}
        permutation = np.asarray([index[int(node)] for node in motor.sensory_ids + motor.internal_ids + motor.output_ids])
        arrays = dict(edge_pre=permutation[motor.edge_pre], edge_post=permutation[motor.edge_post], edge_weight=motor.edge_weight)
    actual = {key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()}
    dims = [graph.sensory_dim, graph.internal_dim, graph.output_dim]
    if actual != cfg["arrays"] or dims != cfg["dimensions"]:
        raise ValueError(f"Graph differs from fixed input contract: {actual}; dimensions={dims}")
    np.savez_compressed(output, **arrays, root_ids=ids, sensory_dim=dims[0], internal_dim=dims[1], output_dim=dims[2])
    write_json(root / "inputs/graph.json", dict(name=name, dimensions=dims, arrays=actual, sha256=sha256(output)))
    return output


def prepare(root, spec):
    root = Path(root).resolve()
    manifest = read_json(CONFIGS / "inputs/flywire783.json")
    records = [dict(path=f"inputs/flywire783/{name}", url=row["source_url"], sha256=row["sha256"], bytes=row["bytes"])
               for name, row in manifest["files"].items()]
    download(records, root)
    graph = root / "inputs/graph.npz"
    if graph.exists():
        meta = read_json(root / "inputs/graph.json")
        if meta["name"] != spec["graph"] or sha256(graph) != meta["sha256"]:
            raise ValueError("Prepared graph changed; use a fresh run directory")
    else:
        build_graph(root, spec["graph"])
    family = spec["family"]
    if family == "ant":
        fetch(read_json(CONFIGS / "inputs/teacher.json"), root)
    if family in {"turning", "speed", "terrain"}:
        download(read_json(CONFIGS / "inputs/nmf_assets.json"), root)
        number = {"terrain": 2, "turning": 3, "speed": 4}[family]
        dest = root / f"data/figure{number}/graph.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(os.path.relpath(graph, dest.parent))
    elif family == "body" and spec["experiment"] != "tripod":
        records = read_json(CONFIGS / "inputs/robot_assets.json")
        body = spec["experiment"]
        records = [r for r in records if ("unitree_go1" in r["path"] if body == "go1" else body in r["path"] or r["path"].endswith(("LICENSE", "README.md")))]
        download(records, root)
    write_json(root / "inputs/complete.json", dict(graph_sha256=sha256(graph), graph=spec["graph"]))

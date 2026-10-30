"""Connectome loading and randomization controls.

Loads the larva connectome adjacency matrix in SIO order (sensory /
internal / output partition) and provides three randomization controls
required by the Q3/Q5 ablations:

  shuffled_weights         keep mask, shuffle non-zero weights
  degree_preserved_random  keep per-row in-degree, randomize targets
  erdos_renyi              i.i.d. random binary mask matched in density

A single ``ConnectomeData`` dataclass exposes everything downstream
modules need: matrix, partition sizes, ID lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class ConnectomeData:
    W: np.ndarray | None       # Dense (N, N) SIO-ordered adjacency
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


_OUTPUT_TYPES = {"DN-SEZ", "RGN"}


def _classify_neurons(annot: pd.DataFrame, sensory_type: str) -> tuple[list, list]:
    sensory_ids: list[int] = []
    output_ids: list[int] = []
    for _, row in annot.iterrows():
        cell_type = row["celltype"]
        ann = row.get("additional_annotations", "")
        for col in ("left_id", "right_id"):
            id_str = str(row[col]).lower()
            if id_str == "no pair":
                continue
            try:
                nid = int(id_str)
            except ValueError:
                continue
            if cell_type == "sensory":
                if sensory_type == "all" or ann == sensory_type:
                    sensory_ids.append(nid)
            elif cell_type in _OUTPUT_TYPES:
                output_ids.append(nid)
    return sorted(set(sensory_ids)), sorted(set(output_ids))


def load_larva(
    connectivity_csv: str | Path | None = None,
    annotation_csv: str | Path | None = None,
    rescale: float = 4e-2,
    sensory_type: str = "all",
) -> ConnectomeData:
    """Load the larva connectome and return SIO-ordered ``ConnectomeData``."""
    conn_path = Path(connectivity_csv or _DATA_DIR / "ad_connectivity_matrix.csv")
    annot_path = Path(annotation_csv or _DATA_DIR / "science.add9330_data_s2.csv")

    annot = pd.read_csv(annot_path)
    sensory_ids, output_ids = _classify_neurons(annot, sensory_type)

    df = pd.read_csv(conn_path, index_col=0) * rescale
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    matrix_ids = sorted(df.index.tolist())

    sensory_ids = [n for n in sensory_ids if n in df.index]
    output_ids = [n for n in output_ids if n in df.index]
    internal_ids = [n for n in matrix_ids
                    if n not in sensory_ids and n not in output_ids]
    ordered = sensory_ids + internal_ids + output_ids

    W = df.loc[ordered, ordered].values.astype(np.float32)
    return ConnectomeData(
        W=W,
        sensory_dim=len(sensory_ids),
        internal_dim=len(internal_ids),
        output_dim=len(output_ids),
        sensory_ids=tuple(sensory_ids),
        internal_ids=tuple(internal_ids),
        output_ids=tuple(output_ids),
        metadata={"species": "larva", "storage": "dense"},
    )


# ----------------------------------------------------------------------
# Adult connectome loader: full sparse graph, no dense N x N matrix.
# ----------------------------------------------------------------------


_ADULT_FILES = {
    "classification": "classification.csv",
    "connections": "connections_princeton.csv",
    "neurons": "neurons.csv",
}

_DEFAULT_NT_SIGNS = {
    "ACH": 1.0,
    "GLUT": 1.0,
    "GABA": -1.0,
    "DA": 1.0,
    "SER": 1.0,
    "OCT": 1.0,
}


def _adult_file(adult_dir: str | Path, key: str) -> Path:
    base = Path(adult_dir) / _ADULT_FILES[key]
    if base.exists():
        return base
    gz_path = Path(str(base) + ".gz")
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Missing {_ADULT_FILES[key]} in {adult_dir}")


def _norm_text(series: pd.Series, case_sensitive: bool = False) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text if case_sensitive else text.str.lower()


def _select_adult_ids(classification: pd.DataFrame,
                      selector: Mapping[str, object]) -> list[int]:
    root_ids = selector.get("root_ids")
    if root_ids is not None:
        return sorted({int(x) for x in root_ids})

    field = str(selector.get("field", "flow"))
    include = selector.get("include")
    exclude = selector.get("exclude", [])
    case_sensitive = bool(selector.get("case_sensitive", False))
    if field not in classification.columns:
        raise ValueError(f"adult classification has no field {field!r}")

    values = _norm_text(classification[field], case_sensitive)
    mask = pd.Series(True, index=classification.index)
    if include is not None:
        inc = {str(x).strip() if case_sensitive else str(x).strip().lower()
               for x in include}
        mask &= values.isin(inc)
    if exclude:
        exc = {str(x).strip() if case_sensitive else str(x).strip().lower()
               for x in exclude}
        mask &= ~values.isin(exc)
    return sorted(classification.loc[mask, "root_id"].astype(np.int64).unique())


def _load_adult_nt_types(adult_dir: str | Path) -> dict[int, str]:
    neurons = pd.read_csv(_adult_file(adult_dir, "neurons"),
                          usecols=["root_id", "nt_type"])
    rows = neurons.dropna()
    return {
        int(row.root_id): str(row.nt_type).strip().upper()
        for row in rows.itertuples(index=False)
        if str(row.nt_type).strip()
    }


def _collect_adult_connection_ids(connections_path: Path,
                                  chunksize: int) -> set[int]:
    ids: set[int] = set()
    for chunk in pd.read_csv(
        connections_path,
        usecols=["pre_root_id", "post_root_id"],
        chunksize=chunksize,
    ):
        ids.update(chunk["pre_root_id"].astype(np.int64).tolist())
        ids.update(chunk["post_root_id"].astype(np.int64).tolist())
    return ids


def _normalize_sparse_matrix(W: sp.csr_matrix, normalization: str,
                             global_gain: float) -> sp.csr_matrix:
    if normalization == "none":
        return (W * global_gain).astype(np.float32).tocsr()
    if normalization == "global_max_abs":
        max_abs = abs(W).max()
        scale = global_gain / max_abs if max_abs else 1.0
        return (W * scale).astype(np.float32).tocsr()
    if normalization != "row_l1":
        raise ValueError(f"Unsupported normalization: {normalization!r}")

    row_abs = np.asarray(abs(W).sum(axis=1)).ravel()
    row_scale = np.zeros_like(row_abs, dtype=np.float32)
    nonzero = row_abs > 0
    row_scale[nonzero] = float(global_gain) / row_abs[nonzero]
    return sp.diags(row_scale, format="csr").dot(W).astype(np.float32).tocsr()


def load_adult(
    adult_dir: str | Path,
    sensory_selector: Mapping[str, object],
    output_selector: Mapping[str, object],
    min_syn_count: int = 1,
    normalization: str = "row_l1",
    global_gain: float = 1.0,
    chunksize: int = 250_000,
    polarity: Mapping[str, object] | None = None,
) -> ConnectomeData:
    """Load the full adult connectome as coalesced SIO-ordered edge lists."""
    adult_dir = Path(adult_dir)
    polarity = polarity or {}
    nt_signs = {
        str(k).strip().upper(): float(v)
        for k, v in polarity.get("nt_signs", _DEFAULT_NT_SIGNS).items()
    }
    unknown_action = str(polarity.get("unknown_action", "zero")).lower()

    classification = pd.read_csv(_adult_file(adult_dir, "classification"))
    connections_path = _adult_file(adult_dir, "connections")
    all_ids = _collect_adult_connection_ids(connections_path, chunksize)

    sensory_ids = [
        root_id for root_id in _select_adult_ids(classification, sensory_selector)
        if root_id in all_ids
    ]
    sensory_set = set(sensory_ids)
    output_set = {
        root_id for root_id in _select_adult_ids(classification, output_selector)
        if root_id in all_ids and root_id not in sensory_set
    }
    internal_ids = sorted(
        root_id for root_id in all_ids
        if root_id not in sensory_set and root_id not in output_set
    )
    output_ids = sorted(output_set)
    ordered_ids = sensory_ids + internal_ids + output_ids
    id_to_index = {int(root_id): i for i, root_id in enumerate(ordered_ids)}
    nt_by_neuron = _load_adult_nt_types(adult_dir)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    total_rows = kept_rows = dropped_unknown = 0
    for chunk in pd.read_csv(
        connections_path,
        usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["syn_count"] >= min_syn_count].copy()
        pre_idx = chunk["pre_root_id"].map(id_to_index)
        post_idx = chunk["post_root_id"].map(id_to_index)
        valid = pre_idx.notna() & post_idx.notna()
        if not valid.any():
            continue

        chunk = chunk.loc[valid]
        pre_idx = pre_idx.loc[valid].astype(np.int64).to_numpy()
        post_idx = post_idx.loc[valid].astype(np.int64).to_numpy()
        nts = chunk["pre_root_id"].map(nt_by_neuron)
        nts = nts.fillna(chunk["nt_type"]).fillna("").astype(str).str.upper()
        signs = nts.map(nt_signs).to_numpy(dtype=np.float32, na_value=np.nan)
        known = ~np.isnan(signs)
        dropped_unknown += int((~known).sum())
        if unknown_action == "zero":
            signs[~known] = 0.0
            known = np.ones_like(known, dtype=bool)
        elif unknown_action == "excitatory":
            signs[~known] = 1.0
            known = np.ones_like(known, dtype=bool)
        elif unknown_action != "drop":
            raise ValueError(f"Unsupported unknown_action: {unknown_action!r}")
        if not known.any():
            continue

        weights = chunk["syn_count"].astype(np.float32).to_numpy()[known] * signs[known]
        nonzero = weights != 0
        if not nonzero.any():
            continue
        rows.append(pre_idx[known][nonzero])
        cols.append(post_idx[known][nonzero])
        data.append(weights[nonzero])
        kept_rows += int(nonzero.sum())

    shape = (len(ordered_ids), len(ordered_ids))
    if rows:
        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        data_arr = np.concatenate(data).astype(np.float32)
    else:
        row_arr = np.array([], dtype=np.int64)
        col_arr = np.array([], dtype=np.int64)
        data_arr = np.array([], dtype=np.float32)
    W = sp.coo_matrix((data_arr, (row_arr, col_arr)),
                      shape=shape, dtype=np.float32).tocsr()
    W.sum_duplicates()
    W = _normalize_sparse_matrix(W, normalization, global_gain)
    W.eliminate_zeros()
    coo = W.tocoo()

    metadata = {
        "species": "adult",
        "storage": "sparse_edges",
        "adult_dir": str(adult_dir),
        "normalization": normalization,
        "global_gain": global_gain,
        "min_syn_count": min_syn_count,
        "num_neurons": int(shape[0]),
        "num_sensory": len(sensory_ids),
        "num_internal": len(internal_ids),
        "num_output": len(output_ids),
        "connection_rows_total": total_rows,
        "connection_rows_kept": kept_rows,
        "connection_rows_dropped_unknown_nt": dropped_unknown,
        "matrix_nnz": int(W.nnz),
    }
    print(
        "Adult connectome loaded: "
        f"N={shape[0]} S/I/O={len(sensory_ids)}/"
        f"{len(internal_ids)}/{len(output_ids)} nnz={W.nnz} "
        f"normalization={normalization} gain={global_gain}",
        flush=True,
    )
    return ConnectomeData(
        W=None,
        sensory_dim=len(sensory_ids),
        internal_dim=len(internal_ids),
        output_dim=len(output_ids),
        sensory_ids=tuple(sensory_ids),
        internal_ids=tuple(internal_ids),
        output_ids=tuple(output_ids),
        edge_pre=coo.row.astype(np.int64),
        edge_post=coo.col.astype(np.int64),
        edge_weight=coo.data.astype(np.float32),
        metadata=metadata,
    )


# ----------------------------------------------------------------------
# Randomization controls
# ----------------------------------------------------------------------


def shuffled_weights(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Keep sparsity mask, shuffle non-zero weights uniformly across edges."""
    if data.W is None:
        raise ValueError("shuffled_weights currently supports dense connectomes only")
    rng = np.random.default_rng(seed)
    W = data.W.copy()
    mask = W != 0
    vals = W[mask]
    rng.shuffle(vals)
    W[mask] = vals
    return _replace(data, W=W)


def degree_preserved_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Per-row in-degree preserved; column targets randomized.

    Picks for each row k = nnz(row) destinations uniformly from the valid
    column set and assigns the original row's nonzero values (shuffled
    among themselves) to those targets.
    """
    if data.W is None:
        raise ValueError("degree_preserved_random currently supports dense connectomes only")
    rng = np.random.default_rng(seed)
    N = data.N
    W = np.zeros_like(data.W)
    for i in range(N):
        nz = np.flatnonzero(data.W[i])
        if nz.size == 0:
            continue
        targets = rng.choice(N, size=nz.size, replace=False)
        vals = data.W[i, nz].copy()
        rng.shuffle(vals)
        W[i, targets] = vals
    return _replace(data, W=W)


def erdos_renyi(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Edge mask uniformly i.i.d. with same density; weights ~ N(0, std)."""
    if data.W is None:
        raise ValueError("erdos_renyi currently supports dense connectomes only")
    rng = np.random.default_rng(seed)
    mask_orig = data.W != 0
    p = mask_orig.mean()
    new_mask = rng.random(data.W.shape) < p
    std = float(np.std(data.W[mask_orig])) if mask_orig.any() else 1.0
    W = (rng.standard_normal(data.W.shape).astype(np.float32) * std) * new_mask
    return _replace(data, W=W.astype(np.float32))


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


RANDOMIZERS = {
    "shuffled_weights": shuffled_weights,
    "degree_preserved_random": degree_preserved_random,
    "erdos_renyi": erdos_renyi,
}


def randomize(data: ConnectomeData, kind: str, seed: int = 0) -> ConnectomeData:
    if kind not in RANDOMIZERS:
        raise ValueError(f"Unknown randomization: {kind!r}; "
                         f"valid: {sorted(RANDOMIZERS)}")
    return RANDOMIZERS[kind](data, seed=seed)

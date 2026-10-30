"""Historical FlyWire loader used for the archived Figure 6 graph.

Unknown neuron transmitter values fall back to the connection-table transmitter.
The newer training-source loader is deliberately not used for graph rebuilding.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
import scipy.sparse as sp


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


_ADULT_FILES = {
    "classification": "classification.csv",
    "connections": "connections_princeton.csv",
    "neurons": "neurons.csv",
}


_DEFAULT_NT_SIGNS = {
    "ACH": 1.0,
    "ACETYLCHOLINE": 1.0,
    "GLUT": 1.0,
    "GLUTAMATE": 1.0,
    "GABA": -1.0,
    "HIST": -1.0,
    "HISTAMINE": -1.0,
    "DA": 1.0,
    "DOPAMINE": 1.0,
    "SER": 1.0,
    "SEROTONIN": 1.0,
    "OCT": 1.0,
    "OCTOPAMINE": 1.0,
    "TYR": 1.0,
    "TYRAMINE": 1.0,
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

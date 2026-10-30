"""Connectome loading and randomization controls.

Loads connectomes in SIO order (sensory / internal / output partition)
and provides randomization controls required by the Q3/Q5 ablations:

  shuffled_weights         keep mask, shuffle non-zero weights
  degree_preserved_random  directed degree-preserving edge swaps
  matched_random           partition/size/spectrum-matched sparse graph
  erdos_renyi              i.i.d. random binary mask matched in density
  size_density_random      exact N/E random graph with matched rate scaling
  strict_block_degree_rewire
                           directed switch null preserving local statistics

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


_DATA_DIR = Path(__import__('os').environ['FLYCONTROL_RUN']) / 'inputs/larva'


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
_BANC_FILES = {
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


def _banc_file(banc_dir: str | Path, key: str) -> Path:
    base = Path(banc_dir) / _BANC_FILES[key]
    if base.exists():
        return base
    gz_path = Path(str(base) + ".gz")
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Missing {_BANC_FILES[key]} in {banc_dir}")


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


def _load_adult_nt_types(
    adult_dir: str | Path,
    min_score: float | None = None,
) -> dict[int, str]:
    usecols = ["root_id", "nt_type"]
    if min_score is not None:
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_nt_score must be in [0, 1]")
        usecols.append("nt_type_score")
    neurons = pd.read_csv(_adult_file(adult_dir, "neurons"), usecols=usecols)
    if min_score is not None:
        # Retain low-confidence IDs with an explicit empty transmitter.  This
        # prevents the per-connection nt_type column from silently bypassing
        # the declared confidence threshold through the fallback below.
        score = pd.to_numeric(neurons["nt_type_score"], errors="coerce")
        neurons.loc[~score.ge(float(min_score)), "nt_type"] = ""
    rows = neurons.dropna(subset=["root_id"])
    if min_score is not None:
        return {
            int(row.root_id): (
                ""
                if pd.isna(row.nt_type)
                else str(row.nt_type).strip().upper()
            )
            for row in rows.itertuples(index=False)
        }
    return {
        int(row.root_id): str(row.nt_type).strip().upper()
        for row in rows.itertuples(index=False)
        if not pd.isna(row.nt_type) and str(row.nt_type).strip()
    }


def _normalize_banc_columns(neurons: pd.DataFrame) -> pd.DataFrame:
    columns = {
        col: col.strip().lower().replace(" ", "_").replace("/", "_")
        for col in neurons.columns
    }
    return neurons.rename(columns=columns).rename(columns={"root_id": "root_id"})


def _banc_nt_sign(raw: object, nt_signs: Mapping[str, float]) -> float:
    if raw is None or pd.isna(raw):
        return np.nan
    text = str(raw).strip()
    if not text:
        return np.nan
    tokens = [
        part.strip().upper()
        for part in text.replace(";", ",").split(",")
        if part.strip()
    ]
    signs = [nt_signs[token] for token in tokens if token in nt_signs]
    if not signs:
        return np.nan
    has_negative = any(sign < 0 for sign in signs)
    has_positive = any(sign > 0 for sign in signs)
    if has_negative and has_positive:
        # Multiple incompatible transmitter labels do not define a unique
        # presynaptic sign.  Leave them for the declared unknown_action rather
        # than silently resolving the ambiguity in the excitatory direction.
        return 1.0
    if has_negative:
        return -1.0
    if has_positive:
        return 1.0
    # A registered zero is a deliberate conservative hypothesis (for example
    # glutamate/monoamines under ACh+/GABA-/others-zero), not an unknown label.
    return 0.0


def _load_banc_metadata(banc_dir: str | Path) -> pd.DataFrame:
    neurons = pd.read_csv(_banc_file(banc_dir, "neurons"))
    neurons = _normalize_banc_columns(neurons)
    if "root_id" not in neurons.columns:
        raise ValueError("BANC neurons file must contain 'Root ID'")
    return neurons


def _load_banc_nt_signs(
    banc_dir: str | Path,
    nt_signs: Mapping[str, float],
) -> dict[int, float]:
    neurons = _load_banc_metadata(banc_dir)
    verified = neurons.get("verified_nt_type", pd.Series("", index=neurons.index))
    predicted = neurons.get("predicted_nt_type", pd.Series("", index=neurons.index))
    raw_nt = verified.where(
        verified.fillna("").astype(str).str.strip() != "",
        predicted,
    )
    signs = raw_nt.map(lambda value: _banc_nt_sign(value, nt_signs))
    return {
        int(root_id): float(sign)
        for root_id, sign in zip(neurons["root_id"], signs)
        if not np.isnan(sign)
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
    if normalization == "post_l1":
        # W[pre, post] stores a directed presynaptic -> postsynaptic edge.
        # Normalize each postsynaptic neuron's total absolute incoming drive.
        # This is deliberately distinct from row_l1, which normalizes the
        # outgoing budget of every presynaptic neuron.
        post_abs = np.asarray(abs(W).sum(axis=0)).ravel()
        post_scale = np.zeros_like(post_abs, dtype=np.float32)
        nonzero = post_abs > 0
        post_scale[nonzero] = float(global_gain) / post_abs[nonzero]
        return W.dot(sp.diags(post_scale, format="csr")).astype(
            np.float32).tocsr()
    if normalization != "row_l1":
        raise ValueError(f"Unsupported normalization: {normalization!r}")

    row_abs = np.asarray(abs(W).sum(axis=1)).ravel()
    row_scale = np.zeros_like(row_abs, dtype=np.float32)
    nonzero = row_abs > 0
    row_scale[nonzero] = float(global_gain) / row_abs[nonzero]
    return sp.diags(row_scale, format="csr").dot(W).astype(np.float32).tocsr()


def _coalesce_edge_arrays(
    row_arr: np.ndarray,
    col_arr: np.ndarray,
    data_arr: np.ndarray,
    shape: tuple[int, int],
    normalization: str,
    global_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if row_arr.size == 0:
        return row_arr, col_arr, data_arr.astype(np.float32)

    order = np.lexsort((col_arr, row_arr))
    row_arr = row_arr[order]
    col_arr = col_arr[order]
    data_arr = data_arr[order].astype(np.float32)
    changed = (row_arr[1:] != row_arr[:-1]) | (col_arr[1:] != col_arr[:-1])
    starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero(changed).astype(np.int64) + 1,
    ))
    row_arr = row_arr[starts]
    col_arr = col_arr[starts]
    data_arr = np.add.reduceat(data_arr, starts).astype(np.float32)
    nonzero = data_arr != 0.0
    row_arr = row_arr[nonzero]
    col_arr = col_arr[nonzero]
    data_arr = data_arr[nonzero]

    if normalization == "none":
        data_arr = (data_arr * float(global_gain)).astype(np.float32)
    elif normalization == "global_max_abs":
        max_abs = float(np.max(np.abs(data_arr))) if data_arr.size else 0.0
        scale = float(global_gain) / max_abs if max_abs else 1.0
        data_arr = (data_arr * scale).astype(np.float32)
    elif normalization == "row_l1":
        row_abs = np.bincount(
            row_arr, weights=np.abs(data_arr), minlength=shape[0])
        row_scale = np.zeros(shape[0], dtype=np.float32)
        nonzero_rows = row_abs > 0
        row_scale[nonzero_rows] = float(global_gain) / row_abs[nonzero_rows]
        data_arr = (data_arr * row_scale[row_arr]).astype(np.float32)
    elif normalization == "post_l1":
        post_abs = np.bincount(
            col_arr, weights=np.abs(data_arr), minlength=shape[1])
        post_scale = np.zeros(shape[1], dtype=np.float32)
        nonzero_posts = post_abs > 0
        post_scale[nonzero_posts] = float(global_gain) / post_abs[nonzero_posts]
        data_arr = (data_arr * post_scale[col_arr]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization: {normalization!r}")

    nonzero = data_arr != 0.0
    return (
        row_arr[nonzero].astype(np.int64),
        col_arr[nonzero].astype(np.int64),
        data_arr[nonzero].astype(np.float32),
    )


def normalize_connectome(
    data: ConnectomeData,
    normalization: str,
    global_gain: float,
) -> ConnectomeData:
    """Normalize a fully constructed graph and record the operation order.

    Structured topology nulls must be generated from raw synapse counts and
    normalized only after rewiring.  Otherwise a postsynaptic normalization
    is silently invalidated when normalized weights move to new targets.
    """
    metadata = dict(data.metadata or {})
    metadata.update({
        "normalization": str(normalization),
        "global_gain": float(global_gain),
        "normalization_stage": "after_graph_construction",
    })
    if data.is_sparse:
        pre, post, weight = _sparse_arrays(data)
        normalized_pre, normalized_post, normalized_weight = (
            _coalesce_edge_arrays(
                np.asarray(pre, dtype=np.int64),
                np.asarray(post, dtype=np.int64),
                np.asarray(weight, dtype=np.float32),
                (data.N, data.N),
                str(normalization),
                float(global_gain),
            )
        )
        metadata["matrix_nnz"] = int(normalized_weight.size)
        return _replace(
            data,
            edge_pre=normalized_pre,
            edge_post=normalized_post,
            edge_weight=normalized_weight,
            metadata=metadata,
        )
    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    normalized = _normalize_sparse_matrix(
        sp.csr_matrix(data.W), str(normalization), float(global_gain)
    ).toarray().astype(np.float32)
    metadata["matrix_nnz"] = int(np.count_nonzero(normalized))
    return _replace(data, W=normalized, metadata=metadata)


def induced_subgraph(
    data: ConnectomeData,
    *,
    sensory_ids: set[int] | list[int] | tuple[int, ...],
    internal_ids: set[int] | list[int] | tuple[int, ...],
    output_ids: set[int] | list[int] | tuple[int, ...],
    label: str,
) -> ConnectomeData:
    """Return an S/I/O-preserving induced subgraph with remapped edge indices."""
    requested = [set(map(int, ids)) for ids in (sensory_ids, internal_ids, output_ids)]
    if requested[0] & requested[1] or requested[0] & requested[2] or requested[1] & requested[2]:
        raise ValueError("induced-subgraph S/I/O root-ID sets must be disjoint")
    current = [
        tuple(map(int, data.sensory_ids)),
        tuple(map(int, data.internal_ids)),
        tuple(map(int, data.output_ids)),
    ]
    selected = [
        tuple(root_id for root_id in ids if root_id in keep)
        for ids, keep in zip(current, requested)
    ]
    for role, ids in zip(("sensory", "internal", "output"), selected):
        if not ids:
            raise ValueError(f"induced subgraph has no {role} nodes")
    unknown = set.union(*requested).difference(set().union(*map(set, current)))
    if unknown:
        raise ValueError(
            f"induced subgraph requested {len(unknown)} unknown root IDs"
        )

    old_order = current[0] + current[1] + current[2]
    new_order = selected[0] + selected[1] + selected[2]
    old_index = {root_id: index for index, root_id in enumerate(old_order)}
    keep_old = np.asarray([old_index[root_id] for root_id in new_order], dtype=np.int64)
    remap = np.full(data.N, -1, dtype=np.int64)
    remap[keep_old] = np.arange(len(keep_old), dtype=np.int64)
    metadata = dict(data.metadata or {})
    metadata.update({
        "subgraph": str(label),
        "source_num_neurons": int(data.N),
        "source_num_edges": int(
            data.edge_weight.size if data.is_sparse else np.count_nonzero(data.W)
        ),
        "subgraph_node_fraction": float(len(keep_old) / data.N),
    })
    common = {
        "sensory_dim": len(selected[0]),
        "internal_dim": len(selected[1]),
        "output_dim": len(selected[2]),
        "sensory_ids": selected[0],
        "internal_ids": selected[1],
        "output_ids": selected[2],
        "metadata": metadata,
    }
    if data.is_sparse:
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        mask = (remap[data.edge_pre] >= 0) & (remap[data.edge_post] >= 0)
        pre = remap[data.edge_pre[mask]]
        post = remap[data.edge_post[mask]]
        weight = data.edge_weight[mask].copy()
        metadata["subgraph_num_edges"] = int(weight.size)
        metadata["subgraph_edge_fraction"] = float(
            weight.size / max(1, data.edge_weight.size)
        )
        return ConnectomeData(
            W=None,
            edge_pre=pre,
            edge_post=post,
            edge_weight=weight,
            **common,
        )
    assert data.W is not None
    matrix = data.W[np.ix_(keep_old, keep_old)].copy()
    metadata["subgraph_num_edges"] = int(np.count_nonzero(matrix))
    metadata["subgraph_edge_fraction"] = float(
        np.count_nonzero(matrix) / max(1, np.count_nonzero(data.W))
    )
    return ConnectomeData(W=matrix, **common)


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
    polarity_mode = str(polarity.get("mode", "nt_derived")).lower()
    if polarity_mode not in {"nt_derived", "unsigned"}:
        raise ValueError("polarity mode must be 'nt_derived' or 'unsigned'")
    nt_signs = {
        str(k).strip().upper(): float(v)
        for k, v in polarity.get("nt_signs", _DEFAULT_NT_SIGNS).items()
    }
    unknown_action = str(polarity.get("unknown_action", "zero")).lower()
    min_nt_score_raw = polarity.get("min_nt_score")
    min_nt_score = (
        None if min_nt_score_raw is None else float(min_nt_score_raw)
    )

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
    nt_by_neuron = (
        _load_adult_nt_types(adult_dir, min_score=min_nt_score)
        if polarity_mode == "nt_derived"
        else {}
    )

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
        if polarity_mode == "unsigned":
            signs = np.ones(len(chunk), dtype=np.float32)
        else:
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
        "polarity_mode": polarity_mode,
        "min_nt_score": min_nt_score,
        "unknown_nt_action": unknown_action,
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


def load_banc(
    banc_dir: str | Path,
    sensory_selector: Mapping[str, object],
    output_selector: Mapping[str, object],
    min_syn_count: int = 1,
    normalization: str = "row_l1",
    global_gain: float = 1.0,
    chunksize: int = 250_000,
    polarity: Mapping[str, object] | None = None,
) -> ConnectomeData:
    """Load BANC as sparse SIO edge lists without constructing dense W."""
    banc_dir = Path(banc_dir)
    polarity = polarity or {}
    polarity_mode = str(polarity.get("mode", "nt_derived")).lower()
    if polarity_mode not in {"nt_derived", "unsigned"}:
        raise ValueError("polarity mode must be 'nt_derived' or 'unsigned'")
    nt_signs = {
        str(k).strip().upper(): float(v)
        for k, v in polarity.get("nt_signs", _DEFAULT_NT_SIGNS).items()
    }
    unknown_action = str(polarity.get("unknown_action", "zero")).lower()

    classification = _load_banc_metadata(banc_dir)
    connections_path = _banc_file(banc_dir, "connections")
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
    sign_by_neuron = (
        _load_banc_nt_signs(banc_dir, nt_signs)
        if polarity_mode == "nt_derived"
        else {}
    )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    total_rows = kept_rows = dropped_unknown = 0
    for chunk in pd.read_csv(
        connections_path,
        usecols=["pre_root_id", "post_root_id", "syn_count"],
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
        if polarity_mode == "unsigned":
            signs = np.ones(len(chunk), dtype=np.float32)
        else:
            signs = chunk["pre_root_id"].map(sign_by_neuron).to_numpy(
                dtype=np.float32, na_value=np.nan)
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
    row_arr, col_arr, data_arr = _coalesce_edge_arrays(
        row_arr, col_arr, data_arr, shape, normalization, global_gain)

    metadata = {
        "species": "banc",
        "storage": "sparse_edges",
        "banc_dir": str(banc_dir),
        "normalization": normalization,
        "polarity_mode": polarity_mode,
        "unknown_nt_action": unknown_action,
        "global_gain": global_gain,
        "min_syn_count": min_syn_count,
        "num_neurons": int(shape[0]),
        "num_sensory": len(sensory_ids),
        "num_internal": len(internal_ids),
        "num_output": len(output_ids),
        "connection_rows_total": total_rows,
        "connection_rows_kept": kept_rows,
        "connection_rows_dropped_unknown_nt": dropped_unknown,
        "matrix_nnz": int(data_arr.size),
    }
    print(
        "BANC connectome loaded: "
        f"N={shape[0]} S/I/O={len(sensory_ids)}/"
        f"{len(internal_ids)}/{len(output_ids)} nnz={data_arr.size} "
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
        edge_pre=row_arr,
        edge_post=col_arr,
        edge_weight=data_arr,
        metadata=metadata,
    )


# ----------------------------------------------------------------------
# Randomization controls
# ----------------------------------------------------------------------


def shuffled_weights(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Keep sparsity mask, shuffle non-zero weights uniformly across edges."""
    rng = np.random.default_rng(seed)
    if data.is_sparse:
        if data.edge_weight is None:
            raise ValueError("sparse connectome requires edge_weight")
        weights = data.edge_weight.copy()
        rng.shuffle(weights)
        return _replace(
            data,
            edge_weight=weights.astype(np.float32),
            metadata=_randomized_metadata(data, "shuffled_weights", seed),
        )
    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    W = data.W.copy()
    mask = W != 0
    vals = W[mask]
    rng.shuffle(vals)
    W[mask] = vals
    return _replace(
        data,
        W=W,
        metadata=_randomized_metadata(data, "shuffled_weights", seed),
    )


def transform_edge_polarity(
    data: ConnectomeData, mode: str,
) -> ConnectomeData:
    """Transform weights without changing the evaluated sparse support.

    ``absolute_on_support`` is intended for a from-scratch polarity control:
    the exact or already-rewired graph keeps identical edge endpoints and
    magnitudes, while every retained non-zero edge becomes positive. Edges
    excluded by the upstream transmitter prior remain excluded.
    """
    mode = str(mode).lower()
    if mode != "absolute_on_support":
        raise ValueError(
            "edge polarity transform must be 'absolute_on_support'"
        )
    _require_sparse_edges(data)
    assert data.edge_weight is not None
    metadata = dict(data.metadata or {})
    metadata.update({
        "post_randomization_polarity_transform": mode,
        "polarity_transform_preserves_edge_endpoints": True,
        "polarity_transform_preserves_absolute_weights": True,
    })
    return _replace(
        data,
        edge_weight=np.abs(data.edge_weight).astype(np.float32, copy=False),
        metadata=metadata,
    )


def shuffle_presynaptic_polarity(
    data: ConnectomeData, seed: int = 0
) -> ConnectomeData:
    """Shuffle Dale-like neuron signs while preserving topology and magnitudes.

    This is the polarity-placement null used by EXP08.  It preserves every
    edge, every absolute edge weight, and the number of positive/negative
    presynaptic neurons.  Unlike ``shuffled_weights``, it changes only which
    active presynaptic identities carry each sign.  Mixed-sign outgoing rows
    are rejected because they do not define a neuron-level sign permutation.
    """
    rng = np.random.default_rng(seed)
    if data.is_sparse:
        pre, _post, weights = _sparse_arrays(data)
        active = np.unique(pre)
        signs = np.empty(len(active), dtype=np.float32)
        for index, neuron in enumerate(active):
            outgoing = np.sign(weights[pre == neuron])
            unique = np.unique(outgoing[outgoing != 0])
            if len(unique) != 1:
                raise ValueError(
                    "presynaptic polarity shuffle requires Dale-like rows"
                )
            signs[index] = unique[0]
        shuffled = signs.copy()
        rng.shuffle(shuffled)
        sign_by_neuron = np.ones(data.N, dtype=np.float32)
        sign_by_neuron[active] = shuffled
        new_weights = np.abs(weights) * sign_by_neuron[pre]
        metadata = _randomized_metadata(
            data,
            "presynaptic_polarity_shuffle",
            seed,
            active_presynaptic_neurons=int(len(active)),
            positive_presynaptic_neurons=int(np.sum(signs > 0)),
            negative_presynaptic_neurons=int(np.sum(signs < 0)),
            preserves_topology=True,
            preserves_absolute_edge_weights=True,
        )
        return _replace(
            data,
            edge_weight=new_weights.astype(np.float32),
            metadata=metadata,
        )

    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    W = data.W.copy()
    active = np.flatnonzero(np.any(W != 0, axis=1))
    signs = np.empty(len(active), dtype=np.float32)
    for index, neuron in enumerate(active):
        outgoing = np.sign(W[neuron, W[neuron] != 0])
        unique = np.unique(outgoing)
        if len(unique) != 1:
            raise ValueError("presynaptic polarity shuffle requires Dale-like rows")
        signs[index] = unique[0]
    shuffled = signs.copy()
    rng.shuffle(shuffled)
    W[active] = np.abs(W[active]) * shuffled[:, None]
    return _replace(
        data,
        W=W,
        metadata=_randomized_metadata(
            data,
            "presynaptic_polarity_shuffle",
            seed,
            active_presynaptic_neurons=int(len(active)),
            positive_presynaptic_neurons=int(np.sum(signs > 0)),
            negative_presynaptic_neurons=int(np.sum(signs < 0)),
            preserves_topology=True,
            preserves_absolute_edge_weights=True,
        ),
    )


def _choice_unique_targets(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Sample k unique integer targets without paying O(n) for sparse rows."""
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return rng.choice(n, size=k, replace=False).astype(np.int64)
    if k * 8 < n:
        values: set[int] = set()
        while len(values) < k:
            need = k - len(values)
            draw = rng.integers(0, n, size=max(need * 2, 8), dtype=np.int64)
            values.update(int(x) for x in draw)
        return np.fromiter(list(values)[:k], dtype=np.int64, count=k)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def degree_preserved_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Randomize topology while preserving every directed in/out-degree.

    Sparse graphs use directed double-edge swaps:
    ``(a, b), (c, d) -> (a, d), (c, b)``.  The edge list and an O(E)
    hash set are the only topology storage; no N x N matrix is created.
    Existing dense larva behavior is retained for compatibility.
    """
    rng = np.random.default_rng(seed)
    if data.is_sparse:
        pre, post, weights = _sparse_arrays(data)
        edge_count = len(pre)
        if edge_count < 2:
            return _replace(
                data,
                edge_pre=pre.copy(),
                edge_post=post.copy(),
                edge_weight=weights.copy(),
                metadata=_randomized_metadata(
                    data, "degree_preserved_random", seed, successful_swaps=0
                ),
            )

        new_pre = pre.copy()
        new_post = post.copy()
        new_weights = weights.copy()
        keys = set((new_pre * data.N + new_post).tolist())
        if len(keys) != edge_count:
            raise ValueError("sparse edge list contains parallel edges")

        successful = 0
        attempts = 0
        target_swaps = edge_count
        max_attempts = 20 * target_swaps
        while successful < target_swaps and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, edge_count, size=2)
            if i == j:
                continue
            a, b = int(new_pre[i]), int(new_post[i])
            c, d = int(new_pre[j]), int(new_post[j])
            if a == c or b == d:
                continue
            key_i = a * data.N + b
            key_j = c * data.N + d
            swap_i = a * data.N + d
            swap_j = c * data.N + b
            if swap_i in keys or swap_j in keys:
                continue
            keys.remove(key_i)
            keys.remove(key_j)
            keys.add(swap_i)
            keys.add(swap_j)
            new_post[i], new_post[j] = d, b
            successful += 1

        return _replace(
            data,
            edge_pre=new_pre,
            edge_post=new_post,
            edge_weight=new_weights,
            metadata=_randomized_metadata(
                data,
                "degree_preserved_random",
                seed,
                successful_swaps=successful,
                attempted_swaps=attempts,
            ),
        )
    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    N = data.N
    if data.W is None:
        _require_sparse_edges(data)
        rows = []
        cols = []
        weights = []
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        order = np.argsort(data.edge_pre, kind="stable")
        sorted_pre = data.edge_pre[order]
        sorted_weight = data.edge_weight[order]
        unique_pre, starts, counts = np.unique(
            sorted_pre, return_index=True, return_counts=True)
        for i, start, count in zip(unique_pre, starts, counts):
            targets = _choice_unique_targets(rng, N, int(count))
            vals = sorted_weight[start:start + count].copy()
            rng.shuffle(vals)
            rows.append(np.full(int(count), int(i), dtype=np.int64))
            cols.append(targets.astype(np.int64))
            weights.append(vals.astype(np.float32))
        edge_pre, edge_post, edge_weight = _concat_sparse_edges(rows, cols, weights)
        return _replace(
            data,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_weight=edge_weight,
            metadata=_randomized_metadata(data, "degree_preserved_random", seed),
        )
    W = np.zeros_like(data.W)
    for i in range(N):
        nz = np.flatnonzero(data.W[i])
        if nz.size == 0:
            continue
        targets = rng.choice(N, size=nz.size, replace=False)
        vals = data.W[i, nz].copy()
        rng.shuffle(vals)
        W[i, targets] = vals
    return _replace(
        data,
        W=W,
        metadata=_randomized_metadata(data, "degree_preserved_random", seed),
    )


def matched_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Build a sparse random graph matched by S/I/O blocks and spectrum.

    Node count, total edge count, each of the nine directed partition
    edge counts, and signs are exact.  Original weights are shuffled onto
    the new mask, then multiplied by one global factor so the estimated
    spectral radius matches the source graph.  Global scaling preserves
    signs and the shape of the weight distribution.
    """
    if not data.is_sparse:
        raise ValueError("matched_random requires a sparse edge-list connectome")

    from .graph_diagnostics import estimate_spectral_radius

    pre, post, weights = _sparse_arrays(data)
    rng = np.random.default_rng(seed)
    bounds = np.array(
        [0, data.sensory_dim, data.sensory_dim + data.internal_dim, data.N],
        dtype=np.int64,
    )
    pre_part = np.searchsorted(bounds[1:], pre, side="right")
    post_part = np.searchsorted(bounds[1:], post, side="right")
    block_counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(block_counts, (pre_part, post_part), 1)

    random_pre: list[np.ndarray] = []
    random_post: list[np.ndarray] = []
    for source_part in range(3):
        source_start, source_stop = bounds[source_part : source_part + 2]
        source_size = int(source_stop - source_start)
        for target_part in range(3):
            target_start, target_stop = bounds[target_part : target_part + 2]
            target_size = int(target_stop - target_start)
            count = int(block_counts[source_part, target_part])
            capacity = source_size * target_size
            if count > capacity:
                raise ValueError("partition edge count exceeds simple-graph capacity")
            if count == 0:
                random_pre.append(np.empty(0, dtype=np.int64))
                random_post.append(np.empty(0, dtype=np.int64))
                continue
            selected = _sample_unique_indices(rng, capacity, count)
            random_pre.append(source_start + selected // target_size)
            random_post.append(target_start + selected % target_size)

    new_pre = np.concatenate(random_pre).astype(np.int64, copy=False)
    new_post = np.concatenate(random_post).astype(np.int64, copy=False)
    new_weights = weights.copy()
    rng.shuffle(new_weights)

    source_radius = estimate_spectral_radius(data)
    candidate = _replace(
        data,
        edge_pre=new_pre,
        edge_post=new_post,
        edge_weight=new_weights,
    )
    random_radius = estimate_spectral_radius(candidate)
    if source_radius > 0.0 and random_radius <= np.finfo(float).eps:
        raise ValueError("matched random graph has zero spectral radius")
    scale = source_radius / random_radius if random_radius > 0.0 else 1.0
    new_weights = (new_weights.astype(np.float64) * scale).astype(
        weights.dtype, copy=False
    )
    matched = _replace(
        data,
        edge_pre=new_pre,
        edge_post=new_post,
        edge_weight=new_weights,
        metadata=_randomized_metadata(
            data,
            "matched_random",
            seed,
            spectral_radius_source=source_radius,
            spectral_radius_before_scale=random_radius,
            spectral_scale=scale,
        ),
    )
    return matched


def erdos_renyi(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Edge mask uniformly i.i.d. with same density; weights ~ N(0, std)."""
    rng = np.random.default_rng(seed)
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_weight is not None
        nnz = int(data.edge_weight.size)
        edge_pre, edge_post = _sample_unique_pairs(rng, data.N, data.N, nnz)
        std = float(np.std(data.edge_weight)) if nnz else 1.0
        edge_weight = (
            rng.standard_normal(nnz).astype(np.float32) * std
        ).astype(np.float32)
        return _replace(
            data,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_weight=edge_weight,
            metadata=_randomized_metadata(data, "erdos_renyi", seed),
        )
    mask_orig = data.W != 0
    p = mask_orig.mean()
    new_mask = rng.random(data.W.shape) < p
    std = float(np.std(data.W[mask_orig])) if mask_orig.any() else 1.0
    W = (rng.standard_normal(data.W.shape).astype(np.float32) * std) * new_mask
    return _replace(
        data,
        W=W.astype(np.float32),
        metadata=_randomized_metadata(data, "erdos_renyi", seed),
    )


def _presynaptic_signs(data: ConnectomeData) -> np.ndarray:
    """Infer a Dale-like sign for every presynaptic neuron.

    Adult and BANC edge signs are assigned from presynaptic transmitter
    identity before row-L1 normalization.  A few rows have no retained
    outgoing edge; those rows receive the majority sign so an ER draw that
    happens to use them still has a well-defined rate-network sign.
    """
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_pre is not None
        assert data.edge_weight is not None
        row_sum = np.bincount(
            data.edge_pre,
            weights=data.edge_weight.astype(np.float64),
            minlength=data.N,
        )
    else:
        row_sum = data.W.astype(np.float64).sum(axis=1)
    signs = np.sign(row_sum).astype(np.float32)
    nonzero = signs != 0
    majority = 1.0
    if nonzero.any() and float(signs[nonzero].sum()) < 0:
        majority = -1.0
    signs[~nonzero] = majority
    return signs


def _row_l1_weights(
    edge_pre: np.ndarray,
    row_sign: np.ndarray,
    n_rows: int,
    global_gain: float,
) -> np.ndarray:
    """Return equal-magnitude, Dale-like weights with exact row-L1 gain."""
    degree = np.bincount(edge_pre, minlength=n_rows)
    denom = degree[edge_pre].astype(np.float32)
    return (
        row_sign[edge_pre]
        * np.float32(global_gain)
        / np.maximum(denom, np.float32(1.0))
    ).astype(np.float32)


def size_density_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Exact size-and-density random rate graph.

    This is the deliberately weak ER-SD null used in the NMI baseline ladder.
    It preserves N, E and the S/I/O node partition.  The same S and O indices
    therefore remain the policy input/output gateways.  Recurrent edges are
    sampled uniformly without duplicates, and weights are Dale-like with the
    same row-L1/global-gain operating scale as the biological graph.

    It does *not* preserve node degrees or S/I/O block counts; those are the
    purpose of :func:`strict_block_degree_rewire`.
    """
    rng = np.random.default_rng(seed)
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_weight is not None
        edge_count = int(data.edge_weight.size)
    else:
        edge_count = int(np.count_nonzero(data.W))
    edge_pre, edge_post = _sample_unique_pairs(
        rng, data.N, data.N, edge_count)
    row_sign = _presynaptic_signs(data)
    gain = float((data.metadata or {}).get("global_gain", 1.0))
    edge_weight = _row_l1_weights(
        edge_pre, row_sign, data.N, global_gain=gain)
    metadata = _randomized_metadata(data, "size_density_random", seed)
    metadata.update({
        "null_family": "ER-SD",
        "preserves_exact_N": True,
        "preserves_exact_E": True,
        "preserves_sio_node_partition": True,
        "preserves_node_degrees": False,
        "weight_scaling": "presynaptic_sign_equal_weight_row_l1",
    })
    if data.W is None:
        return _replace(
            data,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_weight=edge_weight,
            metadata=metadata,
        )
    W = np.zeros_like(data.W)
    W[edge_pre, edge_post] = edge_weight
    return _replace(data, W=W, metadata=metadata)


def _node_blocks(data: ConnectomeData) -> np.ndarray:
    block = np.empty(data.N, dtype=np.int8)
    s_end = data.sensory_dim
    i_end = s_end + data.internal_dim
    block[:s_end] = 0
    block[s_end:i_end] = 1
    block[i_end:] = 2
    return block


def _strict_switch_edges(
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    node_block: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator,
    n_rounds: int,
) -> tuple[np.ndarray, int, int]:
    """Vectorized directed double-edge swaps within S/I/O blocks.

    In every round, eligible edges in the same pre-block/post-block are
    randomly paired and their targets are exchanged.  A pair is accepted only
    when both proposed edges are non-self, absent from the current graph, and
    unique among all proposals in that round.  Existing self loops are frozen,
    so their count and placement are preserved exactly.

    Swapping only targets preserves each node's in/out-degree.  Keeping the
    weight attached to its source edge additionally preserves every row's
    signed and absolute weight multiset (and therefore row strength).
    """
    pre = np.asarray(edge_pre, dtype=np.int64)
    post = np.asarray(edge_post, dtype=np.int64).copy()
    edge_count = int(pre.size)
    successful = 0
    attempted = 0

    for _ in range(int(n_rounds)):
        current_keys = np.sort(pre * int(n_nodes) + post)
        pair_a_parts: list[np.ndarray] = []
        pair_b_parts: list[np.ndarray] = []
        pre_block = node_block[pre]
        post_block = node_block[post]
        eligible = pre != post
        for source_block in range(3):
            for target_block in range(3):
                idx = np.flatnonzero(
                    eligible
                    & (pre_block == source_block)
                    & (post_block == target_block)
                )
                if idx.size < 2:
                    continue
                rng.shuffle(idx)
                usable = idx.size - (idx.size % 2)
                pair_a_parts.append(idx[:usable:2])
                pair_b_parts.append(idx[1:usable:2])
        if not pair_a_parts:
            break
        pair_a = np.concatenate(pair_a_parts)
        pair_b = np.concatenate(pair_b_parts)
        attempted += int(pair_a.size)

        pre_a, pre_b = pre[pair_a], pre[pair_b]
        post_a, post_b = post[pair_a], post[pair_b]
        key_a = pre_a * int(n_nodes) + post_b
        key_b = pre_b * int(n_nodes) + post_a
        valid = (
            (pre_a != pre_b)
            & (post_a != post_b)
            & (pre_a != post_b)
            & (pre_b != post_a)
        )

        proposed = np.concatenate([key_a, key_b])
        positions = np.searchsorted(current_keys, proposed)
        exists = np.zeros(proposed.size, dtype=bool)
        inside = positions < current_keys.size
        exists[inside] = current_keys[positions[inside]] == proposed[inside]
        valid &= ~exists[:pair_a.size] & ~exists[pair_a.size:]

        # Simultaneous pairs can propose the same new edge.  Reject every
        # member of such a collision rather than depending on pair order.
        _, inverse, counts = np.unique(
            proposed, return_inverse=True, return_counts=True)
        valid &= (
            (counts[inverse[:pair_a.size]] == 1)
            & (counts[inverse[pair_a.size:]] == 1)
        )
        accepted_a = pair_a[valid]
        accepted_b = pair_b[valid]
        post[accepted_a] = post_b[valid]
        post[accepted_b] = post_a[valid]
        successful += int(valid.sum())

    if edge_count:
        final_keys = pre * int(n_nodes) + post
        if np.unique(final_keys).size != edge_count:
            raise RuntimeError("strict rewiring produced duplicate edges")
    return post, successful, attempted


def strict_block_degree_rewire(
    data: ConnectomeData,
    seed: int = 0,
    n_rounds: int = 10,
) -> ConnectomeData:
    """High-order topology null with exact low-order graph invariants.

    Preserved exactly: N/E, S/I/O node identities and block edge counts,
    every node's in/out-degree, existing self loops, and each presynaptic
    row's signed/absolute weight multiset.  Only higher-order target identity
    is changed.  Ten vectorized switch rounds expose each eligible edge to
    roughly ten random pairing attempts while remaining practical for the
    3.7-million-edge Adult graph.
    """
    rng = np.random.default_rng(seed)
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        pre = data.edge_pre.astype(np.int64, copy=False)
        original_post = data.edge_post.astype(np.int64, copy=False)
        rewired_post, successful, attempted = _strict_switch_edges(
            pre,
            original_post,
            _node_blocks(data),
            data.N,
            rng,
            n_rounds,
        )
        changed_fraction = float(np.mean(rewired_post != original_post))
        metadata = _randomized_metadata(
            data, "strict_block_degree_rewire", seed)
        metadata.update({
            "null_family": "STRICT-REWIRE",
            "switch_rounds": int(n_rounds),
            "successful_switches": int(successful),
            "attempted_switches": int(attempted),
            "changed_edge_fraction": changed_fraction,
            "preserves_exact_in_degree": True,
            "preserves_exact_out_degree": True,
            "preserves_exact_sio_block_counts": True,
            "preserves_exact_row_weight_multiset": True,
            "preserves_existing_self_loops": True,
        })
        return _replace(
            data,
            edge_post=rewired_post,
            edge_weight=data.edge_weight.copy(),
            metadata=metadata,
        )

    pre, post = np.nonzero(data.W)
    weight = data.W[pre, post].copy()
    rewired_post, successful, attempted = _strict_switch_edges(
        pre.astype(np.int64),
        post.astype(np.int64),
        _node_blocks(data),
        data.N,
        rng,
        n_rounds,
    )
    W = np.zeros_like(data.W)
    W[pre, rewired_post] = weight
    metadata = _randomized_metadata(data, "strict_block_degree_rewire", seed)
    metadata.update({
        "null_family": "STRICT-REWIRE",
        "switch_rounds": int(n_rounds),
        "successful_switches": int(successful),
        "attempted_switches": int(attempted),
        "changed_edge_fraction": float(np.mean(rewired_post != post)),
        "preserves_exact_in_degree": True,
        "preserves_exact_out_degree": True,
        "preserves_exact_sio_block_counts": True,
        "preserves_exact_row_weight_multiset": True,
        "preserves_existing_self_loops": True,
    })
    return _replace(data, W=W, metadata=metadata)


def block_preserved_random(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Randomize edges while preserving S/I/O block edge counts."""
    rng = np.random.default_rng(seed)
    blocks = _sio_blocks(data)
    if data.W is None:
        _require_sparse_edges(data)
        assert data.edge_pre is not None
        assert data.edge_post is not None
        assert data.edge_weight is not None
        rows = []
        cols = []
        weights = []
        for pre_lo, pre_hi, post_lo, post_hi in blocks:
            mask = (
                (data.edge_pre >= pre_lo)
                & (data.edge_pre < pre_hi)
                & (data.edge_post >= post_lo)
                & (data.edge_post < post_hi)
            )
            count = int(mask.sum())
            if count == 0:
                continue
            pre, post = _sample_unique_pairs(
                rng, pre_hi - pre_lo, post_hi - post_lo, count)
            vals = data.edge_weight[mask].copy()
            rng.shuffle(vals)
            rows.append((pre + pre_lo).astype(np.int64))
            cols.append((post + post_lo).astype(np.int64))
            weights.append(vals.astype(np.float32))
        edge_pre, edge_post, edge_weight = _concat_sparse_edges(rows, cols, weights)
        return _replace(
            data,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_weight=edge_weight,
            metadata=_randomized_metadata(data, "block_preserved_random", seed),
        )

    W = np.zeros_like(data.W)
    for pre_lo, pre_hi, post_lo, post_hi in blocks:
        block = data.W[pre_lo:pre_hi, post_lo:post_hi]
        nz = np.argwhere(block != 0)
        count = int(nz.shape[0])
        if count == 0:
            continue
        pre, post = _sample_unique_pairs(
            rng, pre_hi - pre_lo, post_hi - post_lo, count)
        vals = block[block != 0].copy()
        rng.shuffle(vals)
        W[pre + pre_lo, post + post_lo] = vals
    return _replace(
        data,
        W=W.astype(np.float32),
        metadata=_randomized_metadata(data, "block_preserved_random", seed),
    )


def zero_edges(data: ConnectomeData, seed: int = 0) -> ConnectomeData:
    """Remove recurrent edges while preserving the exact S/I/O interface.

    This is an interface-matched graph-off control, not a topology null.  It is
    useful when the actor has a shared direct path and graph contribution must
    be tested without changing trainable parameter shapes.
    """
    metadata = _randomized_metadata(
        data,
        "zero_edges",
        seed,
        null_family="GRAPH-OFF",
        preserves_sio_node_identities=True,
        source_edge_count=int(
            data.edge_weight.size
            if data.is_sparse and data.edge_weight is not None
            else np.count_nonzero(data.W)
        ),
    )
    if data.is_sparse:
        return _replace(
            data,
            edge_pre=np.empty(0, dtype=np.int64),
            edge_post=np.empty(0, dtype=np.int64),
            edge_weight=np.empty(0, dtype=np.float32),
            metadata=metadata,
        )
    if data.W is None:
        raise ValueError("connectome has neither dense nor sparse weights")
    return _replace(data, W=np.zeros_like(data.W), metadata=metadata)


def _require_sparse_edges(data: ConnectomeData) -> None:
    if data.edge_pre is None or data.edge_post is None or data.edge_weight is None:
        raise ValueError("sparse connectome requires edge_pre/post/weight")


def _concat_sparse_edges(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )
    return (
        np.concatenate(rows).astype(np.int64),
        np.concatenate(cols).astype(np.int64),
        np.concatenate(weights).astype(np.float32),
    )


def _sample_unique_pairs(
    rng: np.random.Generator,
    n_pre: int,
    n_post: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    capacity = int(n_pre) * int(n_post)
    if count > capacity:
        raise ValueError("cannot sample more unique edges than block capacity")
    if count == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    keys = np.array([], dtype=np.int64)
    while keys.size < count:
        missing = count - keys.size
        batch = max(missing * 2, 16)
        pre = rng.integers(0, n_pre, size=batch, dtype=np.int64)
        post = rng.integers(0, n_post, size=batch, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, pre * int(n_post) + post]))
    # np.unique sorts keys. Taking its prefix systematically deletes high
    # source indices. A uniform subset preserves exchangeability of every
    # candidate edge; sorting is only a storage convention AFTER selection.
    if keys.size > count:
        keys = np.sort(rng.choice(keys, size=count, replace=False))
    return (
        (keys // int(n_post)).astype(np.int64),
        (keys % int(n_post)).astype(np.int64),
    )


def _sio_blocks(data: ConnectomeData) -> list[tuple[int, int, int, int]]:
    ranges = [
        (0, data.sensory_dim),
        (data.sensory_dim, data.sensory_dim + data.internal_dim),
        (data.sensory_dim + data.internal_dim, data.N),
    ]
    return [
        (pre_lo, pre_hi, post_lo, post_hi)
        for pre_lo, pre_hi in ranges
        for post_lo, post_hi in ranges
    ]


def _sparse_arrays(
    data: ConnectomeData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        data.edge_pre is None
        or data.edge_post is None
        or data.edge_weight is None
    ):
        raise ValueError("incomplete sparse edge list")
    if not (
        len(data.edge_pre) == len(data.edge_post) == len(data.edge_weight)
    ):
        raise ValueError("sparse edge arrays must have equal lengths")
    return data.edge_pre, data.edge_post, data.edge_weight


def _sample_unique_indices(
    rng: np.random.Generator, population: int, count: int
) -> np.ndarray:
    """Floyd sample without replacement in O(count) memory and time."""
    if count == 0:
        return np.empty(0, dtype=np.int64)
    selected: set[int] = set()
    for upper in range(population - count, population):
        candidate = int(rng.integers(0, upper + 1))
        selected.add(upper if candidate in selected else candidate)
    result = np.fromiter(sorted(selected), dtype=np.int64, count=count)
    rng.shuffle(result)
    return result


def _randomized_metadata(
    data: ConnectomeData, kind: str, seed: int, **details: object
) -> dict:
    metadata = dict(data.metadata or {})
    metadata.update(
        randomization_kind=kind,
        randomize_seed=int(seed),
        **details,
    )
    # Keep the historical field used by existing experiment manifests while
    # exposing the more explicit ``randomization_kind`` name to new analyses.
    metadata["kind"] = kind
    return metadata


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
    "matched_random": matched_random,
    "erdos_renyi": erdos_renyi,
    "size_density_random": size_density_random,
    "block_preserved_random": block_preserved_random,
    "strict_block_degree_rewire": strict_block_degree_rewire,
    "presynaptic_polarity_shuffle": shuffle_presynaptic_polarity,
    "zero_edges": zero_edges,
}


def randomize(data: ConnectomeData, kind: str, seed: int = 0) -> ConnectomeData:
    if kind not in RANDOMIZERS:
        raise ValueError(f"Unknown randomization: {kind!r}; "
                         f"valid: {sorted(RANDOMIZERS)}")
    return RANDOMIZERS[kind](data, seed=seed)

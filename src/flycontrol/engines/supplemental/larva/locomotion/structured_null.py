"""Anatomy-stratified connectome topology nulls.

The formal BANC null exchanges postsynaptic identities only among targets
with the same S/I/O role, soma side, and top input/output neuropil label.  It
preserves exact in/out degree, self loops, and the weight multiset attached to
each presynaptic row.  Pass raw synapse counts here, then normalize the exact
and rewired graphs separately with :func:`normalize_connectome`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .connectome import ConnectomeData, _replace, induced_subgraph


def _normalize_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        column: column.strip().lower().replace(" ", "_").replace("/", "_")
        for column in frame.columns
    }
    return frame.rename(columns=columns)


def load_node_metadata(path: str | Path) -> pd.DataFrame:
    """Load a BANC neuron metadata CSV/CSV.GZ with canonical column names."""
    frame = _normalize_metadata_columns(pd.read_csv(path))
    required = {"root_id", "soma_side", "top_in_out_region"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"node metadata lacks required columns: {missing}")
    return frame


def load_target_stratum_metadata(
    path: str | Path,
    target_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Load generic node metadata for an anatomy-stratified graph null."""
    frame = _normalize_metadata_columns(pd.read_csv(path))
    required = {"root_id", *target_fields}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"node metadata lacks required columns: {missing}")
    return frame


def banc_leg_vnc_induced_subgraph(
    data: ConnectomeData,
    metadata: pd.DataFrame,
) -> ConnectomeData:
    """Keep leg afferents, VNC-intrinsic neurons, and leg motor neurons only.

    Descending/ascending and central-brain neurons are deliberately excluded:
    this subgraph tests local VNC sensorimotor topology, not a compartment-free
    whole-CNS rate model or endogenous command generation.
    """
    metadata = _normalize_metadata_columns(metadata)
    required = {"root_id", "flow", "body_part", "super_class", "class"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"BANC leg-VNC subset metadata lacks columns: {missing}")
    root_id = metadata.root_id.astype(np.int64)
    flow = metadata.flow.fillna("").astype(str).str.strip().str.lower()
    body = metadata.body_part.fillna("").astype(str).str.strip().str.lower()
    super_class = (
        metadata.super_class.fillna("").astype(str).str.strip().str.lower()
    )
    class_name = metadata["class"].fillna("").astype(str).str.strip().str.lower()
    leg_parts = {"front_leg", "middle_leg", "hind_leg"}
    sensory_ids = set(root_id[(flow == "afferent") & body.isin(leg_parts)]).intersection(
        map(int, data.sensory_ids)
    )
    internal_ids = set(root_id[super_class == "ventral_nerve_cord_intrinsic"]).intersection(
        map(int, data.internal_ids)
    )
    output_ids = set(root_id[class_name == "leg_motor_neuron"]).intersection(
        map(int, data.output_ids)
    )
    return induced_subgraph(
        data,
        sensory_ids=sensory_ids,
        internal_ids=internal_ids,
        output_ids=output_ids,
        label="BANC-LEG-AFFERENT-VNC-INTRINSIC-LEG-MOTOR",
    )


def node_target_strata(
    metadata: pd.DataFrame,
    ordered_root_ids: tuple[int, ...],
    sensory_dim: int,
    internal_dim: int,
    *,
    extra_fields: tuple[str, ...] = (),
    base_fields: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign every graph node an anatomy-aware target stratum."""
    metadata = _normalize_metadata_columns(metadata)
    indexed = metadata.drop_duplicates("root_id").set_index("root_id")
    missing_root_ids = sorted(set(ordered_root_ids).difference(indexed.index))
    if missing_root_ids:
        raise ValueError(
            f"BANC metadata lacks {len(missing_root_ids)} graph nodes; "
            f"first missing root_id={missing_root_ids[0]}"
        )
    ordered = indexed.reindex(np.asarray(ordered_root_ids, dtype=np.int64))
    requested_fields = (
        ("soma_side", "top_in_out_region")
        if base_fields is None
        else tuple(base_fields)
    )
    all_fields = tuple(dict.fromkeys((*requested_fields, *extra_fields)))
    missing_fields = sorted(set(all_fields).difference(ordered.columns))
    if missing_fields:
        raise ValueError(f"target stratum fields absent from metadata: {missing_fields}")

    def normalized_field(field: str) -> np.ndarray:
        return (
            ordered[field].fillna("unannotated")
            .astype(str).str.strip().str.lower().replace({"": "unannotated"})
            .to_numpy()
        )

    role = np.full(len(ordered), "output", dtype=object)
    role[:sensory_dim] = "sensory"
    role[sensory_dim : sensory_dim + internal_dim] = "internal"
    labels = pd.DataFrame({"role": role})
    if base_fields is None:
        # Preserve the historical BANC inventory column names.
        labels["side"] = normalized_field("soma_side")
        labels["neuropil"] = normalized_field("top_in_out_region")
    else:
        for field in requested_fields:
            labels[field] = normalized_field(field)
    for field in extra_fields:
        if field not in requested_fields:
            labels[field] = normalized_field(field)
    tuples = pd.MultiIndex.from_frame(labels)
    codes, uniques = pd.factorize(tuples, sort=True)
    labels["target_stratum"] = codes.astype(np.int32)
    inventory_fields = [
        column for column in labels.columns if column != "target_stratum"
    ]
    inventory = (
        labels.groupby(
            ["target_stratum", *inventory_fields], as_index=False
        ).size().rename(columns={"size": "node_count"})
    )
    if len(uniques) != len(inventory):
        raise RuntimeError("stratum inventory mismatch")
    return codes.astype(np.int32), inventory


def rewire_targets_by_stratum(
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    target_stratum: np.ndarray,
    n_nodes: int,
    *,
    seed: int,
    n_rounds: int = 5,
    locked_edge_mask: np.ndarray | None = None,
    edge_swap_class: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Perform deterministic directed target swaps within target strata."""
    pre = np.asarray(edge_pre, dtype=np.int64)
    original_post = np.asarray(edge_post, dtype=np.int64)
    post = original_post.copy()
    original_out_degree = np.bincount(pre, minlength=n_nodes)
    strata = np.asarray(target_stratum, dtype=np.int32)
    swap_class = None if edge_swap_class is None else np.asarray(edge_swap_class, dtype=np.int64)
    if swap_class is not None and swap_class.shape != pre.shape:
        raise ValueError("edge_swap_class must match the edge arrays")
    if locked_edge_mask is None:
        locked = np.zeros(len(pre), dtype=bool)
    else:
        locked = np.asarray(locked_edge_mask, dtype=bool)
        if locked.shape != pre.shape:
            raise ValueError("locked_edge_mask must match the edge arrays")
    rng = np.random.default_rng(seed)
    successful = 0
    attempted = 0
    for _ in range(int(n_rounds)):
        edge_stratum = strata[post]
        eligible = np.flatnonzero(~locked)
        sort_keys = (rng.random(len(eligible)), edge_stratum[eligible]) if swap_class is None else (
            rng.random(len(eligible)), swap_class[eligible], edge_stratum[eligible])
        order = eligible[np.lexsort(sort_keys)]
        usable = len(order) - (len(order) % 2)
        pair_a = order[:usable:2]
        pair_b = order[1:usable:2]
        attempted += len(pair_a)
        pre_a, pre_b = pre[pair_a], pre[pair_b]
        post_a, post_b = post[pair_a], post[pair_b]
        valid = (
            (edge_stratum[pair_a] == edge_stratum[pair_b])
            & (pre_a != pre_b)
            & (post_a != post_b)
            & (pre_a != post_a)
            & (pre_b != post_b)
            & (pre_a != post_b)
            & (pre_b != post_a)
        )
        if swap_class is not None:
            valid &= swap_class[pair_a] == swap_class[pair_b]
        current_keys = np.sort(pre * int(n_nodes) + post)
        key_a = pre_a * int(n_nodes) + post_b
        key_b = pre_b * int(n_nodes) + post_a
        proposed = np.concatenate([key_a, key_b])
        positions = np.searchsorted(current_keys, proposed)
        exists = np.zeros(len(proposed), dtype=bool)
        inside = positions < len(current_keys)
        exists[inside] = current_keys[positions[inside]] == proposed[inside]
        valid &= ~exists[: len(pair_a)] & ~exists[len(pair_a) :]
        _, inverse, counts = np.unique(
            proposed, return_inverse=True, return_counts=True
        )
        valid &= (
            (counts[inverse[: len(pair_a)]] == 1)
            & (counts[inverse[len(pair_a) :]] == 1)
        )
        accepted_a = pair_a[valid]
        accepted_b = pair_b[valid]
        post[accepted_a] = post_b[valid]
        post[accepted_b] = post_a[valid]
        successful += int(valid.sum())

    final_keys = pre * int(n_nodes) + post
    if np.unique(final_keys).size != len(final_keys):
        raise RuntimeError("target-stratified rewire produced duplicate edges")
    diagnostics = {
        "seed": int(seed),
        "switch_rounds": int(n_rounds),
        "attempted_switch_pairs": int(attempted),
        "successful_switch_pairs": int(successful),
        "changed_edge_fraction": float(np.mean(post != original_post)),
        "locked_edge_count": int(locked.sum()),
        "edge_swap_class_count": None if swap_class is None else int(np.unique(swap_class).size),
        "locked_edges_unchanged": bool(
            np.array_equal(post[locked], original_post[locked])
        ),
        "preserves_exact_out_degree": bool(
            np.array_equal(original_out_degree, np.bincount(pre, minlength=n_nodes))
        ),
        "preserves_exact_in_degree": bool(
            np.array_equal(
                np.bincount(original_post, minlength=n_nodes),
                np.bincount(post, minlength=n_nodes),
            )
        ),
        "preserves_target_stratum_per_edge": bool(
            np.array_equal(strata[original_post], strata[post])
        ),
        "preserves_existing_self_loops": bool(
            np.array_equal(pre == original_post, pre == post)
        ),
        "no_duplicate_edges": True,
    }
    return post, diagnostics


def target_stratified_degree_rewire(
    data: ConnectomeData,
    metadata: pd.DataFrame,
    *,
    seed: int,
    n_rounds: int = 5,
    extra_target_fields: tuple[str, ...] = (),
    base_target_fields: tuple[str, ...] | None = None,
    null_family: str = "BANC-ANATOMY-STRATIFIED-REWIRE",
    preserve_incoming_weight_multiset: bool = False,
) -> ConnectomeData:
    """Build an anatomy-stratified structured null from an unnormalized graph."""
    if not data.is_sparse or data.edge_pre is None or data.edge_post is None:
        raise ValueError("target-stratified rewire requires sparse edge arrays")
    if data.edge_weight is None:
        raise ValueError("target-stratified rewire requires edge weights")
    ordered_ids = data.sensory_ids + data.internal_ids + data.output_ids
    strata, inventory = node_target_strata(
        metadata,
        ordered_ids,
        data.sensory_dim,
        data.internal_dim,
        extra_fields=extra_target_fields,
        base_fields=base_target_fields,
    )
    swap_class = None
    if preserve_incoming_weight_multiset:
        if not np.isfinite(data.edge_weight).all():
            raise ValueError("incoming-weight-preserving null requires finite raw weights")
        # Swap only edges with exactly equal signed raw synapse counts.
        # Every postsynaptic signed-weight multiset is then unchanged.
        _, swap_class = np.unique(data.edge_weight, return_inverse=True)
    rewired_post, diagnostics = rewire_targets_by_stratum(
        data.edge_pre,
        data.edge_post,
        strata,
        data.N,
        seed=seed,
        n_rounds=n_rounds,
        edge_swap_class=swap_class,
    )
    graph_metadata = dict(data.metadata or {})
    graph_metadata.update({
        "kind": "target_stratified_degree_rewire",
        "randomization_kind": "target_stratified_degree_rewire",
        "randomize_seed": int(seed),
        "null_family": str(null_family),
        "target_stratum_count": int(len(inventory)),
        "base_target_stratum_fields": list(
            ("soma_side", "top_in_out_region")
            if base_target_fields is None
            else base_target_fields
        ),
        "extra_target_stratum_fields": list(extra_target_fields),
        "randomization_stage": "before_normalization",
        "preserves_incoming_raw_signed_weight_multiset": bool(preserve_incoming_weight_multiset),
        **diagnostics,
    })
    return _replace(
        data,
        edge_post=rewired_post,
        edge_weight=data.edge_weight.copy(),
        metadata=graph_metadata,
    )

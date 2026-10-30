"""Build anatomical indices from pinned public annotations in graph node order."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from flycontrol.paths import write_json, sha256


def adult_nodes(root, graph_path=None):
    graph = dict(np.load(graph_path or root / 'inputs/graph.npz'))
    ids = graph['root_ids']
    n = len(ids)
    s, i, d = [int(graph[k]) for k in ('sensory_dim', 'internal_dim', 'output_dim')]
    raw = root / 'inputs/flywire783'
    nodes = pd.DataFrame({'root_id': ids, 'state_index': np.arange(n)})
    for name in ('classification', 'neurons', 'consolidated_cell_types'):
        table = pd.read_csv(raw / f'{name}.csv.gz')
        cols = ['root_id'] + [c for c in table if c not in nodes and c != 'root_id']
        nodes = nodes.merge(table[cols], on='root_id', how='left', validate='one_to_one', sort=False)
    nodes = nodes.fillna('unassigned')
    nodes['partition'] = np.where(nodes.state_index < s, 'sensory', np.where(nodes.state_index >= s+i, 'output', 'internal'))
    nodes['model_role'] = nodes.partition.replace({'sensory': 'input'})
    nodes['node_index'] = nodes.state_index
    nodes['refined_group'] = nodes['class']
    nodes.loc[(nodes['class'] == 'AN') & nodes.super_class.eq('ascending'), 'refined_group'] = 'true_AN'
    nodes.loc[(nodes['class'] == 'AN') & nodes.super_class.eq('sensory_ascending'), 'refined_group'] = 'SA'
    w = sparse.csr_matrix((graph['edge_weight'], (graph['edge_post'], graph['edge_pre'])), shape=(n, n))
    wd = w[-d:]
    nodes['out_degree'] = np.diff(w.tocsc().indptr)
    nodes['in_degree'] = np.diff(w.indptr)
    nodes['direct_output_degree'] = np.diff(wd.tocsc().indptr)
    for prefix, mat in (('', w), ('direct_output_', wd)):
        nodes[prefix + ('out_l1' if not prefix else 'l1')] = np.asarray(abs(mat).sum(0)).ravel()
        nodes[prefix + ('out_l2_squared' if not prefix else 'l2_squared')] = np.asarray(mat.multiply(mat).sum(0)).ravel()
    assert np.array_equal(nodes.root_id, ids)
    return nodes, graph


def group_inventory(nodes, graph, output):
    s, i, d = [int(graph[k]) for k in ('sensory_dim', 'internal_dim', 'output_dim')]
    rows, rr, cc = [], [], []
    def add(level, part, parent, label, ids):
        ids = np.asarray(ids, int)
        index = len(rows)
        rows.append(dict(group_id=f'g{index:04d}', level=level, partition=part, parent=parent,
                         label=label, n_cells=len(ids), denominator_cells={'input': s, 'internal': i, 'output': d}[part]))
        rr.extend([index] * len(ids)); cc.extend(ids.tolist())
    inputs, inside, outputs = nodes.iloc[:s], nodes.iloc[s:s+i], nodes.iloc[s+i:]
    for label, g in inputs.groupby('refined_group'): add('input_class', 'input', label, label, g.index)
    for field, level in [('sub_class', 'input_subclass'), ('primary_type', 'input_type')]:
        for (cl, label), g in inputs.groupby(['refined_group', field]): add(level, 'input', cl, label, g.index)
    for cl, g in inside.groupby('class'): add('internal_class', 'internal', cl, cl, g.index)
    for label in ['PFL2', 'PFL3', 'LAL010']:
        ids = inside.index[inside.primary_type.eq(label)]
        if len(ids): add('internal_named', 'internal', 'named', label, ids)
    for label, g in outputs.groupby('primary_type'): add('output_type', 'output', 'DN', label, g.index)
    for label, part, group in [('All afferent', 'input', inputs), ('All internal', 'internal', inside), ('All DN', 'output', outputs)]:
        add('partition', part, part, label, group.index)
    output.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / 'group_inventory.csv', index=False)
    membership = sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(len(rows), len(nodes)))
    sparse.save_npz(output / 'group_membership.npz', membership)
    for level, expected in [('input_class', s), ('input_subclass', s), ('input_type', s), ('internal_class', i), ('output_type', d)]:
        assert inventory.loc[inventory.level.eq(level), 'n_cells'].sum() == expected
    return inventory


def input_masks(nodes, graph, output):
    s = int(graph['sensory_dim'])
    features = ['out_degree', 'direct_output_degree', 'out_l1', 'out_l2_squared', 'direct_output_l1', 'direct_output_l2_squared']
    x = np.log1p(nodes.iloc[:s][features].to_numpy(float))
    x = (x-x.mean(0)) / np.maximum(x.std(0), 1e-6)
    ids = np.flatnonzero(nodes.iloc[:s].refined_group.eq('true_AN'))
    pool = np.flatnonzero(nodes.iloc[:s].refined_group.ne('true_AN'))
    _, neighbors = cKDTree(x[pool]).query(x[ids], k=128)
    selected, used = [], set()
    for js in neighbors:
        for j in js:
            if int(pool[j]) not in used:
                used.add(int(pool[j])); selected.append(int(pool[j])); break
        else:
            remain = np.array([j for j in pool if j not in used])
            j = int(remain[np.argmin(np.sum((x[remain]-x[ids[len(selected)]])**2, axis=1))])
            used.add(j); selected.append(j)
    masks = {str(k): np.asarray(v) for k, v in nodes.iloc[:s].groupby('refined_group').groups.items()}
    masks['matched_nonAN'] = np.array(selected)
    masks['AN_GNG'] = np.flatnonzero(nodes['class'].eq('AN') & nodes.sub_class.eq('AN_GNG'))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'masks.npz', **masks)
    nodes.to_csv(output / 'nodes.csv.gz', index=False)
    return masks


def prepare(root, spec):
    family = spec['family']
    if family not in ('terrain', 'turning', 'speed'): return
    nodes, graph = adult_nodes(root)
    number = {'terrain': 2, 'turning': 3, 'speed': 4}[family]
    output = root / f'data/figure{number}/anatomy'
    output.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(output / 'nodes.csv.gz', index=False)
    nodes.to_csv(output / 'current_graph_annotations.csv', index=False)
    if family in ('speed', 'turning'):
        group_inventory(nodes, graph, output)
    if family == 'turning':
        base = root / 'data/figure3/mechanisms'
        input_masks(nodes, graph, base / 'anatomy/source_data')
        group_inventory(nodes, graph, base / 'sensitivity/source_data')
        from flycontrol.experiments.fig3.anatomy import prepare_interventions
        prepare_interventions(root, nodes, graph)
    write_json(output / 'complete.json', dict(nodes=len(nodes), graph_sha256=sha256(root / 'inputs/graph.npz'),
        annotation_sources={p.name: sha256(p) for p in (root / 'inputs/flywire783').glob('*.csv.gz')}))

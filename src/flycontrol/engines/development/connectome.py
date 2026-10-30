import pandas as pd
import numpy as np

def load_drosophila_matrix(csv_path, signed=False):
    """
    Load and process a Drosophila connectivity matrix.
    """
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)

    # Normalize depending on whether it's signed or unsigned
    if signed:
        max_abs = np.max(np.abs(W))
        W_norm = W / max_abs if max_abs != 0 else W
    else:
        W_min, W_max = W.min(), W.max()
        W_norm = (W - W_min) / (W_max - W_min + 1e-8)

    return W_norm

def load_connectivity_data(connectivity_path, annotation_path, rescale_factor=4e-2, sensory_type='all'):
    """
    Load and process the connectivity matrix and neuron annotations, splitting neurons into
    Sensory, Internal, and Output groups, then return a dictionary of 9 connectivity sub-matrices
    (as NumPy arrays) plus the neuron ID lists.

    Parameters:
    -----------
    connectivity_path : str
        Path to the connectivity matrix CSV file
    annotation_path : str
        Path to the neuron annotation CSV file
    rescale_factor : float
        Factor to rescale the connectivity weights
    normalization : str
        Type of normalization to apply ('minmax', 'clip', or None)
    sensory_type : str
        Type of sensory neurons to use as input. Can be:
        - 'olfactory': olfactory sensory neurons
        - 'visual': visual sensory neurons
        - 'gut': gut sensory neurons
        - 'respiratory': respiratory sensory neurons
        - 'gustatory-external': external gustatory sensory neurons
        - 'all': all sensory neurons
    """

    df_annot = pd.read_csv(annotation_path)
    # output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
    output_types = {'DN-SEZ','RGN'}
    sensory_ids = []
    output_ids = []

    for _, row in df_annot.iterrows():
        cell_type = row['celltype']
        additional_annotations = row['additional_annotations']
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                try:
                    neuron_id = int(id_str)
                except ValueError:
                    continue

                # Classify neuron as sensory based on selected type
                if cell_type == 'sensory':
                    if sensory_type == 'all':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'visual' and additional_annotations == 'visual':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'olfactory' and additional_annotations == 'olfactory':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'gut' and additional_annotations == 'gut':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'respiratory' and additional_annotations == 'respiratory':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'gustatory-external' and additional_annotations == 'gustatory-external':
                        sensory_ids.append(neuron_id)

                # Classify output neurons
                elif cell_type in output_types:
                    output_ids.append(neuron_id)

    sensory_ids = sorted(set(sensory_ids))
    output_ids = sorted(set(output_ids))

    print(f"Annotation file: Found {len(sensory_ids)} {sensory_type} sensory neuron IDs")
    print(f"Annotation file: Found {len(output_ids)} output neuron IDs")

    df_conn = pd.read_csv(connectivity_path, index_col=0)

    # Apply normalization
    df_conn = df_conn * rescale_factor

    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")

    valid_sensory_ids = [nid for nid in sensory_ids if nid in all_neuron_ids]
    valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]

    # Define internal neurons as the rest ---
    valid_internal_ids = [
        nid for nid in all_neuron_ids
        if nid not in valid_sensory_ids and nid not in valid_output_ids
    ]

    print(f"After filtering, found {len(valid_sensory_ids)} {sensory_type} sensory neurons in matrix")
    print(f"After filtering, found {len(valid_output_ids)} output neurons in matrix")
    print(f"Remaining {len(valid_internal_ids)} neurons classified as internal")

    # Create the SIO-ordered adjacency matrix
    ordered_ids = valid_sensory_ids + valid_internal_ids + valid_output_ids
    adjacency = df_conn.loc[ordered_ids, ordered_ids].values  # shape: [N, N]

    # Calculate indices for each group in the SIO-ordered matrix
    num_sensory = len(valid_sensory_ids)
    num_internal = len(valid_internal_ids)
    num_output = len(valid_output_ids)

    W_ss = adjacency[:num_sensory, :num_sensory]
    W_sr = adjacency[:num_sensory, num_sensory:num_sensory+num_internal]
    W_so = adjacency[:num_sensory, num_sensory+num_internal:]

    W_rs = adjacency[num_sensory:num_sensory+num_internal, :num_sensory]
    W_rr = adjacency[num_sensory:num_sensory+num_internal, num_sensory:num_sensory+num_internal]
    W_ro = adjacency[num_sensory:num_sensory+num_internal, num_sensory+num_internal:]

    W_os = adjacency[num_sensory+num_internal:, :num_sensory]
    W_or = adjacency[num_sensory+num_internal:, num_sensory:num_sensory+num_internal]
    W_oo = adjacency[num_sensory+num_internal:, num_sensory+num_internal:]

    return {
        'W_original': df_conn.values,
        'W': adjacency,  # Now in SIO order
        'W_ss': W_ss,
        'W_sr': W_sr,
        'W_so': W_so,
        'W_rs': W_rs,
        'W_rr': W_rr,
        'W_ro': W_ro,
        'W_or': W_or,
        'W_os': W_os,
        'W_oo': W_oo,
        'sensory_ids': valid_sensory_ids,
        'internal_ids': valid_internal_ids,
        'output_ids': valid_output_ids,
        'sensory_type': sensory_type
    }

def count_connections(connectivity_path, source_ids, target_ids):
    """
    统计从源神经元到目标神经元的连接数。

    Parameters:
    -----------
    connectivity_path : str
        连接矩阵CSV文件路径
    source_ids : list
        源神经元ID列表（如PN的ID列表）
    target_ids : list
        目标神经元ID列表（如KC的ID列表）

    Returns:
    --------
    dict : 包含统计信息的字典
        - total_connections: 总连接数（非零连接数）
        - num_source_neurons: 源神经元数量
        - num_target_neurons: 目标神经元数量
        - connection_matrix: 连接矩阵（子矩阵）
    """
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    # 确保ID在矩阵中存在
    valid_source_ids = [nid for nid in source_ids if nid in df_conn.index]
    valid_target_ids = [nid for nid in target_ids if nid in df_conn.columns]

    if not valid_source_ids:
        print(f"警告: 没有找到有效的源神经元ID")
        return {
            'total_connections': 0,
            'num_source_neurons': 0,
            'num_target_neurons': 0,
            'connection_matrix': None
        }

    if not valid_target_ids:
        print(f"警告: 没有找到有效的目标神经元ID")
        return {
            'total_connections': 0,
            'num_source_neurons': len(valid_source_ids),
            'num_target_neurons': 0,
            'connection_matrix': None
        }

    # 提取子矩阵：行是源神经元，列是目标神经元
    submatrix = df_conn.loc[valid_source_ids, valid_target_ids]

    # 统计非零连接数
    total_connections = (submatrix != 0).sum().sum()

    print(f"源神经元数量: {len(valid_source_ids)}")
    print(f"目标神经元数量: {len(valid_target_ids)}")
    print(f"总连接数（非零）: {total_connections}")

    return {
        'total_connections': int(total_connections),
        'num_source_neurons': len(valid_source_ids),
        'num_target_neurons': len(valid_target_ids),
        'connection_matrix': submatrix.values
    }

def count_pn_to_kc_connections(connectivity_path, pn_ids=None, kc_ids=None, annotation_path=None):
    """
    统计PN到KC的连接数。

    Parameters:
    -----------
    connectivity_path : str
        连接矩阵CSV文件路径
    pn_ids : list, optional
        PN神经元ID列表。如果为None，将尝试从注释文件中识别
    kc_ids : list, optional
        KC神经元ID列表。如果为None，将尝试从注释文件中识别
    annotation_path : str, optional
        注释文件路径，用于识别PN和KC

    Returns:
    --------
    dict : 包含统计信息的字典
    """
    # 如果提供了PN和KC的ID列表，直接使用
    if pn_ids is not None and kc_ids is not None:
        return count_connections(connectivity_path, pn_ids, kc_ids)

    # 如果没有提供ID列表，尝试从注释文件中识别
    if annotation_path is None:
        raise ValueError("必须提供pn_ids和kc_ids，或者提供annotation_path以自动识别")

    df_annot = pd.read_csv(annotation_path)
    pn_ids_found = []
    kc_ids_found = []

    for _, row in df_annot.iterrows():
        cell_type = str(row.get('celltype', '')).lower()
        additional_annotations = str(row.get('additional_annotations', '')).lower()

        # 识别PN：通常是投射神经元，可能在celltype或additional_annotations中标注
        if 'pn' in cell_type or 'projection' in cell_type or 'pn' in additional_annotations:
            for col in ['left_id', 'right_id']:
                id_str = str(row.get(col, '')).lower()
                if id_str != "no pair" and id_str != "nan":
                    try:
                        neuron_id = int(id_str)
                        pn_ids_found.append(neuron_id)
                    except (ValueError, TypeError):
                        continue

        # 识别KC：Kenyon细胞
        if 'kc' in cell_type or 'kenyon' in cell_type or 'kc' in additional_annotations:
            for col in ['left_id', 'right_id']:
                id_str = str(row.get(col, '')).lower()
                if id_str != "no pair" and id_str != "nan":
                    try:
                        neuron_id = int(id_str)
                        kc_ids_found.append(neuron_id)
                    except (ValueError, TypeError):
                        continue

    pn_ids_found = sorted(set(pn_ids_found))
    kc_ids_found = sorted(set(kc_ids_found))

    if not pn_ids_found:
        print("警告: 未能从注释文件中识别PN神经元，请手动提供pn_ids")
    if not kc_ids_found:
        print("警告: 未能从注释文件中识别KC神经元，请手动提供kc_ids")

    return count_connections(connectivity_path, pn_ids_found, kc_ids_found)

def count_olfactory_pn_kc_pathway(connectivity_path, annotation_path=None,
                                   olfactory_ids=None, pn_ids=None, kc_ids=None):
    """
    统计从olfactory neuron到PN再到KC的两跳路径连接数。

    Parameters:
    -----------
    connectivity_path : str
        连接矩阵CSV文件路径
    annotation_path : str, optional
        注释文件路径，用于自动识别神经元类型
    olfactory_ids : list, optional
        Olfactory sensory neuron ID列表
    pn_ids : list, optional
        PN神经元ID列表
    kc_ids : list, optional
        KC神经元ID列表

    Returns:
    --------
    dict : 包含统计信息的字典
        - olfactory_to_pn: Olfactory到PN的连接统计
        - pn_to_kc: PN到KC的连接统计
        - two_hop_paths: 两跳路径统计（olfactory -> PN -> KC）
    """
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    # 如果没有提供ID列表，从注释文件中提取
    if annotation_path is not None:
        df_annot = pd.read_csv(annotation_path)

        if olfactory_ids is None:
            olfactory_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                additional_annotations = str(row.get('additional_annotations', '')).strip()
                if cell_type == 'sensory' and 'olfactory' in additional_annotations.lower():
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                olfactory_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

        if pn_ids is None:
            pn_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                if cell_type == 'PN' or cell_type == 'PN-somato':
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                pn_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

        if kc_ids is None:
            kc_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                if cell_type == 'KC':
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                kc_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

    olfactory_ids = sorted(set(olfactory_ids)) if olfactory_ids else []
    pn_ids = sorted(set(pn_ids)) if pn_ids else []
    kc_ids = sorted(set(kc_ids)) if kc_ids else []

    print("=" * 60)
    print("统计 Olfactory -> PN -> KC 路径")
    print("=" * 60)
    print(f"Olfactory神经元数量: {len(olfactory_ids)}")
    print(f"PN神经元数量: {len(pn_ids)}")
    print(f"KC神经元数量: {len(kc_ids)}")
    print()

    # 统计 Olfactory -> PN
    print("1. Olfactory -> PN 连接:")
    olfactory_to_pn = count_connections(connectivity_path, olfactory_ids, pn_ids)
    print()

    # 统计 PN -> KC
    print("2. PN -> KC 连接:")
    pn_to_kc = count_connections(connectivity_path, pn_ids, kc_ids)
    print()

    # 统计两跳路径：Olfactory -> PN -> KC
    print("3. 两跳路径统计 (Olfactory -> PN -> KC):")
    valid_olfactory = [nid for nid in olfactory_ids if nid in df_conn.index]
    valid_pn = [nid for nid in pn_ids if nid in df_conn.index and nid in df_conn.columns]
    valid_kc = [nid for nid in kc_ids if nid in df_conn.columns]

    if not valid_olfactory or not valid_pn or not valid_kc:
        print("警告: 缺少有效的神经元ID，无法统计两跳路径")
        two_hop_stats = {
            'olfactory_neurons_with_path': 0,
            'total_two_hop_paths': 0,
            'kc_neurons_reachable': 0
        }
    else:
        # 提取连接矩阵
        W_olfactory_pn = df_conn.loc[valid_olfactory, valid_pn]
        W_pn_kc = df_conn.loc[valid_pn, valid_kc]

        # 计算两跳路径：对于每个olfactory神经元，找到它连接的PN，然后找到这些PN连接的KC
        two_hop_paths = 0
        olfactory_with_path = 0
        kc_reachable = set()
        pn_in_pathway = set()

        for olf_idx in valid_olfactory:
            # 找到这个olfactory神经元连接的PN（非零连接）
            connected_pns = W_olfactory_pn.loc[olf_idx, :]
            connected_pn_ids = connected_pns[connected_pns != 0].index.tolist()

            if connected_pn_ids:
                olfactory_with_path += 1
                pn_in_pathway.update(connected_pn_ids)
                # 对于每个连接的PN，找到它连接的KC
                for pn_id in connected_pn_ids:
                    pn_kc_connections = W_pn_kc.loc[pn_id, :]
                    connected_kcs = pn_kc_connections[pn_kc_connections != 0].index.tolist()
                    two_hop_paths += len(connected_kcs)
                    kc_reachable.update(connected_kcs)

        two_hop_stats = {
            'olfactory_neurons_with_path': olfactory_with_path,
            'total_two_hop_paths': two_hop_paths,
            'kc_neurons_reachable': len(kc_reachable),
            'pn_neurons_in_pathway': len(pn_in_pathway)
        }

        print(f"  有路径的Olfactory神经元数量: {two_hop_stats['olfactory_neurons_with_path']}")
        print(f"  总两跳路径数: {two_hop_stats['total_two_hop_paths']}")
        print(f"  可到达的KC神经元数量: {two_hop_stats['kc_neurons_reachable']}")
        print(f"  参与路径的PN神经元数量: {two_hop_stats['pn_neurons_in_pathway']}")

    print("=" * 60)

    return {
        'olfactory_to_pn': olfactory_to_pn,
        'pn_to_kc': pn_to_kc,
        'two_hop_paths': two_hop_stats
    }

def analyze_dnvnc_connections(connectivity_path, annotation_path=None,
                              dnvnc_ids=None, prednvnc_ids=None, mbon_ids=None):
    """
    分析DN-VNC的连接情况，明确区分pre-DN-VNC和DN-VNC，检查MBON到DN-VNC的路径。

    Parameters:
    -----------
    connectivity_path : str
        连接矩阵CSV文件路径
    annotation_path : str, optional
        注释文件路径，用于自动识别神经元类型
    dnvnc_ids : list, optional
        DN-VNC神经元ID列表（不包括pre-DN-VNC）
    prednvnc_ids : list, optional
        pre-DN-VNC神经元ID列表
    mbon_ids : list, optional
        MBON神经元ID列表

    Returns:
    --------
    dict : 包含分析结果的字典
    """
    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    # 如果没有提供ID列表，从注释文件中提取
    if annotation_path is not None:
        df_annot = pd.read_csv(annotation_path)

        if dnvnc_ids is None:
            dnvnc_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                # 只提取DN-VNC，不包括pre-DN-VNC
                if cell_type == 'DN-VNC':
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                dnvnc_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

        if prednvnc_ids is None:
            prednvnc_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                # 单独提取pre-DN-VNC
                if cell_type == 'pre-DN-VNC':
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                prednvnc_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

        if mbon_ids is None:
            mbon_ids = []
            for _, row in df_annot.iterrows():
                cell_type = str(row.get('celltype', '')).strip()
                if cell_type == 'MBON':
                    for col in ['left_id', 'right_id']:
                        id_str = str(row.get(col, '')).strip()
                        if id_str != "no pair" and id_str != "nan" and id_str:
                            try:
                                neuron_id = int(id_str)
                                mbon_ids.append(neuron_id)
                            except (ValueError, TypeError):
                                continue

    dnvnc_ids = sorted(set(dnvnc_ids)) if dnvnc_ids else []
    prednvnc_ids = sorted(set(prednvnc_ids)) if prednvnc_ids else []
    mbon_ids = sorted(set(mbon_ids)) if mbon_ids else []

    print("=" * 80)
    print("DN-VNC 连接分析（区分 pre-DN-VNC 和 DN-VNC）")
    print("=" * 80)
    print(f"DN-VNC神经元数量: {len(dnvnc_ids)}")
    print(f"pre-DN-VNC神经元数量: {len(prednvnc_ids)}")
    print(f"MBON神经元数量: {len(mbon_ids)}")
    print()

    # 确保ID在矩阵中存在
    valid_dnvnc = [nid for nid in dnvnc_ids if nid in df_conn.columns]

    if not valid_dnvnc:
        print("警告: 没有找到有效的DN-VNC神经元ID")
        return {}

    print(f"有效的DN-VNC神经元数量: {len(valid_dnvnc)}")
    print()

    # 1. 找出所有连接到DN-VNC的神经元
    print("1. 连接到DN-VNC的所有神经元:")
    print("-" * 80)

    # 连接矩阵中，行是源神经元，列是目标神经元
    # 所以我们需要找到所有行（源神经元）连接到DN-VNC（列）
    all_neurons = df_conn.index.tolist()
    connections_to_dnvnc = df_conn.loc[:, valid_dnvnc]  # 所有神经元到DN-VNC的连接

    # 找出有连接的神经元（非零连接）
    neurons_connected_to_dnvnc = []
    connection_counts = {}

    for neuron_id in all_neurons:
        connections = connections_to_dnvnc.loc[neuron_id, :]
        non_zero_connections = connections[connections != 0]
        if len(non_zero_connections) > 0:
            neurons_connected_to_dnvnc.append(neuron_id)
            connection_counts[neuron_id] = len(non_zero_connections)

    print(f"连接到DN-VNC的神经元总数: {len(neurons_connected_to_dnvnc)}")
    print(f"总连接数: {sum(connection_counts.values())}")
    print()

    # 2. 检查MBON是否直接连接到DN-VNC（不包括pre-DN-VNC）
    print("2. MBON到DN-VNC的直接连接（不经过pre-DN-VNC）:")
    print("-" * 80)

    valid_mbon = [nid for nid in mbon_ids if nid in df_conn.index]
    valid_prednvnc = [nid for nid in prednvnc_ids if nid in df_conn.index and nid in df_conn.columns]

    if not valid_mbon:
        print("警告: 没有找到有效的MBON神经元ID")
        mbon_direct_connections = {}
        mbon_direct_count = 0
    else:
        mbon_to_dnvnc = df_conn.loc[valid_mbon, valid_dnvnc]
        mbon_direct_connections = {}
        mbon_direct_count = 0

        for mbon_id in valid_mbon:
            connections = mbon_to_dnvnc.loc[mbon_id, :]
            non_zero = connections[connections != 0]
            if len(non_zero) > 0:
                mbon_direct_connections[mbon_id] = len(non_zero)
                mbon_direct_count += len(non_zero)

        if mbon_direct_connections:
            print(f"✅ MBON可以直接连接到DN-VNC（不经过pre-DN-VNC）")
            print(f"  有直接连接的MBON数量: {len(mbon_direct_connections)}")
            print(f"  MBON到DN-VNC的直接连接总数: {mbon_direct_count}")
            print(f"  平均每个MBON连接到 {mbon_direct_count / len(mbon_direct_connections):.2f} 个DN-VNC")
        else:
            print("❌ MBON没有直接连接到DN-VNC（不经过pre-DN-VNC）")

    print()

    # 3. 检查MBON是否通过pre-DN-VNC连接到DN-VNC
    print("3. MBON通过pre-DN-VNC连接到DN-VNC的路径:")
    print("-" * 80)

    if not valid_mbon:
        print("无法分析：没有有效的MBON神经元")
        path_analysis = {}
    elif not valid_prednvnc:
        print("无法分析：没有有效的pre-DN-VNC神经元")
        path_analysis = {}
    else:
        # 检查MBON -> pre-DN-VNC -> DN-VNC的路径
        # 步骤1: MBON连接到哪些pre-DN-VNC
        mbon_to_prednvnc = df_conn.loc[valid_mbon, valid_prednvnc]
        mbon_prednvnc_connections = {}

        for mbon_id in valid_mbon:
            connections = mbon_to_prednvnc.loc[mbon_id, :]
            non_zero = connections[connections != 0]
            if len(non_zero) > 0:
                mbon_prednvnc_connections[mbon_id] = non_zero.index.tolist()

        # 步骤2: 检查这些pre-DN-VNC是否连接到DN-VNC
        prednvnc_to_dnvnc = df_conn.loc[valid_prednvnc, valid_dnvnc]

        mbon_via_prednvnc_paths = 0
        mbon_with_prednvnc_path = 0
        prednvnc_in_pathway = set()

        for mbon_id, connected_prednvnc_list in mbon_prednvnc_connections.items():
            has_path = False
            for prednvnc_id in connected_prednvnc_list:
                if prednvnc_id in valid_prednvnc:
                    connections = prednvnc_to_dnvnc.loc[prednvnc_id, :]
                    non_zero = connections[connections != 0]
                    if len(non_zero) > 0:
                        mbon_via_prednvnc_paths += len(non_zero)
                        prednvnc_in_pathway.add(prednvnc_id)
                        has_path = True
            if has_path:
                mbon_with_prednvnc_path += 1

        if mbon_via_prednvnc_paths > 0:
            print(f"✅ 找到MBON -> pre-DN-VNC -> DN-VNC的路径！")
            print(f"  通过pre-DN-VNC的路径总数: {mbon_via_prednvnc_paths}")
            print(f"  参与路径的pre-DN-VNC数量: {len(prednvnc_in_pathway)}")
            print(f"  可以通过pre-DN-VNC到达DN-VNC的MBON数量: {mbon_with_prednvnc_path}")
        else:
            print("❌ MBON无法通过pre-DN-VNC到达DN-VNC")

        # 综合路径分析
        if mbon_direct_connections:
            if mbon_via_prednvnc_paths > 0:
                print()
                print("📊 路径总结:")
                print(f"  - MBON可以直接到达DN-VNC: ✅ ({mbon_direct_count} 条直接连接)")
                print(f"  - MBON也可以通过pre-DN-VNC到达DN-VNC: ✅ ({mbon_via_prednvnc_paths} 条路径)")
                path_analysis = {
                    'direct_path_exists': True,
                    'direct_connections': mbon_direct_count,
                    'mbon_with_direct_path': len(mbon_direct_connections),
                    'via_prednvnc_path_exists': True,
                    'via_prednvnc_paths_count': mbon_via_prednvnc_paths,
                    'prednvnc_in_pathway_count': len(prednvnc_in_pathway),
                    'mbon_with_prednvnc_path': mbon_with_prednvnc_path
                }
            else:
                print()
                print("📊 路径总结:")
                print(f"  - MBON可以直接到达DN-VNC: ✅ ({mbon_direct_count} 条直接连接)")
                print(f"  - MBON通过pre-DN-VNC到达DN-VNC: ❌")
                path_analysis = {
                    'direct_path_exists': True,
                    'direct_connections': mbon_direct_count,
                    'mbon_with_direct_path': len(mbon_direct_connections),
                    'via_prednvnc_path_exists': False
                }
        else:
            if mbon_via_prednvnc_paths > 0:
                print()
                print("📊 路径总结:")
                print(f"  - MBON可以直接到达DN-VNC: ❌")
                print(f"  - MBON通过pre-DN-VNC到达DN-VNC: ✅ ({mbon_via_prednvnc_paths} 条路径)")
                path_analysis = {
                    'direct_path_exists': False,
                    'via_prednvnc_path_exists': True,
                    'via_prednvnc_paths_count': mbon_via_prednvnc_paths,
                    'prednvnc_in_pathway_count': len(prednvnc_in_pathway),
                    'mbon_with_prednvnc_path': mbon_with_prednvnc_path
                }
            else:
                print()
                print("📊 路径总结:")
                print(f"  - MBON可以直接到达DN-VNC: ❌")
                print(f"  - MBON通过pre-DN-VNC到达DN-VNC: ❌")
                print("   分析其他多跳路径...")

                # 分析其他两跳路径：MBON -> X -> DN-VNC（X不是pre-DN-VNC）
                mbon_connections = df_conn.loc[valid_mbon, :]
                mbon_targets = set()
                for mbon_id in valid_mbon:
                    targets = mbon_connections.loc[mbon_id, :]
                    mbon_targets.update(targets[targets != 0].index.tolist())

                # 排除pre-DN-VNC
                other_intermediates = [tid for tid in mbon_targets
                                      if tid in df_conn.index and tid not in valid_prednvnc]

                other_two_hop_paths = 0
                for target_id in other_intermediates:
                    target_to_dnvnc = df_conn.loc[target_id, valid_dnvnc]
                    non_zero = target_to_dnvnc[target_to_dnvnc != 0]
                    if len(non_zero) > 0:
                        other_two_hop_paths += len(non_zero)

                if other_two_hop_paths > 0:
                    print(f"  - MBON通过其他神经元到达DN-VNC: ✅ ({other_two_hop_paths} 条路径)")
                    path_analysis = {
                        'direct_path_exists': False,
                        'via_prednvnc_path_exists': False,
                        'other_two_hop_path_exists': True,
                        'other_two_hop_paths_count': other_two_hop_paths
                    }
                else:
                    print(f"  - MBON通过其他神经元到达DN-VNC: ❌")
                    path_analysis = {
                        'direct_path_exists': False,
                        'via_prednvnc_path_exists': False,
                        'other_two_hop_path_exists': False
                    }

    print()
    print("=" * 80)
    print("分析摘要:")
    print("=" * 80)
    print(f"连接到DN-VNC的神经元总数: {len(neurons_connected_to_dnvnc)}")
    print(f"DN-VNC神经元数量: {len(valid_dnvnc)}")
    print(f"pre-DN-VNC神经元数量: {len(valid_prednvnc)}")
    if valid_mbon:
        if path_analysis.get('direct_path_exists'):
            print(f"✅ MBON可以直接到达DN-VNC ({path_analysis.get('direct_connections', 0)} 个直接连接)")
        if path_analysis.get('via_prednvnc_path_exists'):
            print(f"✅ MBON可以通过pre-DN-VNC到达DN-VNC ({path_analysis.get('via_prednvnc_paths_count', 0)} 条路径)")
        if path_analysis.get('other_two_hop_path_exists'):
            print(f"✅ MBON可以通过其他神经元到达DN-VNC ({path_analysis.get('other_two_hop_paths_count', 0)} 条路径)")
        if not path_analysis.get('direct_path_exists') and not path_analysis.get('via_prednvnc_path_exists') and not path_analysis.get('other_two_hop_path_exists'):
            print(f"❌ MBON无法在2跳内到达DN-VNC")
    print("=" * 80)

    return {
        'dnvnc_count': len(valid_dnvnc),
        'neurons_connected_to_dnvnc': neurons_connected_to_dnvnc,
        'connection_counts': connection_counts,
        'mbon_direct_connections': mbon_direct_connections if valid_mbon else {},
        'mbon_direct_count': mbon_direct_count if valid_mbon else 0,
        'path_analysis': path_analysis if valid_mbon else {}
    }

def convert_unsigned_to_signed(sign_csv_path, adjacency_csv_path, out_csv_path):

    df = pd.read_csv(sign_csv_path)
    df["sign"] = df["sign"].apply(
        lambda x: "true" if str(x).strip().lower() == "true" else "false"
    )

    true_count = (df["sign"] == "true").sum()
    total_count = len(df)
    false_count = total_count - true_count
    true_pct = (true_count / total_count * 100) if total_count else 0
    false_pct = 100 - true_pct
    print(f"True sign count: {true_count} ({true_pct:.2f}%)")
    print(f"False sign count: {false_count} ({false_pct:.2f}%)")

    conn_matrix_df = pd.read_csv(adjacency_csv_path, index_col=0)
    conn_matrix_df.to_csv(out_csv_path)
    print(f"Saved signed connectivity matrix to: {out_csv_path}")

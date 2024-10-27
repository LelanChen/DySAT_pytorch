# -*- encoding:utf-8 -*-
from __future__ import print_function

import pickle

import numpy as np
import networkx as nx
import scipy.sparse as sp
from config import DefaultConfig

cfg = DefaultConfig
np.random.seed(123)


def load_graphs(dataset_str):
    """加载给定数据集中的图形快照"""
    # graphs = np.load("data/{}/{}".format(dataset_str, "graphs_remap.npz"), encoding='bytes', allow_pickle=True)['graph']
    # graphs = np.load("data/Enron_new/graphs.npz", encoding='latin1', allow_pickle=True)['graph']
    with open('D:/科研/小论文/HADGA-pytorch/data/DDoS/graphs_remap.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adj_matrices = list(map(lambda x: np.array(nx.adjacency_matrix(x).todense()), graphs))
    labels = list(map(lambda g: [data.get('label') for _, data in g.nodes(data=True)], graphs))
    classes = list(map(lambda g: [data.get('classes') for _, data in g.nodes(data=True)], graphs))
    return graphs, adj_matrices, labels, classes


def load_feats(dataset_str):
    """ 加载给定数据集名称的节点属性快照（未在实验中使用）"""
    features = np.load("data/{}/{}".format(dataset_str, "node_feat_remap.npz"), encoding='bytes', allow_pickle=True)['node_feat']
    # features = np.load("data/Enron_new/features.npz", allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features


def sparse_to_tuple(sparse_mx):
    """将 scipy 稀疏矩阵转换为元组表示(for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):    # 检测mx是否是COO(三元组(row, col, data))的稀疏矩阵格式
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()    # vstack()垂直（按照行顺序）的把数组给堆叠起来
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # 创建适当的索引 - coords 是索引对的 numpy 数组。
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # 给定list of list，将其转换为list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """行规范化特征矩阵并转换为元组表示"""
    rowsum = np.array(features.sum(1))     # 对特征求和
    r_inv = np.power(rowsum, -1).flatten()  # 求特征和的倒数
    r_inv[np.isinf(r_inv)] = 0.     # 将无穷值替换为0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    """基于 GCN 的邻接矩阵归一化（scipy 稀疏格式），输出采用元组格式"""
    # 邻接矩阵是否存在多边的情况？所以需要归一化
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())


def get_context_pairs(graphs, num_time_steps):
    """ 通过随机游走采样，为每个快照加载/生成上下文对。"""
    load_path = "data/{}/train_pairs_n2v_{}.pkl".format(FLAGS.dataset, str(num_time_steps - 2))
    try:
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(0, num_time_steps):
            context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print ("Saved pairs")

    return context_pairs_train


def get_evaluation_data(adjs, num_time_steps, dataset):
    """ 加载训练/评估/测试示例以评估链路预测性能"""
    eval_idx = num_time_steps - 2
    eval_path = "data/{}/eval_{}.npz".format(dataset, str(eval_idx))
    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
        next_adjs = adjs[eval_idx + 1]
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions.
    数据集划分
    对于链接预测任务（在所有链接上），下一时间步邻接矩阵中的所有链接被认为是正样本。
    Out: 链接预测的正负对列表 (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # 删除对角线元素
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # 限制新链接到现有节点的约束，仅包含现有节点的连接和新连接，不包含新加入节点的连接
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.随机生成负样本
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

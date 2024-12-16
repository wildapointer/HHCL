import random
from itertools import permutations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F



def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_hyperedge_node_similarity(x: Tensor, hyperedge_index: Tensor):
    """
    计算每个超边内节点与超边特征的相似度，并求平均。

    参数：
    - x: Tensor，节点特征，形状为 [num_nodes, feat_dim]
    - hyperedge_index: Tensor，超边索引，形状为 [2, num_relations]

    返回：
    - hyperedge_similarities: Tensor，每个超边的平均节点-超边相似度，形状为 [num_edges]
    - mean_similarity: float，所有超边的平均相似度
    """
    node_idx, edge_idx = hyperedge_index  # [num_relations]
    num_nodes = x.size(0)
    num_edges = edge_idx.max().item() + 1  # 假设超边索引从 0 开始
    device = x.device

    # Step 1: 计算超边特征，通过对超边内的节点特征求和
    # 将节点特征按照超边索引聚合
    hyperedge_features = torch.zeros((num_edges, x.size(1)), device=device)  # [num_edges, feat_dim]
    scatter_add(src=x[node_idx], index=edge_idx, dim=0, out=hyperedge_features)

    # 计算每个超边内的节点数量，用于求平均
    hyperedge_sizes = scatter_add(torch.ones_like(edge_idx, dtype=torch.float32, device=device),
                                  edge_idx, dim=0, dim_size=num_edges)  # [num_edges]

    # 取平均，得到超边特征
    hyperedge_features = hyperedge_features / hyperedge_sizes.unsqueeze(1)

    # Step 2: 计算节点与超边特征的相似度
    x_node = x[node_idx]  # [num_relations, feat_dim]
    e_edge = hyperedge_features[edge_idx]  # [num_relations, feat_dim]

    # 归一化
    x_node_norm = F.normalize(x_node, p=2, dim=1)
    e_edge_norm = F.normalize(e_edge, p=2, dim=1)

    # 计算余弦相似度
    similarities = torch.sum(x_node_norm * e_edge_norm, dim=1)  # [num_relations]

    # Step 3: 对每个超边内的相似度求平均
    hyperedge_similarities = scatter_mean(similarities, edge_idx, dim=0, dim_size=num_edges)  # [num_edges]

    # 计算所有超边的平均相似度
    mean_similarity = hyperedge_similarities.mean().item()

    return hyperedge_similarities, mean_similarity


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_incidence_randomly(
    hyperedge_index: torch.Tensor,
    p: float = 0.2,
    max_p: float = 1
) -> torch.Tensor:
    """
    随机遮盖连接关系，按照给定的丢弃概率 p 进行遮盖。
    
    参数:
        hyperedge_index (torch.Tensor): 超边节点关系，形状为 [2, num_pairs]。
        p (float): 基础丢弃概率。
        max_p (float): 丢弃概率的最大值。
    
    返回:
        torch.Tensor: 新的 hyperedge_index，已遮盖部分连接关系。
    """
    if hyperedge_index.numel() == 0:
        return hyperedge_index  # 如果没有连接，直接返回
    
    # 生成随机丢弃概率
    drop_prob = torch.clamp(torch.full((hyperedge_index.size(1),), p, device=hyperedge_index.device), max=max_p)
    
    # 生成遮盖掩码，1表示保留，0表示遮盖
    sel_mask = torch.bernoulli(1.0 - drop_prob).to(torch.bool)  # [num_pairs]
    
    # 生成新的 hyperedge_index
    new_hyperedge_index = hyperedge_index[:, sel_mask]
    
    return new_hyperedge_index


def drop_incidence_based_on_similarity_multiround(
    x: torch.Tensor,
    p: float,
    hyperedge_index: torch.Tensor
) -> torch.Tensor:
    """
    根据节点与超边之间的相似度随机遮盖连接关系，低相似度的连接更有可能被遮盖，确保遮盖比例接近 p。

    参数:
        x (torch.Tensor): 节点特征，形状为 [num_nodes, feature_dim]。
        p (float): 基础丢弃概率。
        hyperedge_index (torch.Tensor): 超边节点关系，形状为 [2, num_pairs]。

    返回:
        torch.Tensor: 新的 hyperedge_index，已遮盖部分连接关系。
    """
    if hyperedge_index.numel() == 0:
        return hyperedge_index  # 如果没有连接，直接返回

    num_edges = hyperedge_index[1].max().item() + 1

    # 获取连接的节点特征
    connected_node_features = x[hyperedge_index[0]]  # [num_pairs, feature_dim]

    # 计算每个超边的特征：连接节点特征的平均
    sum_x = scatter_add(connected_node_features, hyperedge_index[1], dim=0, dim_size=num_edges)  # [num_edges, feature_dim]
    counts = scatter_add(torch.ones_like(hyperedge_index[0], dtype=torch.float), hyperedge_index[1], dim=0, dim_size=num_edges).unsqueeze(1)  # [num_edges, 1]
    hyperedge_features = sum_x / (counts + 1e-8)  # [num_edges, feature_dim]

    # 获取每个连接的超边特征和节点特征
    hyperedge_feat = hyperedge_features[hyperedge_index[1]]  # [num_pairs, feature_dim]
    node_feat = x[hyperedge_index[0]]  # [num_pairs, feature_dim]

    # 计算节点与超边之间的余弦相似度
    similarities = F.cosine_similarity(node_feat, hyperedge_feat, dim=1)  # [num_pairs]

    # 计算丢弃概率权重并归一化
    weights = (similarities.max() - similarities) / (similarities.max() - similarities.min() + 1e-8)  # 线性归一化
    # weights = weights * p
    weights = torch.clamp(weights, max=0.9)  # 将权重限制在 0.9 以下

    # 初始化掩码并开始随机遮盖
    num_pairs = hyperedge_index.size(1)
    num_to_drop = int(num_pairs * p)  # 需要遮盖的目标数量
    sel_mask = torch.ones(num_pairs, dtype=torch.bool, device=hyperedge_index.device)

    # 随机遮盖的过程
    while sel_mask.sum() > num_pairs - num_to_drop:
        # 根据权重生成遮盖掩码
        temp_mask = torch.bernoulli(1 - weights).to(torch.bool)
        sel_mask = sel_mask & temp_mask  # 逐步更新掩码，遮盖选中的连接
    
    # 生成新的 hyperedge_index
    new_hyperedge_index = hyperedge_index[:, sel_mask]
    
    return new_hyperedge_index



def drop_incidence_based_on_similarity(
    x: torch.Tensor,
    p: float,
    hyperedge_index: torch.Tensor
) -> torch.Tensor:
    """
    根据节点与超边之间的相似度来遮盖连接关系，低相似度的连接更有可能被遮盖。

    参数:
        x (torch.Tensor): 节点特征，形状为 [num_nodes, feature_dim]。
        p (float): 基础丢弃概率。
        hyperedge_index (torch.Tensor): 超边节点关系，形状为 [2, num_pairs]。

    返回:
        torch.Tensor: 新的 hyperedge_index，已遮盖部分连接关系。
    """
    if hyperedge_index.numel() == 0:
        return hyperedge_index  # 如果没有连接，直接返回

    num_edges = hyperedge_index[1].max().item() + 1

    # 获取连接的节点特征
    connected_node_features = x[hyperedge_index[0]]  # [num_pairs, feature_dim]

    # 计算每个超边的特征：连接节点特征的平均
    sum_x = scatter_add(connected_node_features, hyperedge_index[1], dim=0, dim_size=num_edges)  # [num_edges, feature_dim]
    counts = scatter_add(torch.ones_like(hyperedge_index[0], dtype=torch.float), hyperedge_index[1], dim=0, dim_size=num_edges).unsqueeze(1)  # [num_edges, 1]
    hyperedge_features = sum_x / (counts + 1e-8)  # [num_edges, feature_dim]

    # 获取每个连接的超边特征和节点特征
    hyperedge_feat = hyperedge_features[hyperedge_index[1]]  # [num_pairs, feature_dim]
    node_feat = x[hyperedge_index[0]]  # [num_pairs, feature_dim]

    # 计算节点与超边之间的余弦相似度
    similarities = F.cosine_similarity(node_feat, hyperedge_feat, dim=1)  # [num_pairs]
    
    # 计算丢弃概率权重
    # weights = (similarities.max() - similarities) / (similarities.max() - similarities.mean() + 1e-8)
    weights = (similarities.max() - similarities) / (similarities.max() - similarities.min() + 1e-8)#綫性歸一化
    weights = weights * p
    weights = torch.clamp(weights, max=0.9)
    
    # 生成掩码，直接基于 weights
    sel_mask = torch.bernoulli(1. - weights).to(torch.bool)
    
    # 生成新的 hyperedge_index
    new_hyperedge_index = hyperedge_index[:, sel_mask]
    
    return new_hyperedge_index




def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

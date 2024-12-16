from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, softmax
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0, 
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False, 
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None


    def forward(
        self, x: Tensor, hyperedge_index: Tensor, 
        num_nodes: Optional[int] = None, num_edges: Optional[int] = None
    ):

    
        # #控制相似度調節力度的方法
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        # 首先计算初始的归一化因子
        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]
                
            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x_n2e = self.lin_n2e(x)
        # 首次从节点到超边的聚合，用于计算相似度
        e = self.propagate(hyperedge_index, x=x_n2e, norm=norm_n2e, size=(num_nodes, num_edges))

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        # 计算节点和超边之间的相似度
        x_node = x_n2e[node_idx]  # [num_relations, hid_dim]
        e_edge = e[edge_idx]      # [num_relations, hid_dim]

        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(x_node, e_edge)  # [num_relations]

        # 根据相似度调整 norm_n2e 和 norm_e2n

        # 调整 norm_n2e（节点到超边的聚合）
        # 对于每个超边，归一化连接的节点的相似度
        sim_n2e = similarities.clone()
        sim_n2e_exp = sim_n2e.exp()  # 可以使用 softmax，更稳定
        sim_n2e_norm = scatter_add(sim_n2e_exp, edge_idx, dim=0, dim_size=num_edges)[edge_idx]
        sim_norm_n2e = sim_n2e_exp / sim_n2e_norm

        # 调整 norm_e2n（超边到节点的聚合）
        # 对于每个节点，归一化连接的超边的相似度
        sim_e2n = similarities.clone()
        sim_e2n_exp = sim_e2n.exp()
        sim_e2n_norm = scatter_add(sim_e2n_exp, node_idx, dim=0, dim_size=num_nodes)[node_idx]
        sim_norm_e2n = sim_e2n_exp / sim_e2n_norm

        # 引入參差連接
        beta = 1.0  #相似度的占比
        sim_norm_n2e_res = (1 - beta) * norm_n2e + beta * sim_norm_n2e
        sim_norm_e2n_res = (1 - beta) * norm_e2n + beta * sim_norm_e2n

        # 再次从节点到超边的聚合，使用调整后的 norm_n2e
        e = self.propagate(hyperedge_index, x=x_n2e, norm=sim_norm_n2e_res, size=(num_nodes, num_edges))

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        # 从超边到节点的聚合，使用调整后的 norm_e2n
        x_e2n = self.lin_e2n(e)
        hyperedge_index_rev = hyperedge_index.flip([0])

        n = self.propagate(hyperedge_index_rev, x=x_e2n, norm=sim_norm_e2n_res, size=(num_edges, num_nodes))

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e  # No act, act


    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
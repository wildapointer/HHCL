from typing import Optional
import os.path as osp
import pickle

import torch
from torch_scatter import scatter_add
from torch.utils.data import random_split


class BaseDataset(object):
    def __init__(self, type: str, name: str, similarity: Optional[float] = None, device: str = 'cpu'):
        self.type = type
        self.name = name
        self.similarity = similarity  # 新增相似度参数
        self.device = device

        # 原始数据集路径
        if self.type in ['cocitation', 'coauthorship']:
            self.dataset_dir = osp.join('Semester_2\\TriCL_hater\\dataset', self.type, self.name) 
        else:
            self.dataset_dir = osp.join('Semester_2\\TriCL_hater\\dataset', self.name)

        # 相似度版本路径（仅用于 hypergraph）
        if self.similarity is not None:
            self.similarity_dir = osp.join('Semester_2', 'TriCL_hater', 'simidataset', self.name, f"{self.name}_similarity_{int(self.similarity * 100)}.pickle")
        else:
            self.similarity_dir = None

        self.split_dir = osp.join(self.dataset_dir, 'splits')
        
        self.load_dataset()
        self.preprocess_dataset()
    def load_dataset(self):
        # 加载 features 和 labels（保持使用原始数据路径）
        with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
            
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

        # 加载 hypergraph，根据 similarity 参数选择路径
        hypergraph_path = (
            self.similarity_dir if (self.similarity is not None and self.similarity_dir and osp.exists(self.similarity_dir))
            else osp.join(self.dataset_dir, 'hypergraph.pickle')
        )
        
        print("hypergraph_path:",hypergraph_path)
        # 使用最终确定的路径加载 hypergraph 文件
        with open(hypergraph_path, 'rb') as f:
            self.hypergraph = pickle.load(f)

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)
        return splits
        
    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])

        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: Optional[int] = None, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        return [train_mask, val_mask, test_mask]

# 修改子类以接受 similarity 参数
class CoraCocitationDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('cocitation', 'cora', similarity=similarity, **kwargs)

class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('cocitation', 'citeseer', similarity=similarity, **kwargs)

class PubmedCocitationDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('cocitation', 'pubmed', similarity=similarity, **kwargs)

class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('coauthorship', 'cora', similarity=similarity, **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('coauthorship', 'dblp', similarity=similarity, **kwargs)


class ZooDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('etc', 'zoo', similarity=similarity, **kwargs)


class NewsDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('etc', '20newsW100', similarity=similarity, **kwargs)


class MushroomDataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('etc', 'Mushroom', similarity=similarity, **kwargs)


class NTU2012Dataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('cv', 'NTU2012', similarity=similarity, **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, similarity=None, **kwargs):
        super().__init__('cv', 'ModelNet40', similarity=similarity, **kwargs)
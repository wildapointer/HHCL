from typing import Optional


from .dataset import (
    CoraCocitationDataset,
    CiteseerCocitationDataset,
    PubmedCocitationDataset,   
    CoraCoauthorshipDataset,
    DBLPCoauthorshipDataset,
    ZooDataset,
    NewsDataset,
    MushroomDataset,
    NTU2012Dataset,
    ModelNet40Dataset,
)


class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora', similarity: Optional[float] = None):
        if dataset_name == 'cora':
            return CoraCocitationDataset(similarity=similarity)
        elif dataset_name == 'citeseer':
            return CiteseerCocitationDataset(similarity=similarity)
        elif dataset_name == 'pubmed':
            return PubmedCocitationDataset(similarity=similarity)
        elif dataset_name == 'cora_coauthor':
            return CoraCoauthorshipDataset(similarity=similarity)
        elif dataset_name == 'dblp_coauthor':
            return DBLPCoauthorshipDataset(similarity=similarity)
        elif dataset_name == 'zoo':
            return ZooDataset(similarity=similarity)
        elif dataset_name == '20newsW100':
            return NewsDataset(similarity=similarity)
        elif dataset_name == 'Mushroom':
            return MushroomDataset(similarity=similarity)
        elif dataset_name == 'NTU2012':
            return NTU2012Dataset(similarity=similarity)
        elif dataset_name == 'ModelNet40':
            return ModelNet40Dataset(similarity=similarity)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

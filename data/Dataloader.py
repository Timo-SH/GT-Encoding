import torch
import random
import torch.utils.data
from collections.abc import Mapping
from typing import List, Optional, Sequence, Union
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data import Data

"""Custom Dataloader functionality for pytorch Geometric, allowing for the BasisNet encoding to be computed.
Adapted from: https://github.com/Graph-COM/SPE/blob/master/src/data_utils/dataloder.py"""

def shuffle_group_graphs(graphs):
    """Function to return a shuffled dataset"""
    for g in graphs:
        random.shuffle(g)
    random.shuffle(graphs)


class SameSizeDataLoader:
    """Dataloader for same sized graphs in a datasets"""
    # dataloader that loads graphs with the same size in a batch

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.curr = -1
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.group_graphs = self.group_same_size(dataset)
        if shuffle:
            shuffle_group_graphs(self.group_graphs)
        self.iter = self.batch_same_size(self.group_graphs, batch_size)

    def group_same_size(self, dataset):
        graphs = []
        size2ind = {}
        ind = 0
        # group same size
        for data in dataset:
            if data.num_nodes in size2ind:
                graphs[size2ind[data.num_nodes]].append(data)
            else:
                size2ind[data.num_nodes] = ind
                ind += 1
                graphs.append([data])
        return graphs

    def batch_same_size(self, graphs, batch_size):
        # batch same size
        batched_graphs = []
        for g in graphs:
            for i in range(0, len(g), batch_size):
                # if i + batch_size <= len(g):
                batched_graphs.append(Batch.from_data_list(g[i : i+batch_size])) # batch graphs into a large graph

        return batched_graphs

    def _reset(self, shuffle):
        self.curr = -1
        if shuffle:
            shuffle_group_graphs(self.group_graphs)
            self.iter = self.batch_same_size(self.group_graphs, self.batch_size)

    def __len__(self):
        return len(self.iter)

    def __next__(self):
        self.curr += 1
        if self.curr == self.__len__(): # reach the end of dataloader
            self._reset(self.shuffle)
            raise StopIteration
        batch = self.iter[self.curr]
        return batch

    def __iter__(self):
        return self



class Collater:
    """Custom Collater function to collate elements in a batch. Is used for cases of differently sized graphs
    and encoding features"""
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        """Collate batch elements manually, in case the torch_geometric Dataloader fails to do so"""
        elem = batch[0]
        if isinstance(elem, BaseData):
            # return Batch.from_data_list(batch, self.follow_batch,
                                        # self.exclude_keys)
            return Batch.from_data_list(batch) # use customized one
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        # TODO Deprecated, remove soon.
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )





class Batch(Data):
    # used to handle BasisNet features
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys():
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys():
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
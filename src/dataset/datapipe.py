import io
import numpy as np

from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    TarArchiveLoader,
    WebDataset,
    IterDataPipe,
    Shuffler,
    ShardingFilter
)

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import torch

class NumpyReader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp
    
    def __iter__(self):
        for file, stream in self.source_dp:
            np_bytes = io.BytesIO(stream.read())
            yield file, np.load(np_bytes)

class SlidingWindow(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, lookback: int = 1, rollout: int = 1, simulation_len: int = 200) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.lookback = lookback
        self.rollout = rollout
        self.real_len = simulation_len - self.lookback - self.rollout+1
    
    def __iter__(self):
        for item in self.source_dp:
            data = item.pop('.data')
            self.real_len = len(data) - self.lookback - self.rollout+1
        
            for i in range(self.real_len):
                sample = item.copy()
                sample['.x'] = data[i:i+self.lookback]
                sample['.y'] = data[i+self.lookback:i+self.lookback+self.rollout]
                yield sample
                

class LookbackMerger(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp
    
    def __iter__(self):
        for item in self.source_dp:
            # ToDo: check order of items
            item['.x'] = item['.x'].transpose((1, 0, 2))
            item['.x'] = item['.x'].reshape((item['.x'].shape[0], -1))
            yield item

class AxisPermuter(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp
    
    def __iter__(self):
        for item in self.source_dp:
            item['.x'] = item['.x'].transpose((0, 2, 1))
            item['.y'] = item['.y'].transpose((0, 2, 1))
            yield item

class GraphCreator(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, k: int = 10) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.k = k
    
    def __iter__(self):
        for item in self.source_dp:
            # transform to tensors
            x = torch.tensor(item['.x'], dtype=torch.float32)
            y = torch.tensor(item['.y'], dtype=torch.float32)
            points = torch.tensor(item['.points'], dtype=torch.float32)

            # create pyg graphs
            x_graph = Data(x=x, pos=points)
            y_graph = Data(x=y, pos=points)

            # generate knn edges
            transformation = KNNGraph(k=self.k)
            x_graph = transformation(x_graph)
            y_graph = transformation(y_graph)

            # edge_attribute
            edge_attr = points[y_graph.edge_index.T].reshape((y_graph.edge_index.size(1), -1))
            y_graph.edge_attr = edge_attr
            x_graph.edge_attr = edge_attr

            y_graph = [Data(x=y_graph.x[i], pos=points, edge_index=y_graph.edge_index, edge_attr=y_graph.edge_attr) for i in range(len(y_graph.x))]
            
            item['.x'] = x_graph
            item['.y'] = y_graph
            item['.points'] = points

            yield item


def create_datapipes(
        base_path: str = "data",
        split: str = "train",
        equation: str = "wave", 
        support: str = "cloud", 
        num_points: str = "low",
        lookback: int = 1,
        rollout: int = 1,
        as_graph: bool = False,
        k: int = 10):
    
    datapipe = FileLister(f"{base_path}/{equation}/{split}", f"{support}_{num_points}.tar")
    datapipe = FileOpener(datapipe, mode="b")
    datapipe = TarArchiveLoader(datapipe)
    datapipe = NumpyReader(datapipe)
    datapipe = WebDataset(datapipe)
    datapipe = ShardingFilter(datapipe)
    datapipe = SlidingWindow(datapipe, lookback=lookback, rollout=rollout)
    datapipe = AxisPermuter(datapipe)
    datapipe = LookbackMerger(datapipe)
    datapipe = Shuffler(datapipe, buffer_size=10000)
    if as_graph:
        datapipe = GraphCreator(datapipe, k = k)
    return datapipe
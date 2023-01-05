from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import torch


from src.dataset.dataset_base import DynaBenchBase

class DynaBench(DynaBenchBase):
    def __init__(self, name="dyna-benchmark", mode="train", equation="gas_dynamics", support="high", task="forecast", base_path="data", lookback=0, rollout=1):
        super().__init__(name=name, 
                         mode=mode,
                         equation=equation, 
                         support=support,
                         task=task, 
                         base_path=base_path,
                         lookback=lookback, 
                         rollout=rollout)

    def additional_transforms(self, x, y, points):
        # transform to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)

        # create pyg graphs
        x_graph = Data(x=torch.hstack([x, points]), pos=points)
        y_graph = Data(x=y, pos=points)

        # generate knn edges
        transformation = KNNGraph(k=10)
        x_graph = transformation(x_graph)
        y_graph = transformation(y_graph)
        
        return x_graph, y_graph, points

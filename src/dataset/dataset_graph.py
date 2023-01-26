from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import torch


from src.dataset.dataset_base import DynaBenchBase

class DynaBenchGraph(DynaBenchBase):
    def __init__(self, 
                 name="dyna-benchmark", 
                 mode="train", 
                 equation="gas_dynamics", 
                 support="cloud",
                 num_points="high",
                 task="forecast", 
                 base_path="data", 
                 lookback=1, 
                 rollout=1,
                 test_ratio=0.1,
                 val_ratio=0.1,
                 merge_lookback=True,
                 k=10):
        super().__init__(name=name, 
                         mode=mode,
                         equation=equation, 
                         support=support,
                         num_points=num_points,
                         task=task, 
                         base_path=base_path,
                         lookback=lookback, 
                         rollout=rollout,
                         test_ratio=test_ratio,
                         val_ratio=val_ratio,
                         merge_lookback=merge_lookback)
        self.k = k
        if not self.merge_lookback:
            UserWarning("Using the graph dataset without merging lookback and channels is not recommended!")

    def additional_transforms(self, x, y, points):
        if self.support == "grid":
            x = x.reshape((x.shape[0], -1))
            y = y.reshape((y.shape[0], y.shape[1], -1))
            points = points.reshape((-1, 2))

            x = x.transpose((1,0))
            y = y.transpose((2,1,0)) if self.task == "forecast" else y.transpose((1,0))

        # transform to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)

        # create pyg graphs
        x_graph = Data(x=x, pos=points)
        y_graph = Data(x=y, pos=points)

        # generate knn edges
        transformation = KNNGraph(k=self.k)
        x_graph = transformation(x_graph)
        y_graph = transformation(y_graph)

        if self.task == "forecast":
            y_graph = [Data(x=y_graph.x[i], pos=points, edge_index=y_graph.edge_index) for i in range(self.rollout)]
        
        return x_graph, y_graph, points

from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import torch
from typing import Literal


from .dataset_base import DynaBenchBase

class DynaBenchGraph(DynaBenchBase):
    def __init__(
        self,
        mode: Literal['train', 'val', 'test']="train",
        equation: Literal['advection', 'brusselator', 'gas_dynamics', 'kuramoto_sivashinsky', 'wave']="gas_dynamics",
        task: Literal['forecast', 'evolution']="forecast",
        support: Literal['cloud', 'grid']="grid",
        num_points: Literal['high', 'low']="high",
        base_path: str="data",
        lookback: int=1,
        rollout: int=1,
        test_ratio: float=0.1,
        val_ratio:  float=0.1,
        merge_lookback: bool=True,
        k: int=10,
        *args,
        **kwargs,
    ):

        """_summary_
        Initializes a pytorch geometric dataset with selected parameters. The data is loaded lazily.

        :param mode: the selection of data to use (train/val/test), defaults to "train"
        :type mode: Literal[&#39;train&#39;, &#39;val&#39;, &#39;test&#39;], optional
        :param equation: the equation to use, defaults to "gas_dynamics"
        :type equation: Literal[&#39;advection&#39;, &#39;brusselator&#39;, &#39;gas_dynamics&#39;, &#39;kuramoto_sivashinsky&#39;, &#39;wave&#39;], optional
        :param task: Which task is to be calculated, defaults to "forecast"
        :type task: Literal[&#39;forecast&#39;, &#39;evolution&#39;], optional
        :param support: Structure of the points at which the measurements are recorded, defaults to "grid"
        :type support: Literal[&#39;cloud&#39;, &#39;grid&#39;], optional
        :param num_points: _description_, defaults to "high"
        :type num_points: Literal[&#39;high&#39;, &#39;low&#39;], optional
        :param base_path: location where the data is stored, defaults to "data"
        :type base_path: str, optional
        :param lookback: How many past states are used to make the prediction. The additional states can be concatenated along the channel dimension if merge_lookback is set to True, defaults to 1
        :type lookback: int, optional
        :param rollout: How many steps should be predicted in a closed loop setting. Only used for forecast task, defaults to 1
        :type rollout: int, optional
        :param test_ratio: What fraction of simulations to set aside for testing, defaults to 0.1
        :type test_ratio: float, optional
        :param val_ratio: What fraction of simulations to set aside for validation, defaults to 0.1
        :type val_ratio: float, optional
        :param merge_lookback: Whether to merge the additional lookback information into the channel dimension, defaults to True
        :type merge_lookback: bool, optional
        :raises RuntimeError: Data not found. Generate the data first.
        :raises RuntimeError: Data not found. Generate the data first.
        :raises RuntimeError: At least 3 simulations need to be run in order to use the dataset (1 each for train/val/test)
        :param k: Number of neighbors to use for building the graph, defaults to 10
        :type k: int, optional
        """
        super().__init__(mode=mode,
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

        # edge_attribute
        edge_attr = points[y_graph.edge_index.T].reshape((y_graph.edge_index.size(1), -1))
        y_graph.edge_attr = edge_attr
        x_graph.edge_attr = edge_attr

        if self.task == "forecast":
            y_graph = [Data(x=y_graph.x[i], pos=points, edge_index=y_graph.edge_index, edge_attr=y_graph.edge_attr) for i in range(self.rollout)]
        
        return x_graph, y_graph, points

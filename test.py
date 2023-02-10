from src.model.neural_operator import KernelNN
from src.dataset import DynaBenchDataModule
from src.dataset.dataset_graph import DynaBenchGraph

import torch


ds = DynaBenchGraph(support="cloud")
#ds.setup()
#dl = 



width = 16
ker_width = 64
depth = 6
edge_features = 4
node_features = 4


model = KernelNN(width,ker_width,depth,edge_features,node_features)

t = model(ds[0][0])

t
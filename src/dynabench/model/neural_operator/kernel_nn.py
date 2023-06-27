import torch
import torch.nn.functional as F

from .utilities import DenseNet
from .nn_conv import NNConv_old

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, input_size=1, lookback=1, spatial_dimensions=2):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(input_size*lookback, width)

        kernel = DenseNet([spatial_dimensions*2, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')


        self.fc2 = torch.nn.Linear(width, input_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        data.x = x
        return data

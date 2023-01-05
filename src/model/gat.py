from torch_geometric.nn import GATv2Conv
from torch import nn
import torch.functional as F


class GATNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.input_layer = GATv2Conv(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([GATv2Conv(hidden_size, hidden_size) for _ in range(hidden_layers-1)])
        self.output_layer = GATv2Conv(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_layer(x, edge_index)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = self.activation(x)

        x = self.output_layer(x, edge_index)
        return x
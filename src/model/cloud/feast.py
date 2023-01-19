from torch_geometric.nn import FeaStConv
from torch import nn
import torch

class FeaStNet(nn.Module):
    def __init__(self, input_size, lookback,
                 input_layer_heads, hidden_layer_heads, output_layer_heads,
                 hidden_size, hidden_layers=1, 
                 spatial_dimensions=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.input_layer_heads = input_layer_heads
        self.hidden_layer_heads = hidden_layer_heads
        self.output_layer_heads = output_layer_heads
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.spatial_dimensions = spatial_dimensions

        self.input_layer = FeaStConv(input_size*lookback+spatial_dimensions, hidden_size, heads=input_layer_heads)

        self.hidden_layers = nn.ModuleList([FeaStConv(hidden_size+spatial_dimensions, hidden_size, heads=hidden_layer_heads) for _ in range(hidden_layers-1)])
        self.output_layer = FeaStConv(hidden_size+spatial_dimensions, input_size, heads=output_layer_heads)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.input_layer(torch.hstack([x, pos]), edge_index)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(torch.hstack([x, pos]), edge_index)
            x = self.activation(x)

        x = self.output_layer(torch.hstack([x, pos]), edge_index)
        data.x = x
        return data
from torch import nn
import torch

class BaselineZero(nn.Module):
    def __init__(self, input_size: int, lookback: int, spatial_dimensions: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.spatial_dimensions = spatial_dimensions

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        data.x = torch.zeros_like(x[:, -self.input_size:])
        return data
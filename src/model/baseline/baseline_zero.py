from torch import nn
import torch


class BaselineZero(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        data.x = torch.zeros_like(x[:, -self.output_size:])
        return data
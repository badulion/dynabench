from torch import nn
import torch


class BaselineZero(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        data.x = torch.zeros_like(x)
        return data
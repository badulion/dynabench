from torch import nn
import gin

@gin.configurable
class BaselinePersistence(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        data.x = x[:, -self.output_size:]
        return data
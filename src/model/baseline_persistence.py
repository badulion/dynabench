from torch import nn

class BaselinePersistence(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        return data
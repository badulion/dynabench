import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self,
                 channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same', padding_mode='circular')
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same', padding_mode='circular')
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x
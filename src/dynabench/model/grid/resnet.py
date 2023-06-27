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

class SimpleResnet(nn.Module):
    def __init__(self,
                 input_size, 
                 lookback, 
                 resblock_layers,
                 resblock_channels):
        super().__init__()

        resblock_list = [ResBlock(resblock_channels) for i in range(resblock_layers)]
        self.model = nn.Sequential(
            nn.Conv2d(input_size*lookback, resblock_channels, 3, padding='same', padding_mode='circular'),
            nn.ReLU(),
            *resblock_list,
            nn.Conv2d(resblock_channels, input_size, 3, padding='same', padding_mode='circular')
        )

    def forward(self, x):
        return self.model(x)
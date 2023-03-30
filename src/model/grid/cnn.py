import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_size, 
                 lookback, 
                 hidden_layers,
                 hidden_channels,
                 kernel_size=3):
        super().__init__()
        hidden_list = []
        for i in range(hidden_layers):
            conv_layer = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', padding_mode='circular')
            hidden_list.append(conv_layer)
            hidden_list.append(nn.ReLU())
            
        self.model = nn.Sequential(
            nn.Conv2d(lookback*input_size, hidden_channels, 3, padding='same', padding_mode='circular'),
            nn.ReLU(),
            *hidden_list,
            nn.Conv2d(hidden_channels, input_size, 3, padding='same', padding_mode='circular')
        )
        self.input_size = input_size

    def forward(self, x):
        return self.model(x)
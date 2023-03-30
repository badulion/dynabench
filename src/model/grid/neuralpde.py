from .cnn import SimpleCNN
import torch.nn as nn
import torch

class NeuralPDE(nn.Module):
    def __init__(self,
                 input_size, 
                 lookback, 
                 hidden_layers,
                 hidden_channels,
                 kernel_size=3):
        super().__init__()
        self.dudt = SimpleCNN(input_size=input_size, 
                              lookback=lookback, 
                              hidden_layers=hidden_layers, 
                              hidden_channels=hidden_channels)

        self.input_size = input_size

    def forward(self, x):
        return x[:,-self.input_size:] + self.dudt(x)

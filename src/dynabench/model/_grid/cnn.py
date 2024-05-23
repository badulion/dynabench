import torch.nn as nn
import torch

from typing import Tuple

class CNN(nn.Module):
    """
        Simple 2D CNN model for grid data.

        Parameters
        ----------
        input_size : int
            Number of input channels.
        output_size : int
            Number of output channels.
        hidden_layers : int
            Number of hidden layers. Default is 1.
        hidden_channels : int
            Number of channels in each hidden layer. Default is 64.
        padding : int | str | Tuple[int]
            Padding size. If 'same', padding is calculated to keep the input size the same as the output size. Default is 'same'.
        padding_mode : str
            What value to pad with. Can be 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        kernel_size : int
            Size of the kernel. Default is 3.
        activation : str
            Activation function to use. Can be one of `torch.nn activation functions <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`_. Default is 'relu'.

    """
    def __init__(self,
                 input_size: int, 
                 output_size: int, 
                 hidden_layers: int = 1,
                 hidden_channels: int = 64,
                 padding: int | str | Tuple[int] = 'same',
                 padding_mode: str = 'circular',
                 kernel_size: int = 3,
                 activation: str = 'ReLU'):
        super().__init__()
        self.input_layer = nn.Conv2d(input_size, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.hidden_layers = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_channels, output_size, kernel_size, padding=padding, padding_mode=padding_mode)
            
        self.activation = getattr(nn, activation)()  

    def forward(self, x: torch.Tensor):
        """
            Forward pass of the model. Should not be called directly, instead call the model instance.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_size, height, width).

            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, output_size, height, width).
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_layer(x)
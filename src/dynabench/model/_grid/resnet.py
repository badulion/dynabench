import torch.nn as nn
import torch

from typing import Tuple

class BottleneckResBlock(nn.Module):
    """
        Bottleneck Residual Block for ResNet model. Consists of 3 convolutional layers with kernel size 1, k and 1 respectively. 
        The bottleneck design is used to reduce the number of parameters in the model.


        Parameters
        ----------
        channels : int
            Number of input channels.
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
                 channels: int = 256,
                 padding: int | str | Tuple[int] = 'same',
                 padding_mode: str = 'circular',
                 kernel_size: int = 3,
                 activation: str = 'relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//4, 1)
        self.conv2 = nn.Conv2d(channels//4, channels//4, kernel_size, padding=padding, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(channels//4, channels, 1)
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
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = x + residual
        x = self.activation(x)
        return x

class ResBlock(nn.Module):
    """
        Simple Residual Block for ResNet model. Consists of 2 convolutional layers with kernel size k. 
        The input is added to the output of the second convolutional layer.


        Parameters
        ----------
        channels : int
            Number of input channels.
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
                 channels: int = 64,
                 padding: int | str | Tuple[int] = 'same',
                 padding_mode: str = 'circular',
                 kernel_size: int = 3,
                 activation: str = 'relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
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
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x

class ResNet(nn.Module):
    """
        Simple 2D ResNet model for grid data.

        Parameters
        ----------
        input_size : int
            Number of input channels.
        output_size : int  
            Number of output channels.
        resblock_layers : int
            Number of residual blocks. Default is 1.
        resblock_channels : int
            Number of channels in each residual block. Default is 64.
        resblock_type : str
            Type of residual block to use. Can be 'simple' or 'bottleneck'. Default is 'simple'.
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
                 resblock_layers: int = 1,
                 resblock_channels: int = 64,
                 resblock_type: str = 'simple',
                 padding: int | str | Tuple[int] = 'same',
                 padding_mode: str = 'circular',
                 kernel_size: int = 3,
                 activation: str = 'ReLU'):
        super().__init__()

        if resblock_type == 'simple':
            _ResBlock = ResBlock
        elif resblock_type == 'bottleneck':
            _ResBlock = BottleneckResBlock
        else:
            raise ValueError(f"Unknown resblock type: {resblock_type}")
        
        self.input_layer = nn.Conv2d(input_size, resblock_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.resblocks = [ _ResBlock(channels=resblock_channels, padding=padding, padding_mode=padding_mode, kernel_size=kernel_size, activation=activation) for _ in range(resblock_layers)]
        self.output_layer = nn.Conv2d(resblock_channels, output_size, kernel_size, padding=padding, padding_mode=padding_mode)
        self.activation = getattr(nn, activation)()


    def forward(self, x):
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
        for resblock in self.resblocks:
            x = resblock(x)
        return self.output_layer(x)
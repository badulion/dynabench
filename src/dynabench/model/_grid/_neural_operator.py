# MIT License

# Copyright (c) 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_modes: tuple):
        '''
        Spectral convolution using fft.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        n_modes: tuple of ints (2d -> 2)
            Modes used in the Fourier Transform. At most floor(N/2) + 1
        '''
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1, self.n2 = n_modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, dtype=torch.cfloat))
    
    def compl_mul2d(self, a: torch.Tensor, b: torch.Tensor):
        '''
        Parameters
        ----------
        a: torch.Tensor
            Input tensor (batch, in_channel, x, y)
        b: torch.Tensor
            Weights
        '''
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor):
        n1, n2 = self.n1, self.n2
        batchsize = x.shape[0]

        # Compute Fast Fourier Transform
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])

        # Choose respective modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :n1, :n2] = self.compl_mul2d(x_ft[:, :, :n1, :n2], self.weights1)
        out_ft[:, :, -n1:, :n2] = self.compl_mul2d(x_ft[:, :, -n1:, :n2], self.weights2)

        # Compute Inverse Fast Fourier Transform
        out = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return out


class NeuralFourierBlock2d(nn.Module):
    def __init__(self, width: int, n_modes: tuple, activation=True):
        '''
        The Fourier Block implements the entire Fourier Layer as defined in paper [Fourier Neural Operators](https://arxiv.org/pdf/2010.08895).

        Parameters
        ----------
        width: int
            Higher dimensional number of channels.
        n_modes: tuple of ints (2d -> 2)
            Modes used in the Fourier Transform. At most floor(N/2) + 1
        activation: bool
            Apply activation function after the block.
        '''
        super(NeuralFourierBlock2d, self).__init__()
        # Fourier Convolutional layer
        # Fourier -> linear transform (R) + filter out higher modes -> inv Fourier
        self.conv = SpectralConv2d(width, width, n_modes)
        # W - local linear transform on input
        self.W = nn.Conv1d(width, width, 1)
        # Activation
        self.bn = torch.nn.BatchNorm2d(width)
        self.activation = activation
        self.width = width

    def forward(self, x: torch.Tensor):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        # Apply Fourier Convolution
        out = self.conv(x)
        # Apply local linear transform + Fourier Convolution output
        out += self.W(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # Apply Activation
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class FourierNeuralOperator(nn.Module):
    def __init__(self, n_layers: int, n_modes: tuple, width: int, channels: int, pad: tuple = None, **kwargs):
        '''
        Neural Operator model from the Paper [Fourier Neural Operator](https://arxiv.org/abs/2010.08895).
        Implementation of Fourier model for 2D data in [Git](https://github.com/alexander-telepov/fourier-neural-operator/tree/main).
        
        Parameters
        ----------
        n_layers: int
            Number of Fourier Blocks in the model.
        n_modes: tuple of ints (2d -> 2)
            Modes used in the Fourier Transform.
        width: int
            Higher dimensional number of channels.
        t_in: int
            Number of input time steps.
        t_out: int
            Number of output time steps.
        pad: bool
            Padding applied to the input.
        '''
        super(FourierNeuralOperator, self).__init__()
        self.n_modes = n_modes
        self.width = width
        self.pad = pad

        # Embedding layer
        self.fc0 = nn.Linear(channels, width)

        # Fourier Blocks
        layers = [NeuralFourierBlock2d(width, n_modes, activation=True) for i in range(n_layers - 1)]
        layers.append(NeuralFourierBlock2d(width, n_modes, activation=False))
        self.fourier_blocks = nn.Sequential(*layers)

        # Reduce to output dimension
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, channels)

    def forward(self, x: torch.Tensor):
        # batching
        batched = True
        if x.dim() == 3:
            x = x.unsqueeze(0)
            batched = False

        if self.pad is not None:
            x = F.pad(x, self.pad, "replicate")
        # permute for embedding
        x = x.permute(0, 2, 3, 1)
        # Lift to higher dimensional layer space
        out = self.fc0(x)

        # Fourier Blocks
        out = out.permute(0, 3, 1, 2)
        out = self.fourier_blocks(out)
        out = out.permute(0, 2, 3, 1)

        # Reduce to output dimension
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        out = out.permute(0, 3, 1, 2)

        if self.pad is not None:
            out = out[:,:,self.pad[0]:-self.pad[1],self.pad[2]:-self.pad[3]]

        if not batched: out = out.squeeze(0)
        
        return out
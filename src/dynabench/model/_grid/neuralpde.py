import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint, odeint
from .cnn import CNN

from typing import List, Optional

class NeuralPDE(nn.Module):
    """
        Neural PDE model for grid data. The model combines a CNN with a differentiable ODE solver to learn the dynamics of the data using the method of lines. The CNN is used to approximate the spatial derivatives of the data, while the ODE solver is used to approximate the temporal evolution of the data.
        The model has been taken from `NeuralPDE: Modelling Dynamical Systems from Data <https://arxiv.org/abs/2111.07671>`_ by Dulny et al.

        Parameters
        ----------
        input_dim : int
            Number of input channels.
        hidden_channels : int
            Number of channels in each hidden layer of the CNN. Default is 64.
        hidden_layers : int 
            Number of hidden layers in the CNN. Default is 1.
        solver : dict
            Dictionary of solver parameters. Default is {"method": "dopri5"}.
        use_adjoint : bool
            Whether to use the adjoint method for backpropagation. Default is True.

    """
    def __init__(self,
                 input_dim: int,
                 hidden_channels: int = 64,
                 hidden_layers: int = 1,
                 solver: dict = {"method": "dopri5"},
                 use_adjoint: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        
        self.cnn = CNN(
            input_size=input_dim,
            output_size=input_dim,
            hidden_layers=hidden_layers,
            hidden_channels=hidden_channels
        )
        
        self.solver = solver
        self.use_adjoint = use_adjoint

    def _ode(self, t, x):
        return self.cnn(x)
    
    def forward(self, 
                x: torch.Tensor, 
                t_eval: List[float]=[0.0, 1.0]):
        
        """
            Forward pass of the model. Should not be called directly, instead call the model instance.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_size, height, width).
            t_eval : List[float], default [0.0, 1.0]
                List of times to evaluate the ODE solver at. Default is [0.0, 1.0].

            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, rollout, output_size, height, width).
        """

        t_eval = torch.tensor(t_eval, dtype=x.dtype, device=x.device)
        if self.use_adjoint:
            pred = odeint_adjoint(self._ode, x, t_eval, **self.solver, adjoint_params=self.cnn.parameters())[1:]
        else:
            pred =  odeint(self._ode, x, t_eval, **self.solver)[1:]
        
        pred = torch.swapaxes(pred, 0, 1)
        return pred
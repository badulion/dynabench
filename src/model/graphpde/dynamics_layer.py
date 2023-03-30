import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class Dynamics(MessagePassing):
    def __init__(self, gamma, phi):
        super(Dynamics, self).__init__(aggr='mean', flow='target_to_source')
        self.gamma = gamma
        self.phi = phi

    def forward(self, u, edge_index, pos):
        return self.propagate(edge_index, u=u, pos=pos)

    def message(self, u_i, u_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        phi_input = torch.cat([u_i, u_j-u_i, rel_pos], dim=1)
        return self.phi(phi_input)

    def update(self, aggr, u):
        gamma_input = torch.cat([u, aggr], dim=1)
        dudt = self.gamma(gamma_input)
        return dudt
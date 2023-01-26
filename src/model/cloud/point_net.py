from torch_geometric.nn import PointNetConv
from torch import nn
from ..components import MLP

class PointNet(nn.Module):
    def __init__(self, 
                 input_size, 
                 lookback, 
                 intermediate_hidden_size,
                 hidden_size, 
                 local_nn_hidden_size,
                 local_nn_hidden_layers,
                 global_nn_hidden_size,
                 global_nn_hidden_layers,
                 hidden_layers=1,
                 spatial_dimensions=2) -> None:

        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.input_layer = PointNetConv(local_nn=MLP(input_size=input_size*lookback+spatial_dimensions, output_size=intermediate_hidden_size, hidden_size=local_nn_hidden_size, hidden_layers=local_nn_hidden_layers), 
                                        global_nn=MLP(input_size=intermediate_hidden_size, output_size=hidden_size, hidden_size=global_nn_hidden_size, hidden_layers=global_nn_hidden_layers))

        self.hidden_layers = nn.ModuleList([
            PointNetConv(local_nn=MLP(input_size=hidden_size+spatial_dimensions, output_size=intermediate_hidden_size, hidden_size=local_nn_hidden_size, hidden_layers=local_nn_hidden_layers), 
                         global_nn=MLP(input_size=intermediate_hidden_size, output_size=hidden_size, hidden_size=global_nn_hidden_size, hidden_layers=global_nn_hidden_layers))
        for _ in range(hidden_layers-1)])

        self.output_layer = PointNetConv(local_nn=MLP(input_size=hidden_size+spatial_dimensions, output_size=intermediate_hidden_size, hidden_size=local_nn_hidden_size, hidden_layers=local_nn_hidden_layers), 
                                         global_nn=MLP(input_size=intermediate_hidden_size, output_size=input_size, hidden_size=global_nn_hidden_size, hidden_layers=global_nn_hidden_layers))
        self.activation = nn.ReLU()

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.input_layer(x, pos, edge_index)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x, pos, edge_index)
            x = self.activation(x)

        x = self.output_layer(x, pos, edge_index)
        data.x = x
        return data
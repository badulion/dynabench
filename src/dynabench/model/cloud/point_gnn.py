#from torch_geometric.nn.conv import PointGNNConv
from torch import nn
from ..components import MLP, PointGNNConv

class PointGNN(nn.Module):
    def __init__(self, 
                 input_size, 
                 lookback, 
                 intermediate_hidden_size,
                 hidden_size, 
                 encoder_nn_hidden_size,
                 encoder_nn_hidden_layers,
                 decoder_nn_hidden_size,
                 decoder_nn_hidden_layers,
                 offset_nn_hidden_size,
                 offset_nn_hidden_layers,
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

        self.input_layer = MLP(input_size=input_size*lookback, output_size=hidden_size, hidden_size=encoder_nn_hidden_size, hidden_layers=encoder_nn_hidden_layers)

        self.hidden_layers = nn.ModuleList([
            PointGNNConv(mlp_h=MLP(input_size=hidden_size, output_size=spatial_dimensions, hidden_size=offset_nn_hidden_size, hidden_layers=offset_nn_hidden_layers),
                         mlp_f=MLP(input_size=hidden_size+spatial_dimensions, output_size=intermediate_hidden_size, hidden_size=local_nn_hidden_size, hidden_layers=local_nn_hidden_layers),
                         mlp_g=MLP(input_size=intermediate_hidden_size, output_size=hidden_size, hidden_size=global_nn_hidden_size, hidden_layers=global_nn_hidden_layers))
        for _ in range(hidden_layers-1)])

        self.output_layer = MLP(input_size=hidden_size, output_size=input_size, hidden_size=decoder_nn_hidden_size, hidden_layers=decoder_nn_hidden_layers)

        self.activation = nn.ReLU()

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x, pos, edge_index)
            x = self.activation(x)

        x = self.output_layer(x)
        data.x = x
        return data
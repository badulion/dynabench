# models
from dynabench.model._grid.cnn import CNN
from dynabench.model._grid.resnet import ResNet
from dynabench.model.point.point_transformer import PointTransformerV1
from dynabench.model._grid._neural_operator import FourierNeuralOperator
from dynabench.model.point._fno import Geo_FNO
from dynabench.model._grid.neuralpde import NeuralPDE

import torch
import pytest


# data inputs
@pytest.fixture(scope="module")
def batched_input_grid_low():
    return (torch.ones((16, 1, 15, 15)), torch.arange(2))

@pytest.fixture(scope="module")
def batched_input_grid_med():
    return (torch.ones((16, 1, 22, 22)), torch.arange(2))

@pytest.fixture(scope="module")
def batched_input_grid_high():
    return (torch.ones((16, 1, 30, 30)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_grid_low():
    return (torch.ones((1, 15, 15)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_grid_med():
    return (torch.ones((1, 22, 22)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_grid_high():
    return (torch.ones((1, 30, 30)), torch.arange(2))

# for point model data input is x, p
@pytest.fixture(scope="module")
def batched_input_point_low():
    return (torch.ones((16, 225, 1)), torch.ones((16, 225, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def batched_input_point_med():
    return (torch.ones((16, 484, 1)), torch.ones((16, 484, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def batched_input_point_high():
    return (torch.ones((16, 900, 1)), torch.ones((16, 900, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_point_low():
    return (torch.ones((225, 1)), torch.ones((225, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_point_med():
    return (torch.ones((484, 1)), torch.ones((484, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_point_high():
    return (torch.ones((900, 1)), torch.ones((900, 2)), torch.arange(2))

# channel data
@pytest.fixture(scope="module")
def batched_input_grid_low_channel():
    return (torch.ones((16, 4, 15, 15)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_grid_low_channel():
    return (torch.ones((4, 15, 15)), torch.arange(2))

@pytest.fixture(scope="module")
def batched_input_point_low_channel():
    return (torch.ones((16, 225, 4)), torch.ones((16, 225, 2)), torch.arange(2))

@pytest.fixture(scope="module")
def unbatched_input_point_low_channel():
    return (torch.ones((225, 4)), torch.ones((225, 2)), torch.arange(2))

# models for data resolutions low, medium, high
@pytest.fixture
def default_cnn():
    return CNN(input_size=1, output_size=1, hidden_channels=16, hidden_layers=3, kernel_size=3, padding=1, activation="ReLU")

@pytest.fixture
def default_cnn_channel():
    return CNN(input_size=4, output_size=4, hidden_channels=16, hidden_layers=3, kernel_size=3, padding=1, activation="ReLU")

@pytest.fixture
def default_resnet():
    return ResNet(input_size=1, output_size=1, activation="ReLU")

@pytest.fixture
def default_resnet_channel():
    return ResNet(input_size=4, output_size=4, activation="ReLU")

@pytest.fixture
def default_point_transformer_v1_low():
    return PointTransformerV1(input_dim=1, num_points=225, num_neighbors=16, num_blocks=3, transformer_dim=512)

@pytest.fixture
def default_point_transformer_v1_low_channel():
    return PointTransformerV1(input_dim=4, num_points=225, num_neighbors=16, num_blocks=3, transformer_dim=512)

@pytest.fixture
def default_point_transformer_v1_med():
    return PointTransformerV1(input_dim=1, num_points=484, num_neighbors=16, num_blocks=3, transformer_dim=512)

@pytest.fixture
def default_point_transformer_v1_high():
    return PointTransformerV1(input_dim=1, num_points=900, num_neighbors=16, num_blocks=3, transformer_dim=512)

@pytest.fixture
def default_neural_operator():
    return FourierNeuralOperator(n_layers=5, n_modes=[8,8], width=64, channels=1)

@pytest.fixture
def default_neural_operator_channel():
    return FourierNeuralOperator(n_layers=5, n_modes=[8,8], width=64, channels=4)

@pytest.fixture
def default_fno_geo():
    return Geo_FNO(width=32, modes=(8,8), channels=1, grid_size=(20,20), num_blocks=3)

@pytest.fixture
def default_fno_geo_channel():
    return Geo_FNO(width=32, modes=(8,8), channels=4, grid_size=(20,20), num_blocks=3)

@pytest.fixture
def default_neuralpde():
    return NeuralPDE(input_dim=1)

@pytest.fixture
def default_neuralpde_channel():
    return NeuralPDE(input_dim=4)
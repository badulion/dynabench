from dynabench.model._grid.cnn import CNN
import torch

import pytest


@pytest.fixture
def batched_input_low():
    return torch.ones((16, 1, 15, 15))

@pytest.fixture
def unbatched_input_low():
    return torch.ones((1, 15, 15))

@pytest.fixture
def default_cnn():
    return CNN(input_size=1, output_size=1, hidden_channels=16, hidden_layers=3, kernel_size=3, padding=1, activation="ReLU")
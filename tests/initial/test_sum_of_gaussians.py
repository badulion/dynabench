import numpy as np
from unittest.mock import MagicMock

def test_sum_of_gaussians_initial_condition_init(sum_of_gaussians_initial_condition):
    assert sum_of_gaussians_initial_condition.components == 3
    assert sum_of_gaussians_initial_condition.zero_level == 0.5
    assert sum_of_gaussians_initial_condition.parameters == {"param1": 1.0}
    assert sum_of_gaussians_initial_condition.spatial_dim == 2

def test_sum_of_gaussians_initial_condition_str(sum_of_gaussians_initial_condition):
    assert str(sum_of_gaussians_initial_condition) == "I(x, y) = sum_i A_i exp(-40(x-x_i)^2 + (y-y_i)^2)"

def test_sum_of_gaussians_initial_condition_generate(sum_of_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    initial_condition = sum_of_gaussians_initial_condition.generate(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= -1.5)  # Considering the zero_level and random uniform range
    assert np.all(initial_condition <= 2.5)   # Considering the zero_level and random uniform range

def test_sum_of_gaussians_initial_condition_generate_reproducibility(sum_of_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    initial_condition_1 = sum_of_gaussians_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = sum_of_gaussians_initial_condition.generate(grid, random_state=42)
    assert np.array_equal(initial_condition_1, initial_condition_2)

def test_sum_of_gaussians_initial_condition_call(sum_of_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    initial_condition = sum_of_gaussians_initial_condition(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= -1.5)  # Considering the zero_level and random uniform range
    assert np.all(initial_condition <= 2.5)   # Considering the zero_level and random uniform range

def test_sum_of_gaussians_initial_condition_generate_different_seeds(sum_of_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    initial_condition_1 = sum_of_gaussians_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = sum_of_gaussians_initial_condition.generate(grid, random_state=43)
    assert not np.array_equal(initial_condition_1, initial_condition_2)
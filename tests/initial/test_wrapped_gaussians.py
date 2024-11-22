import numpy as np
from unittest.mock import MagicMock

def test_wrapped_gaussians_initial_condition_init(wrapped_gaussians_initial_condition):
    assert wrapped_gaussians_initial_condition.components == 3
    assert wrapped_gaussians_initial_condition.zero_level == 0.5
    assert wrapped_gaussians_initial_condition.periodic_levels == 10
    assert wrapped_gaussians_initial_condition.parameters == {"param1": 1.0}
    assert wrapped_gaussians_initial_condition.spatial_dim == 2

def test_wrapped_gaussians_initial_condition_str(wrapped_gaussians_initial_condition):
    assert str(wrapped_gaussians_initial_condition) == "I(x, y) = sum_i A_i exp(-40(x-x_i)^2 + (y-y_i)^2)"

def test_wrapped_gaussians_initial_condition_generate(wrapped_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    grid.get_random_point_within_domain = MagicMock(return_value=(0.5, 0.5))
    initial_condition = wrapped_gaussians_initial_condition.generate(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= -1.5)  # Considering the zero_level and random uniform range
    assert np.all(initial_condition <= 2.5)   # Considering the zero_level and random uniform range

def test_wrapped_gaussians_initial_condition_generate_reproducibility(wrapped_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    grid.get_random_point_within_domain = MagicMock(return_value=(0.5, 0.5))
    initial_condition_1 = wrapped_gaussians_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = wrapped_gaussians_initial_condition.generate(grid, random_state=42)
    assert np.array_equal(initial_condition_1, initial_condition_2)

def test_wrapped_gaussians_initial_condition_call(wrapped_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    grid.get_random_point_within_domain = MagicMock(return_value=(0.5, 0.5))
    initial_condition = wrapped_gaussians_initial_condition(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= -1.5)  # Considering the zero_level and random uniform range
    assert np.all(initial_condition <= 2.5)   # Considering the zero_level and random uniform range

def test_wrapped_gaussians_initial_condition_generate_different_seeds(wrapped_gaussians_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    grid.get_random_point_within_domain = MagicMock(return_value=(0.5, 0.5))
    initial_condition_1 = wrapped_gaussians_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = wrapped_gaussians_initial_condition.generate(grid, random_state=43)
    assert not np.array_equal(initial_condition_1, initial_condition_2)
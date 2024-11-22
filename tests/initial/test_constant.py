import numpy as np
from unittest.mock import MagicMock

def test_constant_initial_condition_init(constant_initial_condition):
    assert constant_initial_condition.value == 5.0
    assert constant_initial_condition.parameters == {"param1": 1.0}
    assert constant_initial_condition.spatial_dim == 2

def test_constant_initial_condition_str(constant_initial_condition):
    assert str(constant_initial_condition) == "I(x, y) = 5.0"

def test_constant_initial_condition_generate(constant_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition = constant_initial_condition.generate(grid)
    assert np.all(initial_condition == 5.0)
    assert initial_condition.shape == (64, 64)

def test_constant_initial_condition_call(constant_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition = constant_initial_condition(grid)
    assert np.all(initial_condition == 5.0)
    assert initial_condition.shape == (64, 64)

def test_constant_initial_condition_generate_reproducibility(constant_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition_1 = constant_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = constant_initial_condition.generate(grid, random_state=42)
    assert np.array_equal(initial_condition_1, initial_condition_2)

def test_constant_initial_condition_call_reproducibility(constant_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition_1 = constant_initial_condition(grid, random_state=42)
    initial_condition_2 = constant_initial_condition(grid, random_state=42)
    assert np.array_equal(initial_condition_1, initial_condition_2)

def test_constant_initial_condition_generate_different_seeds(constant_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition_1 = constant_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = constant_initial_condition.generate(grid, random_state=43)
    assert not np.array_equal(initial_condition_1, initial_condition_2)
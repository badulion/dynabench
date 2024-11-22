import numpy as np
from unittest.mock import MagicMock

def test_random_uniform_initial_condition_init(random_uniform_initial_condition):
    assert random_uniform_initial_condition.low == 0.0
    assert random_uniform_initial_condition.high == 1.0
    assert random_uniform_initial_condition.parameters == {"param1": 1.0}
    assert random_uniform_initial_condition.spatial_dim == 2

def test_random_uniform_initial_condition_str(random_uniform_initial_condition):
    assert str(random_uniform_initial_condition) == "I(x, y) ~ U(0.0, 1.0)"

def test_random_uniform_initial_condition_generate(random_uniform_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition = random_uniform_initial_condition.generate(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= 0.0)
    assert np.all(initial_condition < 1.0)

def test_random_uniform_initial_condition_generate_reproducibility(random_uniform_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition_1 = random_uniform_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = random_uniform_initial_condition.generate(grid, random_state=42)
    assert np.array_equal(initial_condition_1, initial_condition_2)

def test_random_uniform_initial_condition_call(random_uniform_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition = random_uniform_initial_condition(grid, random_state=42)
    assert initial_condition.shape == (64, 64)
    assert np.all(initial_condition >= 0.0)
    assert np.all(initial_condition < 1.0)

def test_random_uniform_initial_condition_generate_different_seeds(random_uniform_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    initial_condition_1 = random_uniform_initial_condition.generate(grid, random_state=42)
    initial_condition_2 = random_uniform_initial_condition.generate(grid, random_state=43)
    assert not np.array_equal(initial_condition_1, initial_condition_2)
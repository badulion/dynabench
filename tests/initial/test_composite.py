import numpy as np
from unittest.mock import MagicMock


def test_composite_initial_condition_init(composite_initial_condition, constant_initial_condition, random_uniform_initial_condition):
    assert composite_initial_condition.components == (constant_initial_condition, random_uniform_initial_condition)

def test_composite_initial_condition_num_variables(composite_initial_condition):
    assert composite_initial_condition.num_variables == 2

def test_composite_initial_condition_generate(composite_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    
    constant_initial_condition = composite_initial_condition.components[0]
    random_uniform_initial_condition = composite_initial_condition.components[1]
    
    constant_initial_condition.generate = MagicMock(return_value=np.full(grid.shape, 5.0))
    random_uniform_initial_condition.generate = MagicMock(return_value=np.random.uniform(0.0, 1.0, grid.shape))
    
    initial_conditions = composite_initial_condition.generate(grid, random_state=42)
    
    assert len(initial_conditions) == 2
    assert np.all(initial_conditions[0] == 5.0)
    assert initial_conditions[1].shape == (64, 64)
    assert np.all(initial_conditions[1] >= 0.0)
    assert np.all(initial_conditions[1] < 1.0)

def test_composite_initial_condition_generate_reproducibility(composite_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    
    initial_conditions_1 = composite_initial_condition.generate(grid, random_state=42)
    initial_conditions_2 = composite_initial_condition.generate(grid, random_state=42)
    
    for ic1, ic2 in zip(initial_conditions_1, initial_conditions_2):
        assert np.array_equal(ic1, ic2)

def test_composite_initial_condition_call(composite_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    
    constant_initial_condition = composite_initial_condition.components[0]
    random_uniform_initial_condition = composite_initial_condition.components[1]
    
    constant_initial_condition.generate = MagicMock(return_value=np.full(grid.shape, 5.0))
    random_uniform_initial_condition.generate = MagicMock(return_value=np.random.uniform(0.0, 1.0, grid.shape))
    
    initial_conditions = composite_initial_condition(grid, random_state=42)
    
    assert len(initial_conditions) == 2
    assert np.all(initial_conditions[0] == 5.0)
    assert initial_conditions[1].shape == (64, 64)
    assert np.all(initial_conditions[1] >= 0.0)
    assert np.all(initial_conditions[1] < 1.0)

def test_composite_initial_condition_generate_different_seeds(composite_initial_condition):
    grid = MagicMock()
    grid.shape = (64, 64)
    
    initial_conditions_1 = composite_initial_condition.generate(grid, random_state=42)
    initial_conditions_2 = composite_initial_condition.generate(grid, random_state=43)
    
    for ic1, ic2 in zip(initial_conditions_1, initial_conditions_2):
        assert not np.array_equal(ic1, ic2)

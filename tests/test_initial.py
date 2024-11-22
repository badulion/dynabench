import pytest
from unittest.mock import MagicMock

def test_initial_condition_init(base_initial_condition):
    assert base_initial_condition.parameters == {"param1": 1.0}
    assert base_initial_condition.spatial_dim == 2

def test_initial_condition_str(base_initial_condition):
    assert str(base_initial_condition) == "Initial condition base class"

def test_initial_condition_num_variables(base_initial_condition):
    assert base_initial_condition.num_variables == 1

def test_initial_condition_generate_not_implemented(base_initial_condition):
    grid = MagicMock()
    with pytest.raises(NotImplementedError):
        base_initial_condition.generate(grid)

def test_initial_condition_call(base_initial_condition):
    grid = MagicMock()
    with pytest.raises(NotImplementedError):
        base_initial_condition(grid)
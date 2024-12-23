
import pytest
from dynabench.solver._base import BaseSolver
import numpy as np

from unittest.mock import MagicMock
from dynabench.solver import PyPDESolver
from dynabench.grid import Grid

class PickableMagickMock(MagicMock):
    def __reduce__(self):
        return (MagicMock, ())

@pytest.fixture
def base_solver(default_base_equation, default_grid, base_initial_condition):
    return BaseSolver(
        equation=default_base_equation,
        grid=default_grid,
        initial_generator=base_initial_condition,
        parameters={"param1": "value1"}
    )
@pytest.fixture
def mock_grid():
    grid = PickableMagickMock(spec=Grid)
    grid.export_as_pypde_grid.return_value = PickableMagickMock()
    grid.shape = (64, 64)
    grid.x = np.linspace(0, 1, 64)
    grid.y = np.linspace(0, 1, 64)
    return grid

@pytest.fixture
def mock_initial_condition():
    mock_ic = PickableMagickMock()
    mock_ic.generate.return_value = np.zeros((64, 64))
    mock_ic.num_variables = 1
    return mock_ic

@pytest.fixture
def pypde_solver(default_base_equation, default_grid, mock_initial_condition):
    return PyPDESolver(
        equation=default_base_equation,
        grid=default_grid,
        initial_generator=mock_initial_condition,
        parameters={'method': 'RK45'}
    )
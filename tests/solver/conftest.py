
import pytest
from dynabench.solver._base import BaseSolver

@pytest.fixture
def base_solver(default_base_equation, default_grid, base_initial_condition):
    return BaseSolver(
        equation=default_base_equation,
        grid=default_grid,
        initial_generator=base_initial_condition,
        parameters={"param1": "value1"}
    )
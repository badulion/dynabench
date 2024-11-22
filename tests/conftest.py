import pytest
from dynabench.grid import Grid, UnitGrid
from dynabench.equation import BaseEquation
from dynabench.initial import InitialCondition

@pytest.fixture
def default_grid():
    return Grid()

@pytest.fixture
def custom_grid():
    return Grid(grid_size=(128, 128), grid_limits=((0, 2), (0, 2)))

@pytest.fixture
def unit_grid():
    return UnitGrid(grid_size=(128, 128))

@pytest.fixture
def custom_grid_unit():
    return Grid(grid_size=(128, 128), grid_limits=((0, 128), (0, 128)))

@pytest.fixture
def default_base_equation():
    return BaseEquation()

@pytest.fixture
def custom_base_equation():
    return BaseEquation(equations=["dt(u) = -u"], parameters={"param1": 1.0})

@pytest.fixture
def base_equation_two_variables():
    return BaseEquation(equations=["dt(u) = -u", "dt(v) = -v"])

# make a very long equation with multiple variables and high order derivatives
@pytest.fixture
def base_equation_complicated():
    return BaseEquation(equations=["dt(u) = -u", "dt(v) = -v", "dtt(w) = -w"])

@pytest.fixture
def base_initial_condition():
    return InitialCondition(parameters={"param1": 1.0})
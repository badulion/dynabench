import pytest
import numpy as np

def test_init_default_grid(default_grid):
    assert default_grid.grid_size == (64, 64)
    assert default_grid.grid_limits == ((0, 64), (0, 64))
    assert len(default_grid.x) == 64
    assert len(default_grid.y) == 64

def test_init_custom_grid(custom_grid):
    assert custom_grid.grid_size == (128, 128)
    assert custom_grid.grid_limits == ((0, 2), (0, 2))
    assert len(custom_grid.x) == 128
    assert len(custom_grid.y) == 128

def test_str(default_grid):
    assert str(default_grid) == "Grid of size (64, 64) and limits ((0, 64), (0, 64))."

def test_shape(default_grid):
    assert default_grid.shape == (64, 64)

def test_dx(default_grid):
    assert default_grid.dx == pytest.approx(1.0)

def test_dy(default_grid):
    assert default_grid.dy == pytest.approx(1.0)

def test_get_meshgrid(default_grid):
    meshgrid = default_grid.get_meshgrid()
    assert len(meshgrid) == 2
    assert meshgrid[0].shape == (64, 64)
    assert meshgrid[1].shape == (64, 64)

def test_get_random_point(default_grid):
    point = default_grid.get_random_point()
    assert point[0] in default_grid.x
    assert point[1] in default_grid.y

def test_get_random_point_within_domain(default_grid):
    point = default_grid.get_random_point_within_domain()
    assert default_grid.grid_limits[0][0] <= point[0] <= default_grid.grid_limits[0][1]
    assert default_grid.grid_limits[1][0] <= point[1] <= default_grid.grid_limits[1][1]

def test_unit_grid_vs_custom_grid(unit_grid, custom_grid_unit):
    assert unit_grid.grid_size == custom_grid_unit.grid_size
    assert unit_grid.grid_limits == custom_grid_unit.grid_limits
    assert np.array_equal(unit_grid.x, custom_grid_unit.x)
    assert np.array_equal(unit_grid.y, custom_grid_unit.y)
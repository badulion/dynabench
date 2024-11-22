import numpy as np
from unittest.mock import MagicMock, patch

def test_pypde_solver_init(pypde_solver):
    assert pypde_solver.parameters == {"param1": 1.0}
    assert pypde_solver.spatial_dim == 2
    assert pypde_solver.equation is not None

def test_pypde_solver_file_exists(pypde_solver, tmp_path):
    equation_filename = pypde_solver.generate_filename(t_span=[0, 0.2], dt_eval=0.1, seed=42)
    filename = tmp_path / equation_filename
    pypde_solver.solve(t_span=[0, 0.2], dt_eval=0.1, random_state=42, out_dir=tmp_path)
    assert filename.exists()

def test_pypde_solver_generate_filename(pypde_solver):
    filename = pypde_solver.generate_filename(t_span=[0, 10], dt_eval=0.1, seed=42)
    assert filename.startswith("base_")
    assert filename.endswith("_dt_0.1_trange_0_10_seed_42.h5")

def test_pypde_solver_initial_condition_generation(pypde_solver, mock_initial_condition, mock_grid):
    initial_condition = pypde_solver.initial_generator.generate(mock_grid, random_state=42)
    assert np.all(initial_condition == 0.0)

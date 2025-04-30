import numpy as np
from unittest.mock import MagicMock, patch

def test_pypde_solver_init(pypde_solver):
    assert pypde_solver.parameters == {'method': 'RK45'}
    assert pypde_solver.spatial_dim == 2
    assert pypde_solver.equation is not None

def test_pypde_solver_file_exists(pypde_solver, tmp_path):
    equation_path = pypde_solver.generate_filename(t_span=[0, 0.2], dt_eval=0.1, random_state=42, out_dir=tmp_path)
    pypde_solver.solve(t_span=[0, 0.2], dt_eval=0.1, random_state=42, out_dir=tmp_path)
    assert equation_path.exists()

def test_generate_filename(pypde_solver):
    eq_descriptor, solver_descriptor, seed_descriptor = pypde_solver.generate_descriptors(
        t_span=[0, 10],
        dt_eval=0.1,
        random_state=42,
        hash_truncate=8
    )
    assert eq_descriptor.startswith("base_")
    assert solver_descriptor == ("dt_0.1_trange_0_10")
    assert seed_descriptor == "seed_42"

def test_pypde_solver_initial_condition_generation(pypde_solver, mock_initial_condition, mock_grid):
    initial_condition = pypde_solver.initial_generator.generate(mock_grid, random_state=42)
    assert np.all(initial_condition == 0.0)

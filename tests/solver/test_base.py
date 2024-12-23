import pytest

def test_init(base_solver, default_base_equation, default_grid, base_initial_condition):
    assert base_solver.equation == default_base_equation
    assert base_solver.grid == default_grid
    assert base_solver.initial_generator == base_initial_condition
    assert base_solver.spatial_dim == 2
    assert base_solver.parameters == {"param1": "value1"}

def test_str(base_solver):
    assert str(base_solver) == "Base Equation Solver"

def test_generate_filename(base_solver):
    filename = base_solver.generate_filename(
        t_span=[0, 10],
        dt_eval=0.1,
        random_state=42,
        hash_truncate=8
    )
    assert filename.startswith("base_")
    assert filename.endswith("_dt_0.1_trange_0_10_seed_42.h5")

def test_generate_filename_different_seeds(base_solver):
    filename1 = base_solver.generate_filename(
        t_span=[0, 100],
        dt_eval=0.1,
        random_state=42,
        hash_truncate=8
    )
    filename2 = base_solver.generate_filename(
        t_span=[0, 10],
        dt_eval=0.01,
        random_state=43,
        hash_truncate=8
    )
    assert filename1 != filename2
    # Check that the hash is the same
    assert filename1.split("_")[1] == filename2.split("_")[1]

def test_base_solver(base_solver):
    with pytest.raises(NotImplementedError):
        base_solver.solve(
            random_state=42,
            t_span=[0, 10],
            dt_eval=0.1,
            out_dir="data/raw"
        )
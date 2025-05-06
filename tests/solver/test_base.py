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
    eq_descriptor, solver_descriptor, seed_descriptor = base_solver.generate_descriptors(
        t_span=[0, 10],
        dt_eval=0.1,
        random_state=42,
        hash_truncate=8
    )
    assert eq_descriptor.startswith("base_")
    assert solver_descriptor == ("dt_0.1_trange_0_10")
    assert seed_descriptor == "seed_42"

def test_generate_filename_different_seeds(base_solver):
    eq_descriptor1, solver_descriptor1, seed_descriptor1 = base_solver.generate_descriptors(
        t_span=[0, 100],
        dt_eval=0.1,
        random_state=42,
        hash_truncate=8
    )
    eq_descriptor2, solver_descriptor2, seed_descriptor2 = base_solver.generate_descriptors(
        t_span=[0, 10],
        dt_eval=0.01,
        random_state=43,
        hash_truncate=8
    )
    assert eq_descriptor1 == eq_descriptor2
    assert solver_descriptor1 != solver_descriptor2
    
    # Check that the seed descriptors are different
    assert seed_descriptor1 != seed_descriptor2

def test_base_solver(base_solver):
    with pytest.raises(NotImplementedError):
        base_solver.solve(
            random_state=42,
            t_span=[0, 10],
            dt_eval=0.1,
            out_dir="data/raw"
        )
from dynabench.equation import CahnHilliardEquation
from dynabench.initial import RandomUniform
from dynabench.grid import Grid
from dynabench.solver import PyPDESolver



# Create an instance of the CahnHilliardEquation class with default parameters
pde_equation = CahnHilliardEquation()

# Create an instance of grid with default parameters
grid = Grid(grid_limits=((0, 64), (0, 64)), grid_size=(64, 64))

# generate an initial condition as a sum of 5 gaussians
intitial = RandomUniform()


# Solve the Cahn-Hilliard equation with the initial condition
solver = PyPDESolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'method': "RK23"})
solver.solve(t_span=[0, 100], dt_eval=1)
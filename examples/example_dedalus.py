from dynabench.equation import SimpleBurgersEquation
from dynabench.initial import RandomUniform
from dynabench.grid import UnitGrid
from dynabench.solver import DedalusSolver



# Create an instance of the CahnHilliardEquation class with default parameters
pde_equation = SimpleBurgersEquation()

# Create an instance of grid with default parameters
grid = UnitGrid(grid_size=(64, 64))

# generate an initial condition as a sum of 5 gaussians
intitial = RandomUniform()


# Solve the Cahn-Hilliard equation with the initial condition
solver = DedalusSolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'dt': 0.001})
solver.solve(t_span=[0, 100], dt_eval=1)
from dynabench.equation import SimpleBurgersEquation
from dynabench.initial import WrappedGaussians
from dynabench.grid import UnitGrid
from dynabench.solver import PyPDESolver



# Create an instance of the SimpleBurgersEquation class with default parameters
pde_equation = SimpleBurgersEquation()

# Create an instance of grid with default parameters
grid = UnitGrid(grid_size=(64, 64))

# generate an initial condition as a sum of 5 gaussians
intitial = WrappedGaussians(components=5)


# Solve the Burgers equation with the initial condition
solver = PyPDESolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'method': "RK23"})
solver.solve(t_span=[0, 100], dt_eval=1)
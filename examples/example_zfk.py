from dynabench.equation import ZFKEquation
from dynabench.initial import WrappedGaussians
from dynabench.grid import UnitGrid, Grid
from dynabench.solver import PyPDESolver



# Create an instance of the SimpleBurgersEquation class with default parameters
pde_equation = ZFKEquation(beta=10)

# Create an instance of grid with default parameters
grid = Grid(grid_size=(256, 256), grid_limits=((0,64), (0,64)))

# generate an initial condition as a sum of 5 gaussians
intitial = WrappedGaussians(components=5)


# Solve the Burgers equation with the initial condition
solver = PyPDESolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'method': "RK23"})
path = solver.solve(t_span=[0, 100], dt_eval=1)

from dynabench.utils.animation import animate_simulation
import h5py
import os
path = os.path.join("data/raw", path)

with h5py.File(path, 'r') as file:
    x = file['data'][:, 0]
    print(file.attrs.keys())
    print(file.attrs['equation'])

animate_simulation(x)
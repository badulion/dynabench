from dynabench.equation import CahnHilliardEquation, AdvectionEquation
from dynabench.initial import WrappedGaussians, RandomUniform
from dynabench.grid import Grid
from dynabench.solver import PyPDESolver

from pde import FieldCollection, ScalarField




# Create an instance of the CahnHilliardEquation class with default parameters
cahn_hilliard = CahnHilliardEquation(evolution_rate=0.001)

# Create an instance of grid with default parameters
grid = Grid()

# generate an initial condition as a sum of 5 gaussians
intitial = RandomUniform(random_state=42)


# Solve the Cahn-Hilliard equation with the initial condition
solver = PyPDESolver(equation=cahn_hilliard, grid=grid, initial_generator=intitial)
solver.solve()
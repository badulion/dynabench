from pde import FieldCollection, PDEBase, ScalarField, VectorField, CartesianGrid, CallbackTracker, ProgressTracker
from pde.trackers.base import TrackerCollection
from pde import FileStorage
from src.utils.initial import sum_of_gaussians_periodic as sum_of_gaussians
import matplotlib.pyplot as plt

from pde.visualization.plotting import ScalarFieldPlot
import numpy as np
import tqdm


class GasDynamics(PDEBase):
    """Gas Dynamics simulating weather"""

    def __init__(self, parameters=[0.01, 0.01, 1, 1], bc="auto_periodic_neumann"):
        super().__init__()
        self.parameters = parameters  # spatial mobility
        self.bc = bc  # boundary condition

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        # extract grid points
        shape = grid.shape
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = (grid.cell_coords[:,:,0]-x_bounds[0])/(x_bounds[1]-x_bounds[0])
        y = (grid.cell_coords[:,:,1]-y_bounds[0])/(y_bounds[1]-y_bounds[0])
        # initialize fields
        p = ScalarField(grid, sum_of_gaussians(x, y, components=5, zero_level=2), label="density")
        T = ScalarField(grid, sum_of_gaussians(x, y, components=5, zero_level=2), label="temperature")
        v = VectorField(grid, "zeros", label="velocity")
        return FieldCollection([p, T, v])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        p, T, v = state
        rhs = state.copy()
        mu, k, gamma, M = self.parameters
        rhs[0] = - v.dot(p.gradient(self.bc)) - p*v.divergence(self.bc)
        rhs[1] = - v.dot(T.gradient(self.bc)) - gamma*T*v.divergence(self.bc)+gamma*M*k/p * T.laplace(self.bc)
        rhs[2] = - v.dot(v.gradient(self.bc)) - (T*p/M).gradient(self.bc) + (mu/p) * v.gradient(self.bc).divergence(self.bc)
        return rhs


# seed numpy
np.random.seed(seed=42)

# parameters
GRID_SIZE = 64
N_high = 500
N_low = 100


# initialize state
grid = CartesianGrid([(0,64), (0, 64)], [GRID_SIZE, GRID_SIZE], periodic=[True, True])
eq = GasDynamics()
state = eq.get_initial_state(grid)
state_with_dyn = FieldCollection([*state, *eq.evolution_rate(state)])

# initialize graph points
indices_high = [(i // GRID_SIZE, i % GRID_SIZE) for i in np.random.choice(GRID_SIZE ** 2, size=N_high, replace=False)]
indices_low = [(i // GRID_SIZE, i % GRID_SIZE) for i in np.random.choice(GRID_SIZE ** 2, size=N_low, replace=False)]


# initialize storage
storage = FileStorage("data.hdf5")
storage.start_writing(state_with_dyn)
def save_state(state, time):

    state_with_dyn = FieldCollection([*state.copy(), *eq.evolution_rate(state)])
    storage.append(state_with_dyn)

tracker_callback = CallbackTracker(save_state, interval=.2)
tracker_progress = ProgressTracker()
sol = eq.solve(state, t_range=100, dt=1e-3, tracker=TrackerCollection([tracker_callback, tracker_progress]))

p = ScalarFieldPlot(state_with_dyn)
p.make_movie(storage, "m.mp4")

storage.close()
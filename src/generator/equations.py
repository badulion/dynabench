from pde import FieldCollection, PDEBase, ScalarField, VectorField
from src.utils.initial import sum_of_gaussians_periodic as sum_of_gaussians

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

class BrusselatorPDE(PDEBase):
    """Brusselator with diffusive mobility"""

    def __init__(self, a=0.1, b=1, diffusivity=[0.1, 0.01], bc="auto_periodic_neumann"):
        super().__init__()
        self.a = a
        self.b = b
        self.diffusivity = diffusivity  # spatial mobility
        self.bc = bc  # boundary condition

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        u = ScalarField(grid, self.a, label="Field $u$")
        v = self.b / self.a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
        return FieldCollection([u, v])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        u, v = state
        rhs = state.copy()
        d0, d1 = self.diffusivity
        rhs[0] = d0 * u.laplace(self.bc) + self.a - (self.b + 1) * u + u**2 * v
        rhs[1] = d1 * v.laplace(self.bc) + self.b * u - u**2 * v
        return rhs

class KuramotoSivashinskyPDE(PDEBase):
    """Implementation of the normalized Kuramoto–Sivashinsky equation"""

    def __init__(self, bc="auto_periodic_neumann"):
        super().__init__()
        self.bc = bc

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        u = ScalarField.random_uniform(grid)
        return FieldCollection([u])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        state, = state
        state_lap = state.laplace(bc=self.bc)
        state_lap2 = state_lap.laplace(bc=self.bc)
        state_grad_sq = state.gradient_squared(bc=self.bc)
        return FieldCollection([-state_grad_sq / 2 - state_lap - state_lap2])


class WavePDE(PDEBase):
    """Implementation of the normalized Kuramoto–Sivashinsky equation"""

    def __init__(self, prop_speed=1, bc="auto_periodic_neumann"):
        super().__init__()
        self.bc = bc
        self.prop_speed = prop_speed

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        # extract grid points
        shape = grid.shape
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = (grid.cell_coords[:,:,0]-x_bounds[0])/(x_bounds[1]-x_bounds[0])
        y = (grid.cell_coords[:,:,1]-y_bounds[0])/(y_bounds[1]-y_bounds[0])

        # initialize fields
        u = ScalarField(grid, sum_of_gaussians(x, y, components=5, zero_level=0), label="wave")
        u_t = ScalarField(grid, "zeros", label="first_derivative")
        return FieldCollection([u, u_t])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        u, u_t = state
        rhs = state.copy()
        rhs[0] = u_t
        rhs[1] = (self.prop_speed ** 2) * u.laplace(self.bc)
        return rhs
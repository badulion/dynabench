from pde import FieldCollection, PDEBase, ScalarField, VectorField
from ..fourier.initial import sum_of_gaussians as initial_function
from ..fourier.initial import random_normal, random_uniform

class GasDynamicsPDE(PDEBase):
    """Gas Dynamics simulating weather"""

    def __init__(self, parameters=[0.01, 0.01, 1, 1], rate=1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.parameters = parameters  # spatial mobility
        self.bc = bc  # boundary condition
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        # extract grid points
        shape = grid.shape
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = (grid.cell_coords[:,:,0]-x_bounds[0])/(x_bounds[1]-x_bounds[0])
        y = (grid.cell_coords[:,:,1]-y_bounds[0])/(y_bounds[1]-y_bounds[0])
        # initialize fields
        p = ScalarField(grid, initial_function(x, y, components=5, zero_level=2), label="density")
        T = ScalarField(grid, initial_function(x, y, components=5, zero_level=2), label="temperature")
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
        return self.rate*rhs

class BrusselatorPDE(PDEBase):
    """Brusselator with diffusive mobility"""

    def __init__(self, parameters=[1, 3, 1, 0.1], rate = 1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.a = parameters[0]
        self.b = parameters[1]
        self.diffusivity = parameters[2:]  # spatial mobility
        self.bc = bc  # boundary condition
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)

        u = ScalarField(grid, self.a, label="Field $u$")
        v = self.b / self.a + 0.1 * ScalarField(grid, random_normal(x, y), label="Field $v$")
        return FieldCollection([u, v])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        u, v = state
        rhs = state.copy()
        d0, d1 = self.diffusivity
        rhs[0] = d0 * u.laplace(self.bc) + self.a - (self.b + 1) * u + u**2 * v
        rhs[1] = d1 * v.laplace(self.bc) + self.b * u - u**2 * v
        return self.rate*rhs

class ReactionDiffusionPDE(PDEBase):
    """Brusselator with diffusive mobility"""

    def __init__(self, parameters=[0.001, 0.005, 0.005, 0.1, 0.1], rate = 1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.parameters = parameters
        self.bc = bc  # boundary condition
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)
        
        # initialize fields
        #u = ScalarField(grid, initial_function(x, y, components=5, zero_level=2), label="activator")
        #v = ScalarField(grid, initial_function(x, y, components=5, zero_level=2), label="inhibitor")
        u = 0.1 * ScalarField(grid, random_normal(x, y, smooth=True), label="activator")
        v = 0.1 * ScalarField(grid, random_normal(x, y, smooth=True), label="inhibitor")
        return FieldCollection([u, v])

    def evolution_rate(self, state, t=0):
        """pure python implementation of the PDE"""
        u, v = state
        rhs = state.copy()
        Du, Dv, k, au, av = self.parameters
        rhs[0] = Du * u.laplace(self.bc) + au*(u - u**3 - k - v)
        rhs[1] = Dv * v.laplace(self.bc) + av*(u - v)
        return self.rate*rhs

class KuramotoSivashinskyPDE(PDEBase):
    """Implementation of the normalized Kuramoto–Sivashinsky equation"""

    def __init__(self, parameters=[], rate = 1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.bc = bc
        self.rate = rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)
        
        u = ScalarField(grid, random_uniform(x, y), label="advection")
        return FieldCollection([u])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        state, = state
        state_lap = state.laplace(bc=self.bc)
        state_lap2 = state_lap.laplace(bc=self.bc)
        state_grad_sq = state.gradient_squared(bc=self.bc)
        #return FieldCollection([state_lap])
        return self.rate * FieldCollection([-state_grad_sq / 2 - state_lap - state_lap2])


class WavePDE(PDEBase):
    """Implementation of the Wave equation"""

    def __init__(self, parameters=[1], rate=1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.bc = bc
        self.prop_speed = parameters[0]
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)

        # initialize fields
        u = ScalarField(grid, initial_function(x, y, components=5, zero_level=0), label="wave")
        u_t = ScalarField(grid, "zeros", label="first_derivative")
        return FieldCollection([u, u_t])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        u, u_t = state
        rhs = state.copy()
        rhs[0] = u_t
        rhs[1] = (self.prop_speed ** 2) * u.laplace(self.bc)
        return self.rate*rhs

class AdvectionPDE(PDEBase):
    """Implementation of the Advection equation"""

    def __init__(self, parameters=[1, 1], rate=1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.bc = bc
        self.travel_speed_x = parameters[0]
        self.travel_speed_y = parameters[1]
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)

        # initialize fields
        u = ScalarField(grid, initial_function(x, y, components=5, zero_level=0), label="advection")
        return FieldCollection([u])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        u, = state
        rhs = state.copy()
        grd = - u.gradient(self.bc)
        rhs[0] = grd[0]*self.travel_speed_x + grd[1]*self.travel_speed_y
        return self.rate*rhs
    
class BurgersPDE(PDEBase):
    """Implementation of the Burgers equation"""

    def __init__(self, parameters=[1], rate=1, bc="auto_periodic_neumann", *args, **kwargs):
        super().__init__()
        self.bc = bc
        self.diffusivity = parameters[0]
        self.rate = rate # evolution rate

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        x, y = extract_coordinates_from_grid(grid)

        # initialize fields
        u = ScalarField(grid, initial_function(x, y, components=5, zero_level=0), label="velocity x")
        v = ScalarField(grid, initial_function(x, y, components=5, zero_level=0), label="velocity y")
        return FieldCollection([u, v])

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        u, v = state
        rhs = state.copy()
        grd_u = - u.gradient(self.bc)
        grd_v = - v.gradient(self.bc)
        rhs[0] = grd_u[0]*u + grd_u[1]*v + self.diffusivity*u.laplace(self.bc)
        rhs[1] = grd_v[0]*u + grd_v[1]*v + self.diffusivity*v.laplace(self.bc)
        return self.rate*rhs


def extract_coordinates_from_grid(grid):
    shape = grid.shape
    x_bounds = grid.axes_bounds[0]
    y_bounds = grid.axes_bounds[1]
    x = (grid.cell_coords[:,:,0]-x_bounds[0])/(x_bounds[1]-x_bounds[0])
    y = (grid.cell_coords[:,:,1]-y_bounds[0])/(y_bounds[1]-y_bounds[0])
    return x, y

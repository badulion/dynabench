from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid, VectorField, CartesianGrid


class GasDynamics(PDEBase):
    """Gas Dynamics simulating weather"""

    def __init__(self, a=1, b=3, diffusivity=[1, 0.1], bc="auto_periodic_neumann"):
        super().__init__()
        self.a = a
        self.b = b
        self.diffusivity = diffusivity  # spatial mobility
        self.bc = bc  # boundary condition

    def get_initial_state(self, grid):
        """prepare a useful initial state"""
        u = ScalarField(grid, self.a, label="Field $u$")
        T = ScalarField()
        v = VectorField
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



# initialize state
grid = CartesianGrid([(0,64), (0, 64)], [128, 128])
eq = BrusselatorPDE(diffusivity=[1, 0.1])
state = eq.get_initial_state(grid)

#print(eq.evolution_rate(state)[0].data)

# simulate the pde
tracker = PlotTracker(interval=1, plot_args={"vmin": 0, "vmax": 5})
sol = eq.solve(state, t_range=5, dt=1e-3, tracker=tracker)

sol.plot()
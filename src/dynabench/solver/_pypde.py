from ._base import BaseSolver

from dynabench.equation import BaseEquation
from dynabench.grid import Grid
from dynabench.initial import InitialCondition

from typing import List

from pde import FileStorage, ProgressTracker
from pde import ScalarField, FieldCollection

class PyPDESolver(BaseSolver):
    """
        Solver class for solving PDEs using the py-pde library.

        Parameters
        ----------
        equation : dynabench.equation.BaseEquation
            The equation to solve.
        grid : dynabench.grid.Grid
            The grid on which the equation is to be solved.
        initial_generator : dynabench.initial.InitialCondition
            The initial condition generator from which the initial condition is to be generated.

    """

    def __init__(self, 
                 equation: BaseEquation,
                 grid: Grid,
                 initial_generator: InitialCondition,
                 parameters: dict = {}, 
                 **kwargs):
        super().__init__(equation, grid, initial_generator, parameters, **kwargs)
        
    def solve(self, 
              random_state: int = 42,
              t_span: List[float] = [0, 1],
              t_eval: List[float] = None):
        """
            Solve the equation.

            Parameters
            ----------
            random_state : int, default 42
                The random state to use for the initial condition.
            t_span : List[float], default [0, 1]
                The time span for the solution.
            t_eval : List[float], default None
                The time points at which the solution is to be evaluated.

            Returns
            -------
            np.ndarray
                The solution of the equation.
        """
        
        pypde_eq = self.equation.export_as_pypde_equation()
        initial_condition = self.initial_generator.generate(self.grid, random_state=random_state)
        pypde_grid = self.grid.export_as_pypde_grid()

        # Create tracker and file storage
        storage = FileStorage("data.h5")

        # create initial py-pde field
        num_variables = self.initial_generator.num_variables
        if num_variables == 1:
            initial_field = ScalarField(pypde_grid, initial_condition)
        else:
            initial_field = FieldCollection([ScalarField(pypde_grid, ic) for ic in initial_condition])

        # Solve the equation
        sol = pypde_eq.solve(initial_field, t_range=10, tracker=["progress", storage.tracker(0.01)], solver="scipy", method="RK23")
        sol.plot()

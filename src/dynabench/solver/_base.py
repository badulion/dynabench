import dynabench.equation
import dynabench.initial
import dynabench.grid

from typing import List
from joblib import hash

class BaseSolver(object):
    """
        Base class for all solvers.

        Parameters
        ----------
        equation : dynabench.equation.BaseEquation
            The equation to solve.
        grid : dynabench.grid.Grid
            The grid on which the equation is to be solved.
        initial_generator : dynabench.initial.InitialCondition
            The initial condition generator from which the initial condition is to be generated.
        parameters : dict, default {}
            Dictionary of parameters for the solver.
    """
    
    def __init__(self, 
                 equation: dynabench.equation.BaseEquation,
                 grid: dynabench.grid.Grid,
                 initial_generator: dynabench.initial.InitialCondition,
                 parameters: dict = {}, 
                 **kwargs):
        self.equation = equation
        self.grid = grid
        self.initial_generator = initial_generator
        self.spatial_dim = 2
        self.parameters = parameters

    def __str__(self):
        return "Base Equation Solver"
    
    def generate_filename(self,
                          t_span: List[float],
                          dt_eval: float,
                          seed: int,
                          hash_truncate: int = 8):
        eq_params = (
            self.equation,
            self.grid,
            self.initial_generator
        )
        return f"{self.equation.name}_{hash(eq_params)[:hash_truncate]}_dt_{dt_eval}_trange_{t_span[0]}_{t_span[1]}_seed_{seed}.h5"
    
    def solve(self, 
              random_state: int = 42,
              t_span: List[float] = [0, 1],
              dt_eval: float = 0.1,
              out_dir: str = "data/raw"):
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
        raise NotImplementedError("The solve method must be implemented in the subclass.")
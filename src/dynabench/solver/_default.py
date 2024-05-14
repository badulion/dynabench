import scipy.integrate as spi
import dynabench.equation
import dynabench.initial
import dynabench.grid
from .diff import DifferentialOperator
import numpy as np
from tqdm import tqdm

from typing import List

class DefaultSolver(object):
    """
        Solver for the finite difference method.

        Parameters
        ----------
        parameters : dict, default {}
            Dictionary of parameters for the solver.
    """
    
    def __init__(self, 
                 grid: dynabench.grid.Grid,
                 parameters: dict = {}, 
                 **kwargs):
        self.grid = grid
        self.spatial_dim = 2
        self.parameters = parameters

    def __str__(self):
        return "Finite difference solver"

    
    def _make_equation_callable(self, equation: dynabench.equation.BaseEquation) -> callable:
        """
            Convert the RHS of the equation string to a callable function.

            Parameters
            ----------
            equation_rhs : str
                The string representation of the equation.

            Returns
            -------
            callable
                The function representing the equation.
        """
        if "acc" in self.parameters:
            acc = self.parameters["acc"]
        else:
            acc = 2
        diff_operator_names = equation.diff_operators
        diff_operator_filters = [DifferentialOperator(op, dx=self.grid.dx, dy=self.grid.dy, acc=acc) for op in diff_operator_names]
        diff_operators_dict = dict(zip(diff_operator_names, diff_operator_filters))

        def call_eq(*vars):
            if len(equation.variables) != len(vars):
                raise ValueError(f"Number of variables do not match. Expected {len(equation.variables)} got {len(vars)}.")
            return [eval(eq, {**dict(zip(equation.variables, vars)), **diff_operators_dict, **equation.parameters}) for eq in equation.rhs]

        return call_eq
    
    def solve(self, 
              equation: dynabench.equation.BaseEquation, 
              initial_condition: dynabench.initial.InitialCondition,
              t_span: List[float] = [0, 1],
              t_eval: List[float] = None):
        """
            Solve the equation.

            Parameters
            ----------
            equation : dynabench.equation.BaseEquation
                The equation to be solved.
            initial_condition : dynabench.initial.InitialCondition
                The initial condition for the equation.
            t_span : List[float], default [0, 1]
                The time span for the solution.
            t_eval : List[float], default None
                The time points at which the solution is to be evaluated.

            Returns
            -------
            np.ndarray
                The solution of the equation.
        """
        initial = np.stack(initial_condition(self.grid), axis=0).flatten()
        func = self._make_equation_callable(equation)
        num_variables = equation.num_variables

        def f(t, u, pbar, state):
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n
            u = u.reshape((num_variables, self.grid.shape[0], self.grid.shape[1]))
            vars = [u[i] for i in range(u.shape[0])]
            du = np.stack(func(*vars), axis=0).flatten()
            return du
        with tqdm(total=1000, unit="‰") as pbar:
            sol = spi.solve_ivp(f,
                                t_span, 
                                initial, 
                                t_eval=t_eval, 
                                args=[pbar, [t_span[0], (t_span[1]-t_span[0])/1000]],
                                **self.parameters
            )

        y = sol.y.reshape((num_variables, self.grid.shape[0], self.grid.shape[1], len(sol.t)))
        y = np.moveaxis(y, -1, 0)
        return y
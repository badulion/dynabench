from numpy import ndarray
from ._base import BaseSolver

from dynabench.equation import BaseEquation
from dynabench.grid import Grid
from dynabench.initial import InitialCondition

from typing import List

from pde import FieldBase, ScalarField, FieldCollection, FileStorage
import os
import h5py

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
        parameters : dict, default {}
            Dictionary of parameters for the solver. 
            See the documentation of `py-pde <https://py-pde.readthedocs.io/en/latest/packages/pde.solvers.scipy.html>`_ and scipy's `solve_ivp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_ for more information.
            

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
        
        save_path = self.generate_filename(t_span=t_span, dt_eval=dt_eval, seed=random_state)
        
        pypde_eq = self.equation.export_as_pypde_equation()
        initial_condition = self.initial_generator.generate(self.grid, random_state=random_state)
        pypde_grid = self.grid.export_as_pypde_grid()

        # Create tracker and file storage
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, save_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        storage = FileStorage(out_path, write_mode="truncate")

        # create initial py-pde field
        num_variables = self.initial_generator.num_variables
        if num_variables == 1:
            initial_field = FieldCollection([ScalarField(pypde_grid, initial_condition)])
        else:
            initial_field = FieldCollection([ScalarField(pypde_grid, ic) for ic in initial_condition])

        # Solve the equation
        sol = pypde_eq.solve(initial_field, t_range=t_span, tracker=["progress", storage.tracker(dt_eval)], solver="scipy", **self.parameters)
        
        # save additional information
        with h5py.File(out_path, "a") as f:
            f["x_coords"] = pypde_grid.axes_coords[0]
            f["y_coords"] = pypde_grid.axes_coords[1]
            f.attrs["variables"] = self.equation.variables
            f.attrs["equation"] = self.equation.equations
            f.attrs["parameter_names"] = list(self.equation.parameters.keys())
            f.attrs["parameter_values"] = list(self.equation.parameters.values())
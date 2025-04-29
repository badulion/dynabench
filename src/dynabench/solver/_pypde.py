from numpy import ndarray
from ._base import BaseSolver

from dynabench.equation import BaseEquation
from dynabench.grid import Grid
from dynabench.initial import InitialCondition

from typing import List

from pde import FieldBase, ScalarField, FieldCollection, FileStorage
import pathlib
import h5py
import numpy as np

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
        
    def solve_single(self, 
                     random_state: int = 42,
                     t_span: List[float] = [0, 1],
                     dt_eval: float = 0.1,
                     out_dir: str = "data/raw"):
        """
            Solve a single equation.

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
        out_dir = pathlib.Path(out_dir)
        eq_descriptor, solver_descriptor, seed_descriptor = self.generate_descriptors(t_span=t_span, dt_eval=dt_eval, random_state=random_state)
        eq_name = f"{seed_descriptor}.h5"
        eq_dir = out_dir / eq_descriptor / solver_descriptor
        save_path = out_dir / eq_descriptor / solver_descriptor / eq_name
        
        eq_dir.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            save_path.unlink()
            
        storage = FileStorage(save_path, write_mode="truncate")

        # create initial py-pde field
        num_variables = self.initial_generator.num_variables
        if num_variables == 1:
            initial_field = FieldCollection([ScalarField(pypde_grid, initial_condition)])
        else:
            initial_field = FieldCollection([ScalarField(pypde_grid, ic) for ic in initial_condition])

        # Solve the equation
        sol = pypde_eq.solve(initial_field, t_range=t_span, tracker=["progress", storage.tracker(dt_eval)], solver="scipy", **self.parameters)
        
        # save additional information
        with h5py.File(save_path, "a") as f:
            x_coords = pypde_grid.axes_coords[0]
            y_coords = pypde_grid.axes_coords[1]
            X, Y = np.meshgrid(x_coords, y_coords)
            points = np.stack([X, Y], axis=-1)
            
            f["x_coords"] = x_coords
            f["y_coords"] = y_coords
            f["points"] = points
            
            f.attrs["variables"] = self.equation.variables
            f.attrs["equation"] = self.equation.equations
            f.attrs["parameter_names"] = list(self.equation.parameters.keys())
            f.attrs["parameter_values"] = list(self.equation.parameters.values())

        return save_path
    
    def solve(self,
              t_span: List[float] = [0, 1],
              dt_eval: float = 0.1,
              random_state: int = 42,
              out_dir: str = "data/raw",
             ) -> List[pathlib.Path]:
        """
            Solve the equation for multiple random states.

            Parameters
            ----------
            t_span : List[float], default [0, 1]
                The time span for the solution.
            dt_eval : float, default 0.1
                The time points at which the solution is to be evaluated.
            random_state : int, default 42
                The random state to use for the initial condition.

            Returns
            -------
            List[pathlib.Path]
                The paths to the saved solutions.
        """
        if type(random_state) == int:
            random_state = [random_state]
        elif type(random_state) == list:
            random_state = random_state
        else:
            raise ValueError("random_state should be an int or a list of ints")
        
        file_paths = []  # Initialize a list to store file paths
        for seed in random_state:
            file_path = self.solve_single(random_state=seed, t_span=t_span, dt_eval=dt_eval, out_dir=out_dir)
            file_paths.append(file_path)  # Collect the returned file path
        return file_paths  # Return the list of file paths
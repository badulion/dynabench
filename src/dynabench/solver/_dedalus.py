import dedalus.public as d3
import numpy as np
from dynabench.grid import Grid
from dynabench.equation import BaseEquation
from dynabench.initial import InitialCondition
from dynabench.solver import BaseSolver
from typing import List, Tuple
from tqdm import tqdm
import h5py
import pathlib

class DedalusSolver(BaseSolver):
    """
    Solver class for solving PDEs using the Dedalus library. Needs to be installed separately.

    Parameters
    ----------
    equation : dynabench.equation.BaseEquation
        The equation to solve.
    grid : dynabench.grid.Grid
        The grid on which the equation is to be solved.
    initial_generator : dynabench.initial.InitialCondition
        The initial condition generator.
    parameters : dict, default {}
        Additional parameters for the solver.
    """

    def __init__(self, 
                 equation: BaseEquation,
                 grid: Grid,
                 initial_generator: InitialCondition,
                 parameters: dict = {}):
        self.equation = equation
        self.grid = grid
        self.initial_generator = initial_generator
        self.parameters = parameters

    def solve(self, 
              t_span: List[float] = [0, 1],
              dt_eval: float = 0.1,
              random_state: int = 42,
              out_dir: str = "data/raw") -> pathlib.Path:
        """
        Solve the equation using Dedalus and save the solution to an HDF5 file.

        Parameters
        ----------
        t_span : List[float], default [0, 1]
            The time span for the solution.
        dt_eval : float, default 0.1
            The time step for saving the solution.
        random_state : int, default 42
            The random state for the initial condition.
        out_dir : str, default "data/raw"
            Directory to save the solution.

        Returns
        -------
        pathlib.Path
            Path to the saved solution file.
        """
        # Build Dedalus grid
        coords = d3.CartesianCoordinates('x', 'y')
        dist = d3.Distributor(coords, dtype=np.float64)
        xbasis = d3.RealFourier(coords['x'], size=self.grid.grid_size[0], bounds=self.grid.grid_limits[0])
        ybasis = d3.RealFourier(coords['y'], size=self.grid.grid_size[1], bounds=self.grid.grid_limits[1])

        # Initialize fields
        fields = [dist.Field(name=var, bases=(xbasis, ybasis)) for var in self.equation.variables]
        initial_conditions = self.initial_generator.generate(self.grid, random_state=random_state)
        for field, ic in zip(fields, initial_conditions if isinstance(initial_conditions, list) else [initial_conditions]):
            field['g'] = ic

        # Build problem
        namespace = {
            'np': np,
            'dx': lambda A: d3.Differentiate(A, coords['x']),  # Substitution for dx
            'dy': lambda A: d3.Differentiate(A, coords['y']),  # Substitution for dy
            **{var: field for var, field in zip(self.equation.variables, fields)}
        }
        namespace.update(self.equation.parameters)  # Expose parameters locally
        problem = d3.IVP(fields, namespace=namespace)

        # Export Dedalus-compatible equations
        lhs_list, rhs_list = self.equation.export_as_dedalus_equation()

        for lhs, rhs in zip(lhs_list, rhs_list):
            problem.add_equation(f"{lhs} = {rhs}")

        # Build solver
        solver = problem.build_solver(d3.SBDF2)
        solver.stop_sim_time = t_span[1]

        # Storage
        u_list = [np.stack([field['g'].copy() for field in fields])]
        t_list = [solver.sim_time]

        # Create output directory and file path
        out_dir = pathlib.Path(out_dir)
        eq_descriptor, solver_descriptor, seed_descriptor = self.generate_descriptors(t_span=t_span, dt_eval=dt_eval, random_state=random_state)
        eq_name = f"{seed_descriptor}.h5"
        eq_dir = out_dir / eq_descriptor / solver_descriptor
        save_path = out_dir / eq_descriptor / solver_descriptor / eq_name
        
        eq_dir.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            save_path.unlink()

        # Time-stepping
        if self.parameters.get('dt') is not None:
            dt = self.parameters['dt']
        else:
            dt = 0.001
        with tqdm(total=t_span[1], desc="Solving PDE") as pbar:
            while solver.proceed:
                solver.step(dt)
                if solver.sim_time >= t_list[-1] + dt_eval:
                    u_list.append(np.stack([field['g'].copy() for field in fields]))
                    t_list.append(solver.sim_time)
                    pbar.update(dt_eval)

        # Save solution to HDF5
        with h5py.File(save_path, "w") as f:
            x_coords = self.grid.x
            y_coords = self.grid.y
            X, Y = np.meshgrid(x_coords, y_coords)
            points = np.stack([X, Y], axis=-1)
            
           
            f.create_dataset("data", data=np.array(u_list))
            f.create_dataset("time", data=np.array(t_list))
            f.create_dataset("x_coords", data=x_coords)
            f.create_dataset("y_coords", data=y_coords)
            f.create_dataset("points", data=points)

            f.attrs["variables"] = self.equation.variables
            f.attrs["equation"] = self.equation.equations
            f.attrs["parameter_names"] = list(self.equation.parameters.keys())
            f.attrs["parameter_values"] = list(self.equation.parameters.values())

        return save_path

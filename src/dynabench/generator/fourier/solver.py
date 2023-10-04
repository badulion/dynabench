import dedalus.public as d3
import numpy as np

from typing import Tuple
import tqdm

from dataclasses import dataclass


@dataclass
class DedalusGrid:
    coords: d3.CartesianCoordinates
    dist: d3.Distributor
    xbasis: d3.RealFourier
    ybasis: d3.RealFourier
    points: np.ndarray
    grid_size: Tuple[int]
    domain_bounds: Tuple[int]

    def build_grid(grid_size: Tuple[int] = (64,64),
                domain_bounds: Tuple[int] = (1,1),
                dealias: float = 3/2,
                dtype=np.float64):
        # Parameters
        Nx, Ny = grid_size
        Dx, Dy = domain_bounds

        # Initialize grid
        coords = d3.CartesianCoordinates('x', 'y')
        dist = d3.Distributor(coords, dtype=dtype)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Dx), dealias=dealias)
        ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Dy), dealias=dealias)

        # Points
        xgrid, ygrid = np.meshgrid(xbasis.global_grid(scale=1).ravel(), ybasis.global_grid(scale=1).ravel())
        points = np.stack([xgrid, ygrid]).transpose(1,2,0)
        points = points/np.array(domain_bounds).reshape(1,1,2)

        # store as dataclass
        return DedalusGrid(coords,dist,xbasis,ybasis,points,grid_size,domain_bounds)

def run_dedalus_solver(grid: DedalusGrid,
                       problem: d3.InitialValueProblem,
                       timestepper: d3.MultistepIMEX = d3.SBDF4,
                       step_size: float = 1e-2,
                       save_dt: float = 1,
                       t_max: float = 200):


    # Build solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = t_max



    # Setup storage
    for var in problem.variables:
        var.change_scales(1)

    u_initial = np.stack([var['g'] for var in problem.variables])
    u_list = [u_initial]
    t_list = [solver.sim_time]

    # Main loop
    bar_format = "{l_bar}{bar}| {n:.02f}/{total:.02f}"
    prog_bar = tqdm.tqdm(total=t_max, bar_format=bar_format)
    next_save_at = save_dt
    while solver.proceed:
        solver.step(step_size)
        prog_bar.update(step_size)
        if solver.sim_time > next_save_at:
            for var in problem.variables:
                var.change_scales(1)
            u_current = np.stack([var['g'] for var in problem.variables])
            u_list.append(u_current)
            t_list.append(solver.sim_time)
            next_save_at += save_dt

    # Convert storage lists to arrays
    u_array = np.array(u_list)
    t_array = np.array(t_list)

    return u_array, t_array, grid.points


if __name__ == "__main__":
    pass
from .equations import *
from .solver import run_dedalus_solver, DedalusGrid

from typing import Dict, Tuple, Optional

def solve_equation(problem_description, random_state=None):

    grid = DedalusGrid.build_grid(problem_description.grid_size, 
                                  problem_description.domain_bounds)
    problem = build_problem(grid, 
                            problem_description.equation, 
                            problem_description.equation_params, 
                            random_state=random_state)

    return run_dedalus_solver(grid, problem, 
                              step_size=problem_description.step_size, 
                              save_dt=problem_description.save_dt, 
                              t_max=problem_description.t_max)
    


def build_problem(grid, equation: str, params: Optional[dict] = None, random_state=None):
    if params is None:
        params = {}
    if equation == "advection":
        return build_problem_advection(grid, **params, random_state=random_state)
    elif equation == "burgers":
        return build_problem_burgers(grid, **params, random_state=random_state)
    elif equation == "gasdynamics":
        return build_problem_gasdynamics(grid, **params, random_state=random_state)
    elif equation == "kuramotosivashinsky":
        return build_problem_kuramotosivashinsky(grid, **params, random_state=random_state)
    elif equation == "reactiondiffusion":
        return build_problem_reactiondiffusion(grid, **params, random_state=random_state)
    elif equation == "wave":
        return build_problem_wave(grid, **params, random_state=random_state)
    else:
        raise KeyError(f"No such equation: {equation}")
    

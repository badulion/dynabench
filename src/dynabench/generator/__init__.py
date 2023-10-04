from dataclasses import dataclass
from typing import Tuple, Optional
from dynabench.generator.fourier import solve_equation

@dataclass
class ProblemDescription:
    equation: str
    equation_params: Optional[dict] = None
    grid_size: Tuple[int] = (64,64)
    domain_bounds: Tuple[int] = (64,64)
    step_size: float = 1e-2
    save_dt: float = 1
    t_max: float = 200

    def solve(self, backend="dedalus", random_state: int = None):
        return solve_equation(self, random_state=random_state)
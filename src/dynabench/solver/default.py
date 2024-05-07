import scipy.integrate as spi
import dynabench.equation
from .diff import DifferentialOperator

class FinDiffSolver(object):
    """
        Solver for the finite difference method.

        Parameters
        ----------
        spatial_dim : int, default 2
            Spatial dimension of the solver. Defaults to 2.
        parameters : dict, default {}
            Dictionary of parameters for the solver.

        Methods
        -------
        solve(*args, **kwargs)
            Solve the equation.
    """
    
    def __init__(self, 
                 spatial_dim: int = 2, 
                 parameters: dict = {}, 
                 **kwargs):
        self.spatial_dim = spatial_dim
        self.parameters = parameters
        self._variables = ["u"]

    def __str__(self):
        return "Finite difference solver"
    
    @property
    def variables(self):
        """
            Get the number of variables of the solver.
        """
        return len(self._variables)
    
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
        diff_operator_names = equation.diff_operators
        diff_operator_filters = [DifferentialOperator(op) for op in diff_operator_names]
        print(diff_operator_names, diff_operator_filters)
        diff_operators_dict = dict(zip(diff_operator_names, diff_operator_filters))

        equation = "dxx(u) + dyy(u) + dxy(u**3) + dx(u) + dy(u) + u"
        return lambda u, t: eval(equation, {'u': u, **diff_operators_dict})
    
    def solve(self, *args, **kwargs):
        """
            Solve the equation.

            Returns
            -------
            np.ndarray
                The solution of the equation.
        """
        def f(u, t):
            return 0


class FinDiffSolver(object):
    """
        Solver for the finite difference method.

        Parameters
        ----------
        spatial_dim : int, default 2
            Spatial dimension of the solver. Defaults to 2.
        parameters : dict, default {}
            Dictionary of parameters for the solver.
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
    
    def solve(self, *args, **kwargs):
        """
            Solve the equation.

            Returns
            -------
            np.ndarray
                The solution of the equation.
        """
        raise NotImplementedError("The solve method must be implemented in the subclass.")	
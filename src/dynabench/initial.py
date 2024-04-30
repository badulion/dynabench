"""
    Module containing different initial condition generators.
"""

import numpy as np



class InitialCondition(object):
    """
        Base class for all initial conditions.

        Parameters
        ----------
        spatial_dim : int, default 2
            Spatial dimension of the initial condition. Defaults to 2.
        parameters : dict, default {}
            Dictionary of parameters for the initial condition.
    """
    
    def __init__(self, 
                 spatial_dim: int = 2, 
                 parameters: dict = {}, 
                 **kwargs):
        self.spatial_dim = spatial_dim
        self.parameters = parameters
        self._variables = ["u"]

    def __str__(self):
        return "u(x, y) = 0"
    
    @property
    def variables(self):
        """
            Get the number of variables of the initial condition.
        """
        return len(self._variables)
    
    def generate(self, *args, **kwargs):
        """
            Generate the initial condition.

            Returns
            -------
            np.ndarray
                The initial condition.
        """
        raise NotImplementedError("The generate method must be implemented in the subclass.")
    
    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
from . import BaseEquation

from typing import List

class AdvectionEquation(BaseEquation):
    """
        Advection equation of the form :math:`u_t + c_x * u_x + c_y * u_y = 0`. 
        Where :math:`c_x` and :math:`c_y` are the velocities in the x and y directions respectively.
        
        Parameters
        ----------
        spatial_dim : int, default 2
            Spatial dimension of the equation. Defaults to 2.
        parameters : dict, default {'velocity_x': '1.0', 'velocity_y': '1.0'}
            Dictionary of parameters for the equation.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.
    """

    def __init__(self, 
                 spatial_dim: int = 2, 
                 parameters: dict = {'velocity_x': 1.0, 'velocity_y': 1.0}, 
                 **kwargs):
        super(AdvectionEquation, self).__init__(spatial_dim, parameters, **kwargs)
        self.linear_terms = [f"{parameters['velocity_x']}*u_x", f"{parameters['velocity_y']}*u_y"]
        self.nonlinear_terms = []
    
    def __str__(self):
        return "u_t + u_x = 0"
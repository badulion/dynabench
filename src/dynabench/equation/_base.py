import numpy as np

from typing import List

class Term(object):
    """
        Represents a term in an equation.

        Parameters
        ----------
        term : str
            The term to be represented.
        linear : bool, default True
            Whether the term is linear. Defaults to True.
    """
    def __init__(self, term: str, linear: bool = True):
        self.term = term
        self.linear = linear

    def __str__(self):
        return self.term

class BaseEquation(object):
    """
        Base class for all equations.

        Parameters
        ----------
        spatial_dim : int, default 2
            Spatial dimension of the equation. Defaults to 2.
        parameters : dict, default {}
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
                 parameters: dict = {}, 
                 **kwargs):
        self.spatial_dim = spatial_dim
        self.parameters = parameters
        self._variables = ["u"]

    def __str__(self):
        return "u_t = 0"
    
    @property
    def variables(self):
        """
            Get the number of variables of the equation.
        """
        return len(self._variables)

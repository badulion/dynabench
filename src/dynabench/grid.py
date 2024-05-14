"""
    Module for representing a grid for finite difference methods.
"""


import numpy as np

class Grid(object):
    """
        Base class for all grids.

        Parameters
        ----------
        grid_size : tuple, default (64, 64)
            Size of the grid. Defaults to (64, 64).
        grid_limits : tuple, default ((0, 1), (0, 1))
            Limits of the grid. Defaults to ((0, 1), (0, 1)).
    """ 
    def __init__(self, 
                 grid_size: tuple = (64, 64), 
                 grid_limits: tuple = ((0, 1), (0, 1)),
                 **kwargs):
        self.grid_size = grid_size
        self.grid_limits = grid_limits
        self.x = np.linspace(grid_limits[0][0], grid_limits[0][1], grid_size[0], endpoint=False)
        self.y = np.linspace(grid_limits[1][0], grid_limits[1][1], grid_size[1], endpoint=False)


    def __str__(self):
        return f"Grid of size {self.grid_size} and limits {self.grid_limits}."
    
    @property
    def shape(self):
        """
            Get the shape of the grid.
        """
        return self.grid_size
    
    @property
    def dx(self):
        """
            Get the step size in the x direction.
        """
        return self.x[1] - self.x[0]
    
    @property
    def dy(self):
        """
            Get the step size in the y direction.
        """
        return self.y[1] - self.y[0]
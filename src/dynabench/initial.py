"""
    Module containing different initial condition generators.
"""

import numpy as np
import dynabench.grid

from typing import List




class InitialCondition(object):
    """
        Base class for all initial conditions.

        Parameters
        ----------
        parameters : dict, default {}
            Dictionary of parameters for the initial condition.
    """
    
    def __init__(self, 
                 parameters: dict = {}, 
                 **kwargs):
        self.spatial_dim = 2
        self.parameters = parameters

    def __str__(self):
        return "Initial condition base class"

    def generate(self, grid: dynabench.grid.Grid):
        """
            Generate the initial condition.

            Parameters
            ----------
            grid : dynabench.grid.Grid
                The grid on which the initial condition is to be generated.

            Returns
            -------
            np.ndarray
                The initial condition.
        """
        raise NotImplementedError("The generate method must be implemented in the subclass.")
    
    def __call__(self, grid: dynabench.grid.Grid):
        return self.generate(grid)
    
class Composite(InitialCondition):
    """
        Composite initial condition generator consisting of multiple initial conditions for the same grid.
        Convenience class to generate multiple initial conditions for different variables.

        Parameters
        ----------
        components : list
            List of single initial conditions.
    """
    
    def __init__(self, *components: list):
        self.components = components
        
    def generate(self, grid: dynabench.grid.Grid):
        return [component(grid) for component in self.components]
    
class Constant(InitialCondition):
    """
        Initial condition with a constant value.

        Parameters
        ----------
        value : float, default 0.0
            The value of the constant.
    """
    
    def __init__(self, value: float = 0.0, **kwargs):
        super(Constant, self).__init__(**kwargs)
        self.value = value
        
    def __str__(self):
        return f"I(x, y) = {self.value}"
    
    def generate(self, grid: dynabench.grid.Grid):
        return self.value+np.zeros(grid.shape)
    

class SumOfGaussians(InitialCondition):
    """
        Initial condition generator for the sum of gaussians.

        Parameters
        ----------
        grid_size : tuple, default (64, 64)
            The size of the grid.
        components : int, default 1
            The number of gaussian components.
        zero_level : float, default 0.0
            The zero level of the initial condition.
        random_state : int, default None
            The random state.
    """
    
    def __init__(self, 
                 components: int = 1, 
                 zero_level: float = 0.0, 
                 random_state: int = None, 
                 **kwargs):
        super(SumOfGaussians, self).__init__(**kwargs)
        self.components = components
        self.zero_level = zero_level
        self.random_state = random_state
        
    def __str__(self):
        return "I(x, y) = sum_i A_i exp(-40(x-x_i)^2 + (y-y_i)^2)"
    
    def generate(self, grid: dynabench.grid.Grid):
        np.random.seed(self.random_state)
        x, y = np.meshgrid(grid.x, grid.y)

        mx = [np.random.choice(grid.shape[0]) for i in range(self.components)]
        my = [np.random.choice(grid.shape[1]) for i in range(self.components)]

        squared_distance_to_center = (x-0.5)**2 + (y-0.5)**2
        gaussian = np.exp(-40*squared_distance_to_center)

        u = self.zero_level+np.zeros_like(x)
        for i in range(self.components):

            component = np.roll(gaussian, (mx[i],my[i]), axis=(0,1))

            u = u + np.random.uniform(-1, 1) * component

        return u
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dynabench.solver.diff import differential_operator
    from scipy.signal import convolve2d

    initial_condition = SumOfGaussians((512,512), components=5, random_state=42)
    u = initial_condition.generate()
    d_u = differential_operator("u_x_1_y_0", dx=1/512, dy=1/512)
    print(d_u)

    for _ in range(10):
        u = convolve2d(u, d_u, mode="same", boundary="wrap")
    plt.imshow(u)
    plt.savefig("initial_condition.png")
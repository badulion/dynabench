"""
    Module containing different initial condition generators.
"""

import numpy as np
import dynabench.grid

from typing import List

from itertools import product




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
    
    @property
    def num_variables(self):
        """
            Get the number of variables.
        """
        return 1

    def generate(self, grid: dynabench.grid.Grid, random_state: int = 42):
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

    @property
    def num_variables(self):
        """
            Get the number of variables.
        """
        return len(self.components)
        
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
    

class RandomUniform(InitialCondition):
    """
        Initial condition with random values drawn from a uniform distribution.

        Parameters
        ----------
        low : float, default 0.0
            The lower bound of the uniform distribution.
        high : float, default 1.0
            The upper bound of the uniform distribution.
    """
    
    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        super(RandomUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high
        
    def __str__(self):
        return f"I(x, y) ~ U({self.low}, {self.high})"
    
    def generate(self, grid: dynabench.grid.Grid, random_state: int = 42):
        np.random.seed(random_state)
        return np.random.uniform(self.low, self.high, grid.shape)
    

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
    """
    
    def __init__(self, 
                 components: int = 1, 
                 zero_level: float = 0.0, 
                 **kwargs):
        super(SumOfGaussians, self).__init__(**kwargs)
        self.components = components
        self.zero_level = zero_level
        
    def __str__(self):
        return "I(x, y) = sum_i A_i exp(-40(x-x_i)^2 + (y-y_i)^2)"
    
    def generate(self, grid: dynabench.grid.Grid, random_state: int = 42):
        np.random.seed(random_state)
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
    
class WrappedGaussians(InitialCondition):
    """
        Initial condition generator for the sum of wrapped gaussians.

        Parameters
        ----------
        components : int, default 1
            The number of gaussian components.
        zero_level : float, default 0.0
            The zero level of the initial condition.
        periodic_levels : int or list, default 10
            The number of periodic levels to calculate the wrapped distribution. :math:`p_w(\theta)=\sum_{k=-\infty}^\infty {p(\theta+2\pi k)}`
    """
    
    def __init__(self, 
                 components: int = 1, 
                 zero_level: float = 0.0, 
                 periodic_levels: int = 10,
                 **kwargs):
        super(WrappedGaussians, self).__init__(**kwargs)
        self.components = components
        self.zero_level = zero_level
        self.periodic_levels = periodic_levels
        
    def __str__(self):
        return "I(x, y) = sum_i A_i exp(-40(x-x_i)^2 + (y-y_i)^2)"
    
    def _wrapped_gaussian_2d(self, x, y, mu, sigma, limits_x = (0, 1), limits_y = (0, 1)):
        def gaussian_2d(x, y, mu, sigma):
            return np.exp(-((x - mu[0])**2 + (y - mu[1])**2)/(2*sigma**2))
        
        n = self.periodic_levels
        dLx = limits_x[1] - limits_x[0]
        dLy = limits_y[1] - limits_y[0]

        components = np.array([gaussian_2d(x, y, mu+[dLx*k_x, dLy*k_y], sigma) for k_x, k_y in product(range(-n, n+1), repeat=2)])
        return components.sum(axis=0)
    
    def generate(self, grid: dynabench.grid.Grid, random_state: int = 42):
        np.random.seed(random_state)
        x, y = np.meshgrid(grid.x, grid.y)

        m = np.array([grid.get_random_point_within_domain() for i in range(self.components)])

        u = self.zero_level+np.zeros_like(x)

        for i in range(self.components):

            component = self._wrapped_gaussian_2d(x, y, m[i], 0.1)

            u = u + np.random.uniform(-1, 1) * component

        return u
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dynabench.solver.diff import differential_operator
    from scipy.signal import convolve2d

    initial_condition = WrappedGaussians((64,64), components=5, random_state=42)
    u = initial_condition.generate()
    d_u = differential_operator("u_x_1_y_0", dx=1/512, dy=1/512)
    print(d_u)

    for _ in range(10):
        u = convolve2d(u, d_u, mode="same", boundary="wrap")
    plt.imshow(u)
    plt.savefig("initial_condition.png")
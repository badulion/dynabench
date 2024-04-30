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
                 grid_size: tuple = (64, 64), 
                 components: int = 1, 
                 zero_level: float = 0.0, 
                 random_state: int = None, 
                 **kwargs):
        super(SumOfGaussians, self).__init__(**kwargs)
        self.grid_size = grid_size
        self.components = components
        self.zero_level = zero_level
        self.random_state = random_state
        
    def __str__(self):
        return "Sum of gaussians"
    
    def generate(self, *args, **kwargs):
        np.random.seed(self.random_state)
        x, y = np.meshgrid(np.linspace(0,1,self.grid_size[0], endpoint=False), np.linspace(0,1,self.grid_size[1], endpoint=False))

        mx = [np.random.choice(self.grid_size[0]) for i in range(self.components)]
        my = [np.random.choice(self.grid_size[1]) for i in range(self.components)]

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
import h5py
import numpy as np

from typing import List

class EquationMovingWindowIterator:
    """
        Iterator for arbitrary equations generated using the dynabench solver. Each sample returned by the __getitem__ method is a tuple of 
        (data_input, data_target, points), where data_input is the input data of shape (L, F, H, W), data_target is the target data of shape (R, F, H, W), and points are the points in the grid of shape (H, W, 2).
        In this context L corresponds to the lookback parameter and R corresponds to the rollout parameter. H and W are the height and width of the grid, respectively. F is the number of variables in the equation system.

        Parameters
        ----------
        data_path : str
            Path to the data file in h5 format.
        lookback : int
            Number of time steps to look back. This corresponds to the L parameter.
        rollout : int
            Number of time steps to predict. This corresponds to the R parameter.
    """
    def __init__(
            self,
            data_path: str,
            lookback: int,
            rollout: int,
    ):
        self.data_path = data_path
        self.lookback = lookback
        self.rollout = rollout


    def __len__(self):
        with h5py.File(self.data_path, "r") as f:
            return len(f['times']) - self.lookback - self.rollout + 1
        
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index < len(self):
            sample = self[self.index]
            self.index += 1
            return sample
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        index = int(index) % len(self)

        # select data
        with h5py.File(self.data_path, "r") as f:
            data_input = f['data'][index:index+self.lookback]
            data_target = f['data'][index+self.lookback:index+self.lookback+self.rollout]
            x_coords = f['x_coords'][:]
            y_coords = f['y_coords'][:]
        
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.stack([X, Y], axis=-1)

        return data_input, data_target, points
    
    def get_full_simulation_data(self):
        """
            This method returns the full simulation data from the data file, along with the points in the grid.
            
            Returns
            -------
            np.ndarray, np.ndarray
                The data and the points. The data has shape (T, F, H, W) and the points have shape (H, W, 2), where T is the number of time steps, F is the number of variables, H and W are the height and width of the grid, respectively.
        """
        with h5py.File(self.data_path, "r") as f:
            data = f['data'][:]
            x_coords = f['x_coords'][:]
            y_coords = f['y_coords'][:]
        
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.stack([X, Y], axis=-1)

        return data, points
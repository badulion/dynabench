import h5py
import numpy as np

from typing import List

class EquationMovingWindowIterator:
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
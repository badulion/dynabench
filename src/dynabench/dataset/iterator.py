import os
import glob
from typing import Any
import h5py
import numpy as np

from .download import download_equation

class DynabenchIterator:

    def __init__(
        self,
        split: str="train",
        equation: str="wave",
        structure: str="cloud",
        resolution: str="low",
        base_path: str="data",
        lookback: int=1,
        rollout: int=1,
        download: bool=False,
        *args,
        **kwargs,
    ) -> None:
        
        # download
        if download:
            download_equation(equation, structure, resolution, base_path)
        
        # parameters
        self.split = split
        self.equation = equation
        self.structure = structure
        self.resolution = resolution
        self.base_path = base_path
        self.lookback = lookback
        self.rollout = rollout

        # get the shapes of the simulations
        self.file_list = glob.glob(os.path.join(base_path, equation, structure, resolution, f"*{split}*.h5"))
        self.file_list.sort()
        
        self.shapes = []
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.shapes.append(f['data'].shape)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # calculate starting indices for each getitem call
        self.usable_simulation_lengths = [(shape[1] - self.lookback - self.rollout+1) for shape in self.shapes]
        self.number_of_simulations = [shape[0] for shape in self.shapes]
        self.datapoints_per_file = [length * number for length, number in zip(self.usable_simulation_lengths, self.number_of_simulations)]
        self.starting_indices = np.cumsum(self.datapoints_per_file) - self.datapoints_per_file[0]

    def _check_exists(self):
        return len(self.file_list) > 0
    
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index > len(self) or index < 0:
            raise IndexError("Index out of bounds")
        
        
        
        # select appropriate file and indices
        file_selector = [i for i, starting_index in enumerate(self.starting_indices) if starting_index <= index][-1]
        raw_idx_within_file = index - self.starting_indices[file_selector]
        simulation_idx = raw_idx_within_file // self.usable_simulation_lengths[file_selector]
        temporal_idx = raw_idx_within_file % self.usable_simulation_lengths[file_selector]
        file = self.file_list[file_selector]

        # select data
        with h5py.File(file, "r") as f:
            data_x = f['data'][simulation_idx, temporal_idx:temporal_idx+self.lookback]
            data_y = f['data'][simulation_idx, temporal_idx+self.lookback:temporal_idx+self.lookback+self.rollout]
            points = f['points'][simulation_idx]


        return data_x, data_y, points

    def __len__(self):
        return sum(self.datapoints_per_file)

if __name__ == "__main__":
    it = DynabenchIterator(equation="advection", structure="cloud", resolution="low", lookback=16)
    x, y, points = it[0]
import os
import glob
from typing import Any
import h5py
import numpy as np

from . import download_equation

class DynabenchSimulationIterator:

    def __init__(
        self,
        split: str="train",
        equation: str="wave",
        structure: str="cloud",
        resolution: str="low",
        base_path: str="data",
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
        self.number_of_simulations = [shape[0] for shape in self.shapes]

        self.file_index_mapping = np.cumsum(self.number_of_simulations) - self.number_of_simulations[0]


    def _check_exists(self):
        return len(self.file_list) > 0
    
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index > len(self) or index < 0:
            raise IndexError("Index out of bounds")
        
        
        
        # select appropriate file and indices
        file_selector = [i for i, starting_index in enumerate(self.file_index_mapping) if starting_index <= index][-1]

        file = self.file_list[file_selector]
        index = index - self.file_index_mapping[file_selector]

        # select data
        with h5py.File(file, "r") as f:
            data = f['data'][index]
            points = f['points'][index]


        return data, points

    def __len__(self):
        return sum(self.number_of_simulations)

if __name__ == "__main__":
    it = DynabenchSimulationIterator(equation="advection", structure="cloud", resolution="low", base_path="data")
    import tqdm

    print(it[501])

    for i in tqdm.tqdm(it):
        pass
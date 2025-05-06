import h5py
import numpy as np

from typing import List, Optional
from ._dataitems import DataItem
from .transforms import BaseTransform, DefaultTransform

class BaseListMovingWindowIterator:
    """
        Iterator for arbitrary equations generated using the dynabench solver. Each sample returned by the __getitem__ method is a tuple of 
        (data_input, data_target, points), where data_input is the input data of shape (L, F, H, W), data_target is the target data of shape (R, F, H, W), and points are the points in the grid of shape (H, W, 2).
        In this context L corresponds to the lookback parameter and R corresponds to the rollout parameter. H and W are the height and width of the grid, respectively. F is the number of variables in the equation system.

        Parameters
        ----------
        data_paths : str
            List of paths to the files containing the simulation data.
        lookback : int
            Number of time steps to look back. This corresponds to the L parameter.
        rollout : int
            Number of time steps to predict. This corresponds to the R parameter.
        squeeze_lookback_dim: bool
            Whether to squeeze the lookback dimension. Defaults to False. If lookback > 1 has no effect.
        is_batched: bool
            Whether the data is batched. Defaults to False. If True, the data is expected to be of shape (B, L, F, H, W), where B is the batch size.
        dtype: np.dtype
            Data type of the input data. Defaults to np.float32. 
    """
    def __init__(
            self,
            data_paths: List[str],
            lookback: int,
            rollout: int,
            squeeze_lookback_dim: bool = True,
            is_batched: bool = False,
            transforms: Optional[BaseTransform] = None,
            dtype: np.dtype=np.float32,
            ) -> None:

        self.lookback = lookback
        self.rollout = rollout
        self.squeeze_lookback_dim = squeeze_lookback_dim
        self.is_batched = is_batched
        self.transforms = transforms
        self.dtype = dtype
        

        # get the shapes of the simulations
        self.file_list = data_paths
        self.file_list.sort()
        
        self.shapes = []
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                shape = f['data'].shape
                if not self.is_batched:
                    shape = (1,) + shape
                self.shapes.append(shape)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # calculate starting indices for each getitem call
        self.usable_simulation_lengths = [(shape[1] - self.lookback - self.rollout+1) for shape in self.shapes]
        self.number_of_simulations = [shape[0] for shape in self.shapes]
        self.datapoints_per_file = [length * number for length, number in zip(self.usable_simulation_lengths, self.number_of_simulations)]
        self.starting_indices = np.cumsum(self.datapoints_per_file) - self.datapoints_per_file[0]

    def _check_exists(self) -> bool:
        return len(self.file_list) > 0
    
    def _load_dataitem_at_index(self, index) -> DataItem:
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
            if self.is_batched:
                data_x = f['data'][simulation_idx, temporal_idx:temporal_idx+self.lookback]
                data_y = f['data'][simulation_idx, temporal_idx+self.lookback:temporal_idx+self.lookback+self.rollout]
                points = f['points'][simulation_idx]
            else:
                data_x = f['data'][temporal_idx:temporal_idx+self.lookback, ...]
                data_y = f['data'][temporal_idx+self.lookback:temporal_idx+self.lookback+self.rollout, ...]
                points = f['points'][:]

        if self.squeeze_lookback_dim and self.lookback == 1:
            data_x = np.squeeze(data_x, axis=0)

        if self.dtype is not None:
            data_x = data_x.astype(self.dtype)
            data_y = data_y.astype(self.dtype)
            points = points.astype(self.dtype)
        
        return DataItem(data_x, data_y, points)

    def __getitem__(self, index) -> DataItem:
        dataitem = self._load_dataitem_at_index(index)
        if self.transforms is None:
            return dataitem
        return self.transforms(dataitem)
        
    def __len__(self) -> int:
        return sum(self.datapoints_per_file)
    
    
class BaseListSimulationIterator:
    """
        Iterates over full simulations. Each sample returned by the __getitem__ method is a tuple of
        (data, points), where data is the simulation data of shape (T, F, H, W) and points are the points in the grid of shape (H, W, 2).
        In this context T corresponds to the number of time steps, H and W are the height and width of the grid, respectively. F is the number of variables in the equation system.
        
        Parameters
        ----------
        data_paths : str
            List of paths to the files containing the simulation data.
        lookback : int
            Number of time steps to look back. This corresponds to the L parameter.
        rollout : int
            Number of time steps to predict. This corresponds to the R parameter.
    """

    def __init__(
        self,
        data_paths: List[str],
        is_batched: bool = False,
        transforms: Optional[BaseTransform] = None,
        dtype: np.dtype=np.float32,
    ) -> None:
        
        self.is_batched = is_batched
        self.transforms = transforms
        self.dtype = dtype

        # get the shapes of the simulations
        self.file_list = data_paths
        self.file_list.sort()
        
        self.shapes = []
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                shape = f['data'].shape
                if not self.is_batched:
                    shape = (1,) + shape
                self.shapes.append(shape)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # calculate starting indices for each getitem call
        self.number_of_simulations = [shape[0] for shape in self.shapes]

        self.file_index_mapping = np.cumsum(self.number_of_simulations) - self.number_of_simulations[0]


    def _check_exists(self) -> bool:
        return len(self.file_list) > 0
    
    def _load_dataitem_at_index(self, index) -> DataItem:
        if index < 0:
            index += len(self)
        if index > len(self) or index < 0:
            raise IndexError("Index out of bounds")
        
        # select appropriate file and indices
        file_selector = [i for i, starting_index in enumerate(self.file_index_mapping) if starting_index <= index][-1]
        raw_idx_within_file = index - self.file_index_mapping[file_selector]
        file = self.file_list[file_selector]

        # select data
        with h5py.File(file, "r") as f:
            if self.is_batched:
                data = f['data'][raw_idx_within_file]
                points = f['points'][raw_idx_within_file]
            else:
                data = f['data'][:]
                points = f['points'][:]   
                
        if self.dtype is not None:
            data = data.astype(self.dtype)
            points = points.astype(self.dtype)

        return DataItem(data, None, points)
    
    def __getitem__(self, index) -> DataItem:
        dataitem = self._load_dataitem_at_index(index)
        if self.transforms is None:
            return dataitem
        return self.transforms(dataitem)

    def __len__(self) -> int:
        return sum(self.number_of_simulations)
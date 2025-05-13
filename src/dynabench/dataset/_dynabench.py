import os
import glob
from typing import Any
import h5py
import numpy as np

from ._base import BaseListMovingWindowIterator
from ._download import download_equation
from ._dataitems import DataItem, GridDataItem, CloudDataItem
from .transforms import BaseTransform, DefaultTransform
from warnings import warn

class DynabenchIterator(BaseListMovingWindowIterator):
    """
    Iterator for the Dynabench dataset. This iterator will iterate over each simulation in the dataset, 
    by moving a window over the simulation data. 
    The window size is defined by the lookback and rollout parameters, which define the number of timesteps
    to be used as input and output, respectively.
       
    Parameters
    ----------
    split : str
        The split of the dataset to use. Can be "train", "val" or "test".
    equation : str
        The equation to use. Can be "advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffustion" or "wave".
    structure : str
        The structure of the dataset. Can be "cloud" or "grid".
    resolution : str
        The resolution of the dataset. Can be *low*, *medium*, *high* or *full*. 
        Low resolution corresponds to 225 points in total (aranged in a 15x15 grid for the grid structure).
        Medium resolution corresponds to 484 points in total (aranged in a 22x22 grid for the grid structure).   
        High resolution corresponds to 900 points in total (aranged in a 30x30 grid for the grid structure).
        Full resolution uses the full simulation grid of shape (64x64) that has been used to numerically solve the simulations.
    base_path : str
        Location where the data is stored. Defaults to "data".
    lookback : int
        Number of timesteps to use for the input data. Defaults to 1.
    squeeze_lookback_dim: bool
        Whether to squeeze the lookback dimension. Defaults to False. If lookback > 1 has no effect.
    rollout : int
        Number of timesteps to use for the target data. Defaults to 1.
    download: int
        Whether to download the data. Defaults to False.
    """
    def __init__(
        self,
        split: str="train",
        equation: str="wave",
        structure: str="cloud",
        resolution: str="low",
        base_path: str="data",
        lookback: int=1,
        squeeze_lookback_dim: bool=False,
        rollout: int=1,
        transforms: BaseTransform=DefaultTransform(),
        dtype: np.dtype=np.float32,
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
        self.download = download

        # get the shapes of the simulations
        self.file_list = glob.glob(os.path.join(base_path, equation, structure, resolution, f"*{split}*.h5"))

        super().__init__(
            data_paths = self.file_list,
            lookback = lookback,
            rollout = rollout,
            squeeze_lookback_dim = squeeze_lookback_dim,
            is_batched = True,
            transforms = transforms,
            dtype = dtype,
        )

class DynabenchSimulationIterator:
    """
    Iterator for the Dynabench dataset. This iterator will iterate all the simulations in the dataset, returning the full simulation as a single sample.
    
    Parameters
    ----------
    split : str
        The split of the dataset to use. Can be "train", "val" or "test".
    equation : str
        The equation to use. Can be "advection", "burgers", "gasdynamics", "kuramotosivashinsky", "reactiondiffustion" or "wave".
    structure : str
        The structure of the dataset. Can be "cloud" or "grid".
    resolution : str
        The resolution of the dataset. Can be *low*, *medium*, *high* or *full*. 
        Low resolution corresponds to 225 points in total (aranged in a 15x15 grid for the grid structure).
        Medium resolution corresponds to 484 points in total (aranged in a 22x22 grid for the grid structure).   
        High resolution corresponds to 900 points in total (aranged in a 30x30 grid for the grid structure).
        Full resolution uses the full simulation grid of shape (64x64) that has been used to numerically solve the simulations.
    base_path : str
        Location where the data is stored. Defaults to "data".
    download: int
        Whether to download the data. Defaults to False.
    dtype: np.dtype
        Data type of the input data. Defaults to np.float32.
    """
    

    def __init__(
        self,
        split: str="train",
        equation: str="wave",
        structure: str="cloud",
        resolution: str="low",
        transforms: BaseTransform=DefaultTransform(),
        base_path: str="data",
        download: bool=False,
        dtype: np.dtype=np.float32,
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
        self.dtype = dtype
        self.download = download

        # get the shapes of the simulations
        self.file_list = glob.glob(os.path.join(base_path, equation, structure, resolution, f"*{split}*.h5"))

        super().__init__(
            data_paths = self.file_list,
            is_batched = True,
            transforms = transforms,
            dtype = dtype,
        )
        

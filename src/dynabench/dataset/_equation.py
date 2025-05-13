import h5py
import numpy as np
import pathlib

from ._base import BaseListMovingWindowIterator, BaseListSimulationIterator
from .transforms import BaseTransform
from typing import List, Optional


class EquationMovingWindowIterator(BaseListMovingWindowIterator):
    """
        Iterator for arbitrary equations generated using the dynabench solver. Each sample returned by the __getitem__ method is a tuple of 
        (data_input, data_target, points), where data_input is the input data of shape (L, F, H, W), data_target is the target data of shape (R, F, H, W), and points are the points in the grid of shape (H, W, 2).
        In this context L corresponds to the lookback parameter and R corresponds to the rollout parameter. H and W are the height and width of the grid, respectively. F is the number of variables in the equation system.

        Parameters
        ----------
        eq_dir : str
            Path to the directory where the generated simulations are stored.
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
        selected_simulations: List[str]
            List of selected simulation names to load. If None, all simulations in the directory are loaded.
    """
    def __init__(
            self,
            eq_dir: str,
            lookback: int,
            rollout: int,
            selected_simulations: Optional[List[str]] = None,
            squeeze_lookback_dim: bool = True,
            is_batched: bool = True,
            transforms: Optional[BaseTransform] = None,
            dtype: np.dtype=np.float32,
            ) -> None:
        eq_dir = pathlib.Path(eq_dir)

        # read the directory and get the list of files
        if selected_simulations is not None:
            data_paths = [path for path in eq_dir.iterdir() if path.name in selected_simulations]
        else:
            data_paths = [path for path in eq_dir.iterdir() if path.name.endswith(".h5")]
        
        super().__init__(
            data_paths = data_paths,
            lookback = lookback,
            rollout = rollout,
            squeeze_lookback_dim = squeeze_lookback_dim,
            is_batched = False,
            transforms = transforms,
            dtype = dtype
        )
        
class EquationSimulationIterator(BaseListSimulationIterator):
    """
        Iterator for full equations generated using the dynabench solver. Each sample returned by the __getitem__ method is a tuple of 
        (data_input, points), where data_input is the input data of shape (L, F, H, W), data_target is the target data of shape (R, F, H, W), and points are the points in the grid of shape (H, W, 2).
        In this context L corresponds to the lookback parameter and R corresponds to the rollout parameter. H and W are the height and width of the grid, respectively. F is the number of variables in the equation system.

        Parameters
        ----------
        eq_dir : str
            Path to the directory where the generated simulations are stored.
        is_batched: bool
            Whether the data is batched. Defaults to False. If True, the data is expected to be of shape (B, L, F, H, W), where B is the batch size.
        dtype: np.dtype
            Data type of the input data. Defaults to np.float32.
        selected_simulations: List[str]
            List of selected simulation names to load. If None, all simulations in the directory are loaded.
    """
    def __init__(
            self,
            eq_dir: str,
            selected_simulations: Optional[List[str]] = None,
            is_batched: bool = True,
            transforms: Optional[BaseTransform] = None,
            dtype: np.dtype=np.float32,
            ) -> None:
        eq_dir = pathlib.Path(eq_dir)

        # read the directory and get the list of files
        if selected_simulations is not None:
            data_paths = [path for path in eq_dir.iterdir() if path.name in selected_simulations]
        else:
            data_paths = [path for path in eq_dir.iterdir() if path.name.endswith(".h5")]
        
        super().__init__(
            data_paths = data_paths,
            is_batched = False,
            transforms = transforms,
            dtype = dtype
        )
    
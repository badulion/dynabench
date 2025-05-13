from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np

from typing import Optional, Any

@dataclass
class DataItem:
    """
    Base class for data items.
    """
    x: npt.NDArray
    y: Optional[npt.NDArray] = None
    pos: Optional[npt.NDArray] = None


@dataclass
class GridDataItem(DataItem):
    """
    Data class for grid data.
    """


@dataclass
class CloudDataItem(DataItem):
    """
    Data class for 2D grid data.
    """
    pos : npt.NDArray # no longer optional
    y: Optional[npt.NDArray] = None
    neighbors: Optional[npt.NDArray] = None
    distances: Optional[npt.NDArray] = None
    edgelist: Optional[npt.NDArray] = None
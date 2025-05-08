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
    y: Optional[npt.NDArray] = field(default_factory=lambda: np.array([], dtype=np.float32))
    pos: Optional[npt.NDArray] = field(default_factory=lambda: np.array([], dtype=np.float32))


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
    knn_graph: Optional[npt.NDArray] = field(default_factory=lambda: np.array([], dtype=np.float32))
from dataclasses import dataclass
import numpy.typing as npt

from typing import Optional, Any

@dataclass
class DataItem:
    """
    Base class for data items.
    """
    x: npt.NDArray
    y: npt.NDArray
    pos: npt.NDArray


@dataclass
class GridItem(DataItem):
    """
    Data class for grid data.
    """


@dataclass
class CloudItem(DataItem):
    """
    Data class for 2D grid data.
    """
    knn_graph: Optional[npt.NDArray] = None
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List
from copy import copy
import einops
import numpy as np
from ._data_items import GridItem, CloudItem, DataItem

from sklearn.neighbors import NearestNeighbors



class BaseTransform(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, data_item: DataItem):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.kwargs)
    
    def _check_data(self, data_item: DataItem):
        if not isinstance(data_item, DataItem):
            raise ValueError(f"Data_item should be an instance of DataItem, got {type(data_item)}")
        

class Compose(BaseTransform):
    """
    Compose function for combining multiple transforms.

    Parameters
    ----------
    transforms : List[BaseTransform]
        List of transforms to be applied to the data
    """
    def __init__(self, transforms: List[BaseTransform]):

        if not isinstance(transforms, Iterable):
            raise ValueError("Transforms should be an iterable")
        elif len(list(transforms)) == 0:
            raise ValueError("No transforms were given")
        else:
            for i in transforms:
                if i is None:
                    raise ValueError("Transform can not be None")
                elif not isinstance(i, BaseTransform):
                    raise ValueError(f"Transform should be an instance of BaseTransform, got {type(i)}")
        
        self.transforms = transforms

    def __call__(self, data_item: DataItem) -> DataItem:
        """
        The method iterates over transformations and apply them to the simulation data.

        Parameters
        ----------
        simulation : DataItem
            simulation data

        Returns
        -------
        DataItem
            augmented simulation data
        """
        self._check_data(data_item)
        result = copy(data_item)
        for aug in self.transforms:
            result = aug(result)
        return result

    def __repr__(self):
        return self.__class__.__name__ + str(self.transforms)


class DefaultTransform(BaseTransform):
    """

    """
    def __init__(self):
        super().__init__()

    def __call__(self, data_item: DataItem) -> DataItem:
        """
        Default transformation for a data item. Does not modify the data.

        Parameters
        ----------
        data_item : DataItem

        Returns
        -------
        DataItem
            transformed data_item
        """
        self._check_data(data_item)

        return data_item
    
    def check_if_valid(self):
        return True
    
class Grid2Cloud(BaseTransform):
    """
    Create a Cloud item from a grid data item

    Parameters
    ----------
    data_item : GridItem

    Returns
    -------
    CloudItem
        data_item with cloud shape
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data_item: GridItem) -> CloudItem:
        """
        Default transformation for a data item. Does not modify the data.

        Parameters
        ----------
        data_item : DataItem

        Returns
        -------
        DataItem
            transformed data_item
        """
        self._check_data(data_item)

        assert isinstance(data_item, GridItem)

        cloud_x = einops.rearrange(data_item.x, 'l c w h -> l (w h) c')

        cloud_y = einops.rearrange(data_item.y, 'r c w h -> r (w h) c')

        cloud_pos = einops.rearrange(data_item.pos, 'w h d -> (w h) d')

        return CloudItem(
            x=cloud_x,
            y=cloud_y,
            pos=cloud_pos
        )
    
    def check_if_valid(self):
        return True

class MakeKNNGraph(BaseTransform):
    """
    Create a KNN graph from the cloud data.

    Parameters
    ----------
    data_item : CloudItem

    Returns
    -------
    CloudItem
        data_item with knn_graph
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def __call__(self, data_item: CloudItem) -> CloudItem:
        """
        Default transformation for a data item. Does not modify the data.

        Parameters
        ----------
        data_item : DataItem

        Returns
        -------
        DataItem
            transformed data_item
        """
        self._check_data(data_item)

        points_padded = np.concatenate((data_item.pos,
                data_item.pos + np.array([0, 1]),
                data_item.pos + np.array([1, 0]), 
                data_item.pos + np.array([1, 1]), 
                data_item.pos + np.array([0, -1]),
                data_item.pos + np.array([-1, 0]),
                data_item.pos + np.array([-1, -1]),
                data_item.pos + np.array([1, -1]),
                data_item.pos + np.array([-1, 1]),
                ), axis=0)

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(points_padded)
        indices = nbrs.kneighbors(points_padded, return_distance=False)
        N = data_item.pos.shape[0]
        src = np.repeat(np.arange(N), self.k)
        dst = (indices[:N,1:]%N).flatten()
        knn_graph = np.stack([dst, src], axis=0)

        return CloudItem(
            x=data_item.x,
            y=data_item.y,
            pos=data_item.pos,
            knn_graph=knn_graph,
        )
    
    def check_if_valid(self):
        return True

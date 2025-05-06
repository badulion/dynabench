from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List
from copy import copy
import einops
import numpy as np
from ._dataitems import GridDataItem, CloudDataItem, DataItem

from scipy.spatial import KDTree



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
    Iterates over transformations and applies them to the data item.

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
        self._check_data(data_item)
        result = copy(data_item)
        for aug in self.transforms:
            result = aug(result)
        return result

    def __repr__(self):
        return self.__class__.__name__ + str(self.transforms)


class DefaultTransform(BaseTransform):
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
    def __init__(self):
        super().__init__()

    def __call__(self, data_item: DataItem) -> DataItem:
        self._check_data(data_item)

        return data_item
    
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

    def __call__(self, data_item: DataItem) -> CloudDataItem:
        self._check_data(data_item)

        cloud_x = einops.rearrange(data_item.x, '... c w h -> ... (w h) c')

        cloud_y = einops.rearrange(data_item.y, '... c w h -> ... (w h) c')

        cloud_pos = einops.rearrange(data_item.pos, 'w h d -> (w h) d')

        return CloudDataItem(
            x=cloud_x,
            y=cloud_y,
            pos=cloud_pos
        )
    
class ToDict(BaseTransform):
    """
    Convert the data item to a dictionary.

    Parameters
    ----------
    data_item : DataItem

    Returns
    -------
    dict
        data_item as a dictionary
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data_item: DataItem) -> dict:
        self._check_data(data_item)

        return data_item.__dict__

class KNNGraph(BaseTransform):
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

    def __call__(self, data_item: CloudDataItem) -> CloudDataItem:
        self._check_data(data_item)

        points = data_item.pos
        points_padded = np.concatenate(
               (points,
                points + np.array([0, 1]),
                points + np.array([1, 0]), 
                points + np.array([1, 1]), 
                points + np.array([0, -1]),
                points + np.array([-1, 0]),
                points + np.array([-1, -1]),
                points + np.array([1, -1]),
                points + np.array([-1, 1]),
                ), axis=0)

        tree = KDTree(points_padded)
        distances, n = tree.query(points, k=self.k+1)
        n = n[:, 1:] # remove the first column, which is the point itself
        n = n % points.shape[0]

        return CloudDataItem(
            x=data_item.x,
            y=data_item.y,
            pos=data_item.pos,
            knn_graph=n,
        )
    
    def check_if_valid(self):
        return True
    
class EdgeListFromKNN(BaseTransform):
    """
    Create an edge list from the KNN graph.

    Parameters
    ----------
    data_item : CloudItem

    Returns
    -------
    CloudItem
        data_item with knn_graph
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data_item: CloudDataItem) -> CloudDataItem:
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

        knn_graph = data_item.knn_graph
        num_points = knn_graph.shape[0]
        k = knn_graph.shape[-1]
        src = np.repeat(np.arange(num_points), k)
        dst = knn_graph.flatten()
        edge_list = np.stack((src, dst), axis=0)
        
        return CloudDataItem(
            x=data_item.x,
            y=data_item.y,
            pos=data_item.pos,
            knn_graph=edge_list,
        )
    
    def check_if_valid(self):
        return True
    
class EdgeList(Compose):
    """
    Create an edge list graph (src, dst) to use with PyG.

    Parameters
    ----------
    data_item : CloudItem

    Returns
    -------
    CloudItem
        data_item with edge_list as knn_graph
    """
    def __init__(self, k: int):
        super().__init__(transforms=[KNNGraph(k=k), EdgeListFromKNN()])
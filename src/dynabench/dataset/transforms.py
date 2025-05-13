from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Tuple
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

        return {key: value for key, value in data_item.__dict__.items() if value is not None}

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
    def __init__(self, k: int, grid_limits: Tuple[float] = (1.0, 1.0)):
        super().__init__()
        self.k = k
        self.grid_limits = grid_limits

    def __call__(self, data_item: CloudDataItem) -> CloudDataItem:
        self._check_data(data_item)

        points = data_item.pos
        self.grid_limits = np.array(self.grid_limits, dtype=np.float32)
        points_padded = np.concatenate(
               (points,
                points + np.array([0, 1]) * self.grid_limits,
                points + np.array([1, 0]) * self.grid_limits,
                points + np.array([1, 1]) * self.grid_limits,
                points + np.array([0, -1]) * self.grid_limits,
                points + np.array([-1, 0]) * self.grid_limits,
                points + np.array([-1, -1]) * self.grid_limits,
                points + np.array([1, -1]) * self.grid_limits,
                points + np.array([-1, 1]) * self.grid_limits,
                ), axis=0)

        tree = KDTree(points_padded)
        _, neighbors = tree.query(points, k=self.k+1)
        neighbors = neighbors[:, 1:] # remove the first column, which is the point itself
        
        # calculate distances
        neighbor_points = points_padded[neighbors]
        points_unsqueezed = np.expand_dims(points, axis=1)
        distances = neighbor_points - points_unsqueezed
        
        neighbors = neighbors % points.shape[0]

        return CloudDataItem(
            x=data_item.x,
            y=data_item.y,
            pos=data_item.pos,
            neighbors=neighbors,
            distances=distances,
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

        neighbors = data_item.neighbors
        num_points = neighbors.shape[0]
        k = neighbors.shape[-1]
        src = np.repeat(np.arange(num_points), k)
        dst = neighbors.flatten()
        edge_list = np.stack((src, dst), axis=0)
        
        return CloudDataItem(
            x=data_item.x,
            y=data_item.y,
            pos=data_item.pos,
            neighbors=data_item.neighbors,
            distances=data_item.distances,
            edgelist=edge_list,
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

class TypeCaster(BaseTransform):
    """
    Cast the data item to the correct type. (In place!!!)
    """
    def __init__(self, dtype: np.dtype = np.float32):        
        super().__init__()
        self.dtype = dtype

    def __call__(self, data_item: DataItem) -> DataItem:
        self._check_data(data_item)
        data_item.x = data_item.x.astype(self.dtype)
        data_item.y = data_item.y.astype(self.dtype)
        if hasattr(data_item, 'pos') and data_item.pos is not None:
            data_item.pos = data_item.pos.astype(self.dtype)
            
        if hasattr(data_item, 'distances') and data_item.distances is not None:
            data_item.distances = data_item.distances.astype(self.dtype)

        return data_item
    
class GridDownsampleFactor(BaseTransform):
    """
        Downsample the grid by a factor.

        Parameters
        ----------
        factor : int
            Factor by which to downsample the grid.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, data_item: DataItem) -> DataItem:
        self._check_data(data_item)

        # Downsample the grid
        if data_item.x.ndim == 3:
            downsampled_x = data_item.x[:, ::self.factor, ::self.factor]
        else:
            downsampled_x = data_item.x[:, :, ::self.factor, ::self.factor]
        
        if hasattr(data_item, 'y') and data_item.y is not None:
            downsampled_y = data_item.y[:, :, ::self.factor, ::self.factor]
        else:
            downsampled_y = None

        if hasattr(data_item, 'pos') and data_item.pos is not None:
            downsampled_pos = data_item.pos[::self.factor, ::self.factor]

        data_item = DataItem(
            x=downsampled_x,
            y=downsampled_y,
            pos=downsampled_pos,
        )

        return data_item
    
class GridDownsampleFFT(BaseTransform):
    """
        Downsample the grid to a smaller size using FFT.

        Parameters
        ----------
        target_size : Tuple[int, int]
            Target size of the grid.
    """

    def __init__(self, target_size: Tuple[int, int] = (1.0, 1.0)):
        super().__init__()
        self.target_size = target_size

    def __call__(self, data_item: DataItem) -> DataItem:
        self._check_data(data_item)

        # Get the original grid size
        original_size = data_item.x.shape[-2:]

        # Downsample using FFT
        downsampled_x = np.fft.rfft2(data_item.x, s=self.target_size)
        downsampled_x = np.fft.irfft2(downsampled_x)
        
        if hasattr(data_item, 'y') and data_item.y is not None:
            downsampled_y = np.fft.rfft2(data_item.y, s=self.target_size)
            downsampled_y = np.fft.irfft2(downsampled_y)
        else:
            downsampled_y = None

        if hasattr(data_item, 'pos') and data_item.pos is not None:
            downsampled_pos = np.fft.rfft2(data_item.pos, s=self.target_size)
            downsampled_pos = np.fft.irfft2(downsampled_pos)

        data_item = DataItem(
            x=downsampled_x,
            y=downsampled_y,
            pos=downsampled_pos,
        )

        return data_item
    
class PointSampling(BaseTransform):
    """
        Point sampling transform for the dataset.

        Parameters
        ----------
        num_points : int
            Number of points to sample.
        k : int
            Number of nearest neighbors to use for the KNN graph.
    """

    def __init__(self, num_points: int = 900):
        super().__init__()
        self.num_points = num_points
        
        

    def __call__(self, data_item: CloudDataItem) -> CloudDataItem:
        self._check_data(data_item)
        
        total_points = data_item.pos.shape[0]
        indices = np.random.choice(total_points, self.num_points, replace=False)

        points = data_item.pos[indices]
        y = data_item.y[:, indices]
        if data_item.x.ndim == 3:
            x = data_item.x[:, indices]
        else:
            x = data_item.x[indices]
        

        return CloudDataItem(
            x=x,
            y=y,
            pos=points
        )
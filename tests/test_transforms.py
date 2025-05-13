import pytest
import numpy as np
import h5py
from dynabench.dataset._base import BaseListMovingWindowIterator, BaseListSimulationIterator
from dynabench.dataset._dataitems import DataItem, GridDataItem, CloudDataItem
from dynabench.dataset.transforms import DefaultTransform, Grid2Cloud, KNNGraph, EdgeList, ToDict, TypeCaster, GridDownsampleFactor, GridDownsampleFFT, PointSampling

@pytest.fixture
def mock_data_paths(tmp_path):
    # Create mock HDF5 files with dummy data
    file_paths = []
    for i in range(2):
        file_path = tmp_path / f"mock_data_{i}.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=np.random.rand(10, 5, 64, 64))
            f.create_dataset("points", data=np.random.rand(64, 64, 2))
        file_paths.append(str(file_path))
    return file_paths

def test_base_list_moving_window_iterator(mock_data_paths):
    iterator = BaseListMovingWindowIterator(
        data_paths=mock_data_paths,
        lookback=2,
        rollout=2,
        transforms=DefaultTransform(),
        dtype=np.float32,
    )
    assert len(iterator) > 0
    data_item = iterator[0]
    assert isinstance(data_item, DataItem)
    assert data_item.x.shape == (2, 5, 64, 64)
    assert data_item.y.shape == (2, 5, 64, 64)
    assert data_item.pos.shape == (64, 64, 2)

def test_base_list_simulation_iterator(mock_data_paths):
    iterator = BaseListSimulationIterator(
        data_paths=mock_data_paths,
        transforms=DefaultTransform(),
        dtype=np.float32,
    )
    assert len(iterator) > 0
    data_item = iterator[0]
    assert isinstance(data_item, DataItem)
    assert data_item.x.shape == (10, 5, 64, 64)
    assert data_item.pos.shape == (64, 64, 2)

def test_grid2cloud_transform():
    grid_data = GridDataItem(
        x=np.random.rand(1, 5, 64, 64),
        y=np.random.rand(1, 5, 64, 64),
        pos=np.random.rand(64, 64, 2),
    )
    transform = Grid2Cloud()
    cloud_data = transform(grid_data)
    assert isinstance(cloud_data, CloudDataItem)
    assert cloud_data.x.shape == (1, 64 * 64, 5)
    assert cloud_data.y.shape == (1, 64 * 64, 5)
    assert cloud_data.pos.shape == (64 * 64, 2)

def test_knn_graph_transform():
    cloud_data = CloudDataItem(
        x=np.random.rand(1, 5, 100),
        y=np.random.rand(1, 5, 100),
        pos=np.random.rand(100, 2),
    )
    transform = KNNGraph(k=5)
    transformed_data = transform(cloud_data)
    assert transformed_data.neighbors.shape == (100, 5)

def test_edge_list_transform():
    cloud_data = CloudDataItem(
        x=np.random.rand(1, 5, 100),
        y=np.random.rand(1, 5, 100),
        pos=np.random.rand(100, 2),
        neighbors=np.random.randint(0, 100, (100, 5)),
    )
    transform = EdgeList(k=5)
    transformed_data = transform(cloud_data)
    assert transformed_data.edgelist.shape[0] == 2
    assert transformed_data.edgelist.shape[1] == 100 * 5

def test_to_dict_transform():
    data_item = DataItem(
        x=np.random.rand(1, 5, 64, 64),
        y=np.random.rand(1, 5, 64, 64),
        pos=np.random.rand(64, 64, 2),
    )
    transform = ToDict()
    data_dict = transform(data_item)
    assert isinstance(data_dict, dict)
    assert "x" in data_dict and "y" in data_dict and "pos" in data_dict

def test_type_caster_transform():
    data_item = DataItem(
        x=np.random.rand(1, 5, 64, 64),
        y=np.random.rand(1, 5, 64, 64),
        pos=np.random.rand(64, 64, 2),
    )
    transform = TypeCaster(dtype=np.float64)
    transformed_data = transform(data_item)
    assert transformed_data.x.dtype == np.float64
    assert transformed_data.y.dtype == np.float64
    assert transformed_data.pos.dtype == np.float64

def test_grid_downsample_factor_transform():
    grid_data = GridDataItem(
        x=np.random.rand(1, 5, 64, 64),
        y=np.random.rand(1, 5, 64, 64),
        pos=np.random.rand(64, 64, 2),
    )
    transform = GridDownsampleFactor(factor=2)
    downsampled_data = transform(grid_data)
    assert downsampled_data.x.shape[-2:] == (32, 32)
    assert downsampled_data.y.shape[-2:] == (32, 32)
    assert downsampled_data.pos.shape == (32, 32, 2)

def test_grid_downsample_fft_transform():
    grid_data = GridDataItem(
        x=np.random.rand(1, 5, 64, 64),
        y=np.random.rand(1, 5, 64, 64),
        pos=np.random.rand(64, 64, 2),
    )
    transform = GridDownsampleFFT(target_size=(32, 32))
    downsampled_data = transform(grid_data)
    assert downsampled_data.x.shape[-2:] == (32, 32)
    assert downsampled_data.y.shape[-2:] == (32, 32)
    assert downsampled_data.pos.shape[-2:] == (32, 32)

def test_point_sampling_transform():
    cloud_data = CloudDataItem(
        x=np.random.rand(16, 100, 2),
        y=np.random.rand(16, 100, 2),
        pos=np.random.rand(100, 2),
    )
    transform = PointSampling(num_points=50)
    sampled_data = transform(cloud_data)
    assert sampled_data.x.shape[-2] == 50
    assert sampled_data.y.shape[-2] == 50
    assert sampled_data.pos.shape[-2] == 50
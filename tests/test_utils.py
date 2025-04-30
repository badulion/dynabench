import torch
import pytest
from dynabench.model.utils import RolloutWrapper, CloudRolloutWrapper, GridRolloutWrapper

class DummyModel(torch.nn.Module):
    def __init__(self, num_channels, lookback):
        super(DummyModel, self).__init__()
        self.num_channels = num_channels
        self.lookback = lookback

    def forward(self, x, p=None):
        if p is not None:
            return x[:,:,-self.num_channels:]  # Example for cloud structure
        return x[:,-self.num_channels:]  # Example for grid structure

@pytest.fixture
def dummy_model():
    return DummyModel(num_channels=4, lookback=3)

def test_grid_rollout_wrapper(dummy_model):
    x = torch.randn(2, 3, 4, 5, 5)  # batch, lookback, feature, spatial_x, spatial_y
    wrapper = RolloutWrapper(model=dummy_model, structure="grid", batch_first=True)
    wrapper_alias = GridRolloutWrapper(model=dummy_model, batch_first=True)
    t_eval = [1, 2, 3]
    output = wrapper(x, t_eval=t_eval)
    output_alias = wrapper_alias(x, t_eval=t_eval)
    assert torch.equal(output, output_alias)
    assert output.shape == (2, len(t_eval), 4, 5, 5)  # batch, rollout, feature, spatial_x, spatial_y

def test_cloud_rollout_wrapper(dummy_model):
    x = torch.randn(2, 3, 25, 4)  # batch, lookback, points, feature
    p = torch.randn(2, 10, 3)  # batch, points, coordinates
    wrapper = RolloutWrapper(model=dummy_model, structure="cloud", batch_first=True)
    wrapper_alias = CloudRolloutWrapper(model=dummy_model, batch_first=True)
    t_eval = [1, 2]
    output = wrapper(x, p, t_eval=t_eval)
    output_alias = wrapper_alias(x, p, t_eval=t_eval)
    assert torch.equal(output, output_alias)
    assert output.shape == (2, len(t_eval), 25, 4)  # batch, rollout, points, feature

def test_grid_rollout_wrapper_output(dummy_model):
    x = torch.randn(2, 3, 4, 5, 5)  # batch, lookback, feature, spatial_x, spatial_y
    wrapper = GridRolloutWrapper(model=dummy_model, batch_first=True)
    t_eval = range(8)
    output = wrapper(x, t_eval=t_eval)
    
    assert (output[:,:1] == output).all()  # Check that the first rollout is the same as the input

def test_cloud_rollout_wrapper_output(dummy_model):
    x = torch.randn(2, 3, 25, 4)  # batch, lookback, points, feature
    p = torch.randn(2, 10, 3)  # batch, points, coordinates
    wrapper = CloudRolloutWrapper(model=dummy_model, batch_first=True)
    t_eval = range(8)
    output = wrapper(x, p, t_eval=t_eval)
    
    assert (output[:,:1] == output).all()  # Check that the first rollout is the same as the input

def test_invalid_structure(dummy_model):
    with pytest.raises(ValueError, match="Structure must be either 'grid' or 'cloud'"):
        RolloutWrapper(model=dummy_model, structure="invalid")

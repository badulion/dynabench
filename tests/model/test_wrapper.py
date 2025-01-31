import torch
import pytest

from dynabench.model.utils import GridIterativeWrapper
from dynabench.model.utils import PointIterativeWrapper


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_grid_low', 'default_cnn'),
                            ('batched_input_grid_low', 'default_resnet'),
                            ('batched_input_grid_low', 'default_neural_operator'),
                        ])
def test_output_shape_grid(input, model, request):
    input = request.getfixturevalue(input)
    model = request.getfixturevalue(model)
    grid_wrapper = GridIterativeWrapper(model=model)
    output = grid_wrapper(*input)
    assert output.shape == (16, 2, 1, 15, 15)


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_point_low', 'default_point_transformer_v1_low'),
                        ])
def test_output_shape_point(input, model, request):
    input = request.getfixturevalue(input)
    model = request.getfixturevalue(model)
    point_wrapper = PointIterativeWrapper(model=model)
    output = point_wrapper(*input)
    assert output.shape == (16, 2, 225, 1)


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_grid_low', 'default_cnn'),
                            ('batched_input_grid_low', 'default_resnet'),
                            ('batched_input_grid_low', 'default_neural_operator'),
                            ('batched_input_point_low', 'default_point_transformer_v1_low'),
                        ])
def test_cuda_copy_grid(input, model, request):
    if torch.cuda.is_available():
        input = request.getfixturevalue(input)
        model = request.getfixturevalue(model)
        grid_wrapper = GridIterativeWrapper(model=model)
        output = grid_wrapper(*input)
        grid_wrapper.cuda()
        input = input.cuda()
        output = model(input)
        assert output.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")
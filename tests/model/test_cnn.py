import torch
import pytest

@pytest.mark.parametrize("input,expected,model",
                        [
                            ('batched_input_grid_low', (16, 1, 15, 15), 'default_cnn'),
                            ('batched_input_grid_med', (16, 1, 22, 22), 'default_cnn'),
                            ('batched_input_grid_high', (16, 1, 30, 30), 'default_cnn'),
                            ('unbatched_input_grid_low', (1, 15, 15), 'default_cnn'),
                            ('unbatched_input_grid_med', (1, 22, 22), 'default_cnn'),
                            ('unbatched_input_grid_high', (1, 30, 30), 'default_cnn'),
                            ('batched_input_grid_low_channel', (16, 4, 15, 15), 'default_cnn_channel'),
                            ('unbatched_input_grid_low_channel', (4, 15, 15), 'default_cnn_channel'),
                        ])
def test_output_shape(input, expected, model, request):
    input = request.getfixturevalue(input)
    model = request.getfixturevalue(model)
    output = model(input[0])
    assert output.shape == expected


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_grid_low', 'default_cnn'),
                            ('batched_input_grid_med', 'default_cnn'),
                            ('batched_input_grid_high', 'default_cnn'),
                        ])
def test_cuda_copy(input, model, request):
    if torch.cuda.is_available():
        input = request.getfixturevalue(input)
        model = request.getfixturevalue(model)
        model.cuda()
        input = input[0].cuda()
        output = model(input)
        assert output.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")
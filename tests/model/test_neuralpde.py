import torch
import pytest

@pytest.mark.parametrize("input,expected,model",
                        [
                            ('batched_input_grid_low', (16, 1, 1, 15, 15), 'default_neuralpde'),
                            ('batched_input_grid_med', (16, 1, 1, 22, 22), 'default_neuralpde'),
                            ('batched_input_grid_high', (16, 1, 1, 30, 30), 'default_neuralpde'),
                            ('unbatched_input_grid_low', (1, 1, 15, 15), 'default_neuralpde'),
                            ('unbatched_input_grid_med', (1, 1, 22, 22), 'default_neuralpde'),
                            ('unbatched_input_grid_high', (1, 1, 30, 30), 'default_neuralpde'),
                            ('batched_input_grid_low_channel', (16, 1, 4, 15, 15), 'default_neuralpde_channel'),
                            ('unbatched_input_grid_low_channel', (4, 1, 15, 15), 'default_neuralpde_channel'),
                        ])
def test_output_shape(input, expected, model, request):
    input = request.getfixturevalue(input)
    model = request.getfixturevalue(model)
    output = model(*input)
    assert output.shape == expected


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_grid_low', 'default_neuralpde'),
                            ('batched_input_grid_med', 'default_neuralpde'),
                            ('batched_input_grid_high', 'default_neuralpde'),
                        ])
def test_cuda_copy(input, model, request):
    if torch.cuda.is_available():
        input = request.getfixturevalue(input)
        model = request.getfixturevalue(model)
        model.cuda()
        input = input.cuda()
        output = model(*input)
        assert output.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")
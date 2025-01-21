import torch
import pytest

def test_output_shape(batched_input_low, default_cnn):
    output = default_cnn(batched_input_low)
    assert output.shape == (16, 1, 15, 15)


def test_cuda_copy(batched_input_low, default_cnn):
    if torch.cuda.is_available():
        default_cnn.cuda()
        batched_input_low = batched_input_low.cuda()
        output = default_cnn(batched_input_low)
        assert output.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")
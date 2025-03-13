import torch
import pytest

@pytest.mark.parametrize("input,expected,model",
                        [
                            ('batched_input_point_low', (16, 225, 1),           'default_fno_geo'),
                            ('batched_input_point_med', (16, 484, 1),           'default_fno_geo'),
                            ('batched_input_point_high', (16, 900, 1),          'default_fno_geo'),
                            ('unbatched_input_point_low', (225, 1),             'default_fno_geo'),
                            ('unbatched_input_point_med', (484, 1),             'default_fno_geo'),
                            ('unbatched_input_point_high', (900, 1),            'default_fno_geo'),
                            ('batched_input_point_low_channel', (16, 225, 4),   'default_fno_geo_channel'),
                            ('unbatched_input_point_low_channel', (225, 4),     'default_fno_geo_channel'),
                        ])
def test_output_shape(input, expected, model, request):
    input = request.getfixturevalue(input)
    model = request.getfixturevalue(model)
    output = model(*input[:2])
    assert output.shape == expected


@pytest.mark.parametrize("input,model",
                        [
                            ('batched_input_point_low',     'default_fno_geo'),
                            ('batched_input_point_med',     'default_fno_geo'),
                            ('batched_input_point_high',    'default_fno_geo'),
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
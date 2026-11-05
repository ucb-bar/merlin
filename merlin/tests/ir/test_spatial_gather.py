"""Check generated gather indices against independent source pooling semantics."""
import pytest

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_gather import plan_gathered_spatial_max
from merlin.runtime.tensor import Tensor
from merlin.sched.ir import evaluate, instances


@pytest.mark.parametrize('height,width,channels,batches', [(5, 5, 17, 2), (6, 8, 3, 1), (1, 1, 1, 1)])
@pytest.mark.parametrize('scale', [0.5, 0.9900199770927429, 2.0, -0.5])
def test_gathered_windows_equal_source_max(height, width, channels, batches, scale):
    source = SpatialMaxReadout((height, width), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    plan = plan_gathered_spatial_max(source, batches=batches, channels=channels, channel_block=16)
    count = batches * height * width * channels
    tensor = Tensor((batches * height * width, channels), [(index * 53 % 256) - 128 for index in range(count)], 'i8')
    readout = tensor.requant_acc_scale(scale).to_i8()
    expected = readout.maxpool2d_rows(in_dims=source.input_hw, pool_size=source.window,
        pool_stride=source.stride, pool_padding=source.padding, pad_value=-128)
    actual = [None] * len(expected.data)
    window = {}
    calls = 0
    for call, env in instances(plan.kernel):
        args = {name: evaluate(value, env) for name, value in call.args}
        calls += 1
        first, columns = args['first_column'], args['columns']
        if call.instr == 'gather_window_row':
            row = args['source_row']
            assert 0 <= row < tensor.shape[0]
            window[args['window_row']] = readout.data[row * channels + first:row * channels + first + columns]
        else:
            assert set(window) == set(range(9))
            row = args['output_row']
            for column in range(columns):
                actual[row * channels + first + column] = max(values[column] for values in window.values())
            window.clear()
    assert actual == expected.data
    assert calls == plan.dynamic_calls


def test_large_plan_retains_counted_loops():
    plan = plan_gathered_spatial_max(SpatialMaxReadout((112, 112), (3, 3), (2, 2), (1, 1, 1, 1), -128),
        batches=1, channels=64, channel_block=16)
    assert plan.dynamic_calls == 125440
    assert plan.temporary_rows == 9
    assert len(plan.kernel.text()) < 12000


def test_unknown_padding_and_empty_windows_refuse():
    with pytest.raises(ValueError, match='minimum-valued'):
        plan_gathered_spatial_max(SpatialMaxReadout((5, 5), (3, 3), (2, 2), (1, 1, 1, 1), 0),
            batches=1, channels=1, channel_block=16)
    with pytest.raises(ValueError, match='completely padded'):
        plan_gathered_spatial_max(SpatialMaxReadout((5, 5), (3, 3), (1, 1), (3, 3, 3, 3), -128),
            batches=1, channels=1, channel_block=16)

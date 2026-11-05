from dataclasses import replace

import pytest

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max
from merlin.runtime.tensor import Tensor
from merlin.sched.ir import evaluate, instances


@pytest.mark.parametrize("height,width,channels,batches", [(5, 5, 17, 2), (6, 8, 3, 1), (1, 1, 1, 1)])
@pytest.mark.parametrize("scale", [0.5, -0.5, 2.0])
def test_every_staged_byte_and_readout_matches_original_pool(height, width, channels, batches, scale):
    source = SpatialMaxReadout((height, width), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    plan = plan_striped_spatial_max(source, batches=batches, channels=channels, channel_block=16,
                                   transfer_rows=4, capacity_rows=256)
    count = batches * height * width * channels
    tensor = Tensor((batches * height * width, channels), [(i * 53 % 256) - 128 for i in range(count)], "i8")
    readout = tensor.requant_acc_scale(scale).to_i8()
    expected = readout.maxpool2d_rows(in_dims=source.input_hw, pool_size=source.window,
                                    pool_stride=source.stride, pool_padding=source.padding, pad_value=-128)
    actual = [None] * len(expected.data)
    cells = {}
    calls = 0
    for call, env in instances(plan.kernel):
        args = {k: evaluate(v, env) for k, v in call.args}
        calls += 1
        first, columns = args["first_column"], args["columns"]
        if call.instr == "stage_spatial_rows":
            for row in range(args["rows"]):
                source_row = args["source_row"] + row
                local = args["local_row"] + row
                assert 0 <= source_row < batches * height * width
                assert 0 <= local < plan.temporary_rows
                assert local not in cells  # no duplicate private write in this lifetime
                cells[local] = readout.data[source_row * channels + first:source_row * channels + first + columns]
        else:
            assert set(cells) == set(range(plan.temporary_rows))
            for x in range(source.output_hw[1]):
                # Decode the selected reader's source-independent rectangular
                # address equation; never recover indices from requested loads.
                rows = [ky * plan.stripe_hw[1] + x * source.stride[1] + kx
                        for ky in range(3) for kx in range(3)]
                for col in range(columns):
                    offset = (args["output_row"] + x) * channels + first + col
                    assert actual[offset] is None
                    actual[offset] = max(cells[r][col] for r in rows)
            cells.clear()
    assert actual == expected.data
    assert not cells
    assert calls == plan.dynamic_calls
    assert plan.revalidate()["physical_mapping_verified"] is False


def test_large_shape_stays_counted_and_retains_complete_read_windows():
    source = SpatialMaxReadout((112, 112), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    plan = plan_striped_spatial_max(source, batches=1, channels=64, channel_block=16,
                                   transfer_rows=16, capacity_rows=1024)
    assert plan.stripe_hw == (3, 113)
    assert plan.temporary_rows == 339
    assert plan.dynamic_calls == 5600  # 5376 staged LOAD calls + 224 complete strip readouts
    assert len(plan.kernel.text()) < 10000


def test_changed_actual_program_and_capacity_refuse():
    source = SpatialMaxReadout((5, 5), (3, 3), (2, 2), (1, 1, 1, 1), -128)
    plan = plan_striped_spatial_max(source, batches=1, channels=3, channel_block=16,
                                   transfer_rows=16, capacity_rows=64)
    with pytest.raises(ValueError, match="original source"):
        replace(plan, kernel=replace(plan.kernel, body=())).revalidate()
    # Move the real first border staging destination one row: it now aliases
    # the following contiguous transfer and leaves the first reader cell empty.
    batch = plan.kernel.body[0]
    y = batch.body[0]
    channel = y.body[0]
    border = channel.body[0]
    call = border.body[0]
    changed = replace(call, args=tuple((k, v + 1 if k == "local_row" else v) for k, v in call.args))
    border = replace(border, body=(changed,))
    channel = replace(channel, body=(border, *channel.body[1:]))
    y = replace(y, body=(channel,))
    batch = replace(batch, body=(y,))
    with pytest.raises(ValueError, match="original source"):
        replace(plan, kernel=replace(plan.kernel, body=(batch,))).revalidate()
    with pytest.raises(ValueError, match="capacity"):
        replace(plan, capacity_rows=1).revalidate()
    with pytest.raises(ValueError, match="minimum-valued"):
        plan_striped_spatial_max(replace(source, pad_value=0), batches=1, channels=3,
                                channel_block=16, transfer_rows=16, capacity_rows=64)

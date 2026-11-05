"""Original window words survive tiled DMA staging and rolling row reuse."""
from dataclasses import replace

import numpy as np
import pytest
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    emit_convolution_schedule,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionResidentInputRows,
    CountedConvolutionPlan,
    Window2D,
)


def layout(window, identity='a'*64):
    span = (window.kernel[0]-1)*window.dilation[0]+1
    pitch = window.input_shape[2]+window.padding[1]+window.padding[3]+2
    return ConvolutionResidentInputRows(window, identity, 4, 16, pitch,
                                        span, 16, 2048, 4, halo_value=0 if any(window.padding) else None)


@pytest.mark.parametrize('shape,kernel,stride,dilation,padding', [
    ((2, 5, 9, 7), (3, 3), (1, 1), (1, 1), (0, 0, 0, 0)),
    ((1, 9, 13, 3), (2, 3), (2, 1), (2, 2), (0, 0, 0, 0)),
    ((1, 6, 8, 8), (1, 1), (3, 1), (1, 1), (0, 0, 0, 0)),
    ((1, 5, 7, 4), (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)),
    ((2, 7, 9, 7), (3, 3), (1, 1), (1, 1), (2, 1, 0, 2)),
    ((1, 4, 8, 3), (2, 2), (2, 1), (2, 2), (2, 3, 1, 0)),
    ((1, 7, 11, 7), (3, 3), (2, 2), (1, 1), (0, 0, 0, 0)),
    ((2, 7, 13, 7), (3, 3), (2, 2), (1, 1), (1, 2, 1, 0)),
    ((1, 9, 17, 3), (2, 3), (2, 3), (2, 2), (2, 1, 0, 3)),
    ((1, 9, 11, 5), (7, 7), (1, 1), (1, 1), (3, 3, 3, 3)),
    ((2, 10, 13, 3), (7, 7), (2, 2), (1, 1), (3, 3, 3, 3)),
])
@pytest.mark.parametrize('stage_channels', [None, 4, 8])
def test_complete_source_products_match_after_tiled_staging_and_ring_reuse(
    shape, kernel, stride, dilation, padding, stage_channels
):
    w = Window2D(shape, kernel, stride=stride, dilation=dilation, padding=padding)
    p = CountedConvolutionPlan(w, 5, 4, min(4, w.output_spatial[1]), 4, 4, 4096, 8, 0, 8, 0, True)
    source = ConvolutionExplorationSource(p, ((0, 4), (4, 1)), 'original-source')
    if stride[1] == 1:
        resident = layout(w, source.identity)
    else:
        # An actual consumer must select the same source row step explicitly.
        span = (kernel[0]-1)*dilation[0]+1
        pitch = shape[2]+padding[1]+padding[3]+2
        resident = ConvolutionResidentInputRows(w, source.identity, 4, 16, pitch,
            span, 16, 2048, 4, halo_value=0 if any(padding) else None, compute_row_step=stride[1])
    resident = replace(resident, stage_channels=stage_channels)
    schedule = plan_convolution_schedule(source)
    image = (np.arange(np.prod(shape)) % 7-3).reshape(shape).astype(np.int8)
    weight = (np.arange(np.prod(kernel)*shape[-1]*5) % 5-2).reshape(-1, 5).astype(np.int8)
    bias = np.array([9, -3, 8, 2, -11], dtype=np.int32)
    sp = {}
    acc = {}
    written = {}
    events = []
    staged = 0

    def lower(kind, a):
        nonlocal staged
        if kind == 'stage_input_rows':
            assert a['columns'] <= (stage_channels or shape[-1])
            # Actual rectangular tiled-load map: columns form channel blocks,
            # rows advance x, each block has its explicit local row pitch.
            staged += a['rows']*a['columns']
            events.append('stage')
            for row in range(a['rows']):
                for col in range(a['columns']):
                    first = a['first_element']+row*a['row_stride_elements']+col
                    assert 0 <= first < image.size
                    cell = (a['local_row']+(col//4)*a['block_stride_rows']+row, col % 4)
                    sp[cell] = int(image.ravel()[a['first_element']+row*a['row_stride_elements']+col])
        elif kind == 'fill_input_rows':
            assert a['value'] == 0
            events.append('stage')
            for row in range(a['rows']):
                for col in range(a['columns']):
                    sp[a['local_row']+(col//4)*a['block_stride_rows']+row, col % 4] = a['value']
        elif kind == 'initialize_bias':
            acc[a['accumulator_row']] = np.broadcast_to(bias[a['output_column']:a['output_column']+a['columns']],
                                                        (a['rows'], a['columns'])).copy()
        elif kind == 'load_weight':
            acc['rhs'] = weight[a['reduction_row']:a['reduction_row']+a['rows'],
                                a['output_column']:a['output_column']+a['columns']]
        elif kind == 'compute':
            assert events[-1] == 'ready'
            assert a.get('lhs_row_step', 1) == stride[1]
            lhs = np.array([[sp[a['lhs_row']+r*a.get('lhs_row_step', 1), c]
                             for c in range(a['depth'])] for r in range(a['rows'])])
            acc[a['accumulator_row']] += lhs.astype(np.int64) @ acc['rhs'].astype(np.int64)
        elif kind == 'store':
            for r in range(a['rows']):
                for c in range(a['columns']):
                    key = a['output_row']+r, a['output_column']+c
                    assert key not in written
                    written[key] = acc[a['accumulator_row']][r, c]
        else:
            raise AssertionError(kind)

    def complete(kind, fields):
        events.append('ready' if kind == 'input_ready' else 'done')

    result = emit_convolution_schedule(schedule, Integers(), lower, complete=complete,
                                      synchronous_completion=True, input_rows=resident)
    oh, ow = w.output_spatial
    expected = np.broadcast_to(bias, (shape[0], oh, ow, 5)).astype(np.int64).copy()
    weights = weight.reshape(*kernel, shape[-1], 5).astype(np.int64)
    top, left, bottom, right = padding
    padded = np.pad(image, ((0, 0), (top, bottom), (left, right), (0, 0)))
    for ky in range(kernel[0]):
        for kx in range(kernel[1]):
            sample = padded[:, ky*dilation[0]:ky*dilation[0]+oh*stride[0]:stride[0],
                            kx*dilation[1]:kx*dilation[1]+ow*stride[1]:stride[1], :]
            expected += sample.astype(np.int64) @ weights[ky, kx]
    actual = np.array([written[r, c] for r in range(shape[0]*oh*ow) for c in range(5)]).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    # One initial span plus only the original frontier per output y; no K/N reload.
    span = (kernel[0]-1)*dilation[0]+1
    staged_y = list(range(span))
    for y in range(1, oh):
        staged_y.extend(range(y*stride[0]+span-min(span, stride[0]), y*stride[0]+span))
    assert staged == shape[0]*sum(0 <= y-padding[0] < shape[1] for y in staged_y)*shape[2]*shape[-1]
    assert result['rolling_transport']['consumers'] == shape[0]*oh
    assert events[-1] == 'done'


def test_injective_source_cells_padding_holes_and_selected_bounds():
    r = layout(Window2D((1, 5, 9, 7), (3, 3)))
    pitch = r.rows_per_source_row*r.channel_tile
    cells = {r.physical_cell(s*pitch+x*7+c) for s in range(r.slots) for x in range(9) for c in range(7)}
    assert len(cells) == r.slots*9*7
    with pytest.raises(ValueError, match='padding hole'):
        r.physical_cell(9*7)
    for update in ({'slots': 1}, {'private_rows': 1}, {'base': 0}, {'row_pitch': 8}, {'channel_tile': True}):
        with pytest.raises(ValueError):
            replace(r, **update)
    for w in (Window2D((1, 5, 9, 7), (3, 3), padding=(1, 1, 1, 1)),
              Window2D((1, 5, 9, 7), (3, 3), stride=(1, 2))):
        with pytest.raises(ValueError, match='halo value|horizontal stride'):
            replace(r, window=w)


@pytest.mark.parametrize('step', [None, False, 0, -1, 1, 3, 2.0, '2'])
def test_nonunit_source_sampling_requires_exact_explicit_consumer_step(step):
    window = Window2D((1, 7, 13, 7), (3, 3), stride=(2, 2), padding=(1, 1, 1, 1))
    with pytest.raises(ValueError, match='compute row-step'):
        ConvolutionResidentInputRows(window, 'a'*64, 4, 16, 16, 3, 16, 1024, 4,
                                    halo_value=0, compute_row_step=step)


def test_explicit_step_cannot_rebind_existing_source_or_mutate_after_selection():
    original = layout(Window2D((1, 5, 9, 7), (3, 3)))
    with pytest.raises(ValueError, match='compute row-step'):
        replace(original, compute_row_step=2)
    selected = replace(original, compute_row_step=1)
    assert selected.row_step == original.row_step == 1
    object.__setattr__(selected, 'compute_row_step', True)
    with pytest.raises(ValueError, match='compute row-step'):
        selected.revalidate()


def test_original_binding_real_completion_and_rhs_overlap_refuse():
    w = Window2D((1, 5, 9, 7), (3, 3))
    p = CountedConvolutionPlan(w, 4, 4, 4, 4, 4, 4096, 8, 0, 8, 0)
    s = ConvolutionExplorationSource(p, ((0, 4),), 'original')
    schedule = plan_convolution_schedule(s)
    r = layout(w, s.identity)
    for update in ({'source_identity': 'b'*64}, {'base': 8, 'private_first': 0}):
        with pytest.raises(ValueError):
            emit_convolution_schedule(schedule, Integers(), lambda *_: None, complete=lambda *_: None,
                                      synchronous_completion=True, input_rows=replace(r, **update))
    with pytest.raises(ValueError, match='real completion'):
        emit_convolution_schedule(schedule, Integers(), lambda *_: None,
                                  synchronous_completion=True, input_rows=r)
    object.__setattr__(r, 'row_pitch', 12)
    with pytest.raises(ValueError, match='changed'):
        r.revalidate()


def test_padding_requires_original_zero_value_and_complete_physical_halo():
    w = Window2D((1, 5, 9, 7), (3, 3), padding=(1, 2, 1, 3))
    r = layout(w)
    assert r.padded_width == 14
    assert r.physical_cell(0) == (r.base+2, 0)
    for value in (None, False, 1, -3, 0.0):
        with pytest.raises(ValueError, match='halo value'):
            replace(r, halo_value=value)
    with pytest.raises(ValueError, match='footprint'):
        replace(r, row_pitch=13)


def test_transfer_column_limit_is_explicit_aligned_and_source_bound():
    r = layout(Window2D((1, 5, 9, 7), (3, 3), padding=(1, 1, 1, 1)))
    for limit in (False, 0, -4, 3, 6):
        with pytest.raises(ValueError, match='whole-channel-block'):
            replace(r, stage_channels=limit)
    assert replace(r, stage_channels=4).initial_load_commands == r.initial_load_commands*2
    assert replace(r, stage_channels=8).initial_load_commands == r.initial_load_commands


@pytest.mark.parametrize('channel_tile', [0, '4', None, True, 4.0])
def test_malformed_channel_tile_refuses_before_staging_modulo(channel_tile):
    r = layout(Window2D((1, 5, 9, 7), (3, 3)))
    with pytest.raises(ValueError, match='footprint'):
        replace(r, channel_tile=channel_tile, stage_channels=4)

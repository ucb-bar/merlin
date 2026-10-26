"""Source input hoisting keeps independent ACC liveouts and exact K order."""
from dataclasses import replace

import numpy as np
import pytest
from test_convolution_load_issue import law as load_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.window_transfer import (
    ConvolutionInputRetentionLaw,
    ConvolutionOutputTile,
    CountedConvolutionPlan,
    Window2D,
    emit_convolution_column_block,
    emit_convolution_load_issue_column_block,
    emit_convolution_tile,
)


def law():
    return ConvolutionInputRetentionLaw("neutral", "source-i32", "original", "conditional", 3,
        input_preserved=True, independent_outputs=True, immutable_mapped_disjoint_buffers=True)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("issued", [False, True])
def test_shared_inputs_preserve_padded_tail_outputs_and_order(bias, issued):
    window = Window2D((1, 3, 5, 5), (3, 2), padding=(1, 1, 1, 0))
    p = CountedConvolutionPlan(window, 9, 4, 4, 4, 3, 32, 12, 0, 8, 0, bias)
    tiles = tuple(ConvolutionOutputTile(n, min(4, 9-n), slot*4) for slot, n in enumerate(range(0, 9, 4)))
    image = (np.arange(75) % 17 - 8).astype(np.int8).reshape(window.input_shape)
    weights = (np.arange(270) % 13 - 6).astype(np.int8).reshape(6*5, 9)
    biases = np.array([2**31-1, -2**31, -9, 0, 7, 1, 2, 3, 4], dtype=np.int32)
    oh, ow = window.output_spatial
    outputs = np.zeros((oh*ow, 9), dtype=np.int32)
    writes = np.zeros_like(outputs)
    lhs = np.zeros((4, 3), dtype=np.int8)
    acc = {}
    rhs = None
    counts = {"retained": 0, "original": 0}
    original_order, actual_order = {}, {}

    def lower(kind, a):
        nonlocal rhs
        if kind == "initialize_bias":
            acc[a['accumulator_row']] = np.broadcast_to(
                biases[a['output_column']:a['output_column']+a['columns']], (a['rows'], a['columns'])).copy()
        elif kind == "load_input":
            counts['retained'] += 1
            for r in range(a['rows']):
                start = a['first_element'] + r*a['row_stride_elements']
                lhs[a['local_row']+r, :a['columns']] = 0 if a['zero'] else image.ravel()[start:start+a['columns']]
        elif kind == "load_weight":
            rhs = weights[a['reduction_row']:a['reduction_row']+a['rows'],
                          a['output_column']:a['output_column']+a['columns']]
            actual_order.setdefault(a['output_column'], []).append(a['reduction_row'])
        elif kind == "compute":
            value = lhs[:a['rows'], :a['depth']].astype(np.int32) @ rhs.astype(np.int32)
            key = a['accumulator_row']
            acc[key] = value if a['first_reduction'] and not a['bias'] else acc[key] + value
        elif kind == "store":
            r, c = a['output_row'], a['output_column']
            outputs[r:r+a['rows'], c:c+a['columns']] = acc[a['accumulator_row']]
            writes[r:r+a['rows'], c:c+a['columns']] += 1
        else:
            raise AssertionError(kind)

    def original(kind, a):
        if kind == 'load_input':
            counts['original'] += 1
        elif kind == 'load_weight':
            original_order.setdefault(a['output_column'], []).append(a['reduction_row'])

    pending = []
    drains = []

    def issue(kind, fields):
        if len(pending) == 2:
            lower(*pending.pop(0))
        pending.append((kind, dict(fields)))

    def complete(kind, fields):
        assert kind in ('compute', 'store') and pending[-1][0] == kind
        while pending:
            lower(*pending.pop(0))
        drains.append(kind)

    for y in range(oh):
        for x in range(0, ow, 4):
            row, rows = y*ow+x, min(4, ow-x)
            for t in tiles:
                emit_convolution_tile(p, Integers(), original, output_row=row, rows=rows,
                    output_column=t.output_column, columns=t.columns, accumulator_row=t.accumulator_row,
                    synchronous_completion=True)
            if issued:
                report = emit_convolution_load_issue_column_block(p, Integers(), issue, complete,
                    output_row=row, rows=rows, output_tiles=tiles, law=law(), issue_law=load_law())
                assert not pending
            else:
                report = emit_convolution_column_block(p, Integers(), lower, output_row=row, rows=rows,
                    output_tiles=tiles, law=law(), synchronous_completion=True)
            assert report['timing'] == 'UNKNOWN'
    expected = np.zeros_like(outputs)
    for y in range(oh):
        for x in range(ow):
            value = biases.astype(np.int64).copy() if bias else np.zeros(9, dtype=np.int64)
            for ky in range(3):
                for kx in range(2):
                    iy, ix = y+ky-1, x+kx-1
                    if 0 <= iy < 3 and 0 <= ix < 5:
                        rhs = weights.reshape(3, 2, 5, 9)[ky, kx].astype(np.int64)
                        value += image[0, iy, ix].astype(np.int64) @ rhs
            expected[y*ow+x] = value.astype(np.int32)
    assert np.array_equal(outputs, expected) and np.all(writes == 1)
    assert actual_order == original_order
    assert counts['original'] == 3*counts['retained']


@pytest.mark.parametrize('tiles,change', [
    ((ConvolutionOutputTile(0, 4, 0), ConvolutionOutputTile(4, 4, 3)), {}),
    ((ConvolutionOutputTile(0, 4, 0), ConvolutionOutputTile(5, 4, 4)), {}),
    ((ConvolutionOutputTile(0, 4, 10),), {}),
    ((ConvolutionOutputTile(0, 4, 0),), {'input_preserved': False}),
])
def test_bad_live_storage_or_missing_retention_refuses_before_effects(tiles, change):
    p = CountedConvolutionPlan(Window2D((1, 2, 4, 3), (1, 1)), 9, 4, 4, 4, 3, 32, 12, 0, 8, 0)
    seen = []
    with pytest.raises(ValueError):
        emit_convolution_column_block(p, Integers(), lambda *x: seen.append(x), output_row=0, rows=4,
            output_tiles=tiles, law=replace(law(), **change), synchronous_completion=True)
    assert seen == []


def test_composed_epoch_counts_all_first_bias_loads_and_retains_sync_guard():
    p = CountedConvolutionPlan(Window2D((1, 2, 4, 3), (1, 1)), 9, 4, 4, 4, 3, 32, 12, 0, 8, 0, True)
    tiles = tuple(ConvolutionOutputTile(n, min(4, 9-n), i*4) for i, n in enumerate(range(0, 9, 4)))
    seen = []
    with pytest.raises(ValueError, match='logical epoch budget'):
        emit_convolution_load_issue_column_block(p, Integers(), lambda *a: seen.append(a), lambda *a: None,
            output_row=0, rows=4, output_tiles=tiles, law=law(), issue_law=replace(load_law(), load_issue_budget=7))
    with pytest.raises(ValueError, match='complete source dependencies'):
        emit_convolution_load_issue_column_block(p, Integers(), lambda *a: seen.append(a), lambda *a: None,
            output_row=0, rows=4, output_tiles=tiles, law=law(),
            issue_law=replace(load_law(), complete_device_drain=False))
    with pytest.raises(ValueError, match='synchronous realization'):
        emit_convolution_column_block(p, Integers(), lambda *a: seen.append(a), output_row=0, rows=4,
            output_tiles=tiles, law=law(), synchronous_completion=False)
    assert seen == []

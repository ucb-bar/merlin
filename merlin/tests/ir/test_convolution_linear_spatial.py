"""Linear valid windows preserve exact samples, reduction order and completion."""
from dataclasses import replace

import numpy as np
import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_compute_epochs import epoch_law
from test_convolution_exploration import issue_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionScheduleChoice,
    emit_convolution_schedule,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan,
    Window2D,
    emit_convolution_tile,
)


class SelectIntegers(Integers):
    def select(self, condition, yes, no):
        return yes if condition else no


def source(shape, kernel, stride=(1, 1), dilation=(1, 1)):
    window = Window2D(shape, kernel, stride, dilation)
    p = CountedConvolutionPlan(window, 5, 4, min(4, window.output_spatial[1]),
                               4, 3, 32, 8, 0, 8, 0, True)
    return ConvolutionExplorationSource(p, ((0, 4), (4, 1)), 'original-valid-source')


def run(s, mode, iteration):
    p = s.plan
    schedule = plan_convolution_schedule(s, ConvolutionScheduleChoice(2, mode != 'sync', 4 if mode == 'epoch' else 1),
        retention_law=retention_law(), issue_law=issue_law() if mode != 'sync' else None,
        epoch_law=epoch_law() if mode == 'epoch' else None)
    image = (np.arange(np.prod(p.window.input_shape)) % 9 - 4).astype(np.int8)
    weights = (np.arange(np.prod(p.window.kernel)*p.window.input_shape[-1]*5) % 7 - 3).astype(np.int8).reshape(-1, 5)
    bias = np.arange(5, dtype=np.int32) - 2
    oh, ow = p.window.output_spatial
    output = np.zeros((p.window.input_shape[0]*oh*ow, 5), dtype=np.int32)
    written = np.zeros_like(output)
    lhs = np.zeros((p.row_tile, 3), dtype=np.int8)
    rhs, accumulators, events = None, {}, []

    def lower(kind, a):
        nonlocal rhs
        events.append((kind, dict(a)))
        if kind == 'initialize_bias':
            accumulators[a['accumulator_row']] = np.broadcast_to(
                bias[a['output_column']:a['output_column']+a['columns']], (a['rows'], a['columns'])).copy()
        elif kind == 'load_input':
            assert not a['zero'] and a['rows'] > 0
            for row in range(a['rows']):
                first = a['first_element'] + row*a['row_stride_elements']
                assert 0 <= first <= image.size-a['columns']
                lhs[a['local_row']+row, :a['columns']] = image[first:first+a['columns']]
        elif kind == 'load_weight':
            rhs = weights[a['reduction_row']:a['reduction_row']+a['rows'],
                          a['output_column']:a['output_column']+a['columns']]
        elif kind == 'compute':
            accumulators[a['accumulator_row']] += lhs[:a['rows'], :a['depth']].astype(np.int32) @ rhs.astype(np.int32)
        elif kind == 'store':
            r, c = a['output_row'], a['output_column']
            output[r:r+a['rows'], c:c+a['columns']] = accumulators[a['accumulator_row']]
            written[r:r+a['rows'], c:c+a['columns']] += 1
        else:
            raise AssertionError(kind)

    result = emit_convolution_schedule(schedule, SelectIntegers(), lower,
        complete=lambda k, a: events.append(('drain', {'kind': k})), synchronous_completion=True,
        spatial_iteration=iteration)
    assert np.all(written == 1)
    if mode != 'sync':
        for index, (kind, _) in enumerate(events):
            if kind == 'store':
                assert events[index+1] == ('drain', {'kind': 'store'})
    return output, events, result


@pytest.mark.parametrize('geometry', [
    ((2, 5, 8, 5), (3, 2), (1, 1), (1, 1)),  # cross-row, image and channel tails
    ((2, 8, 17, 5), (2, 3), (2, 2), (2, 1)),
    ((1, 3, 4, 3), (1, 1), (1, 1), (1, 1)),  # aligned incumbent
])
@pytest.mark.parametrize('mode', ['sync', 'issue', 'epoch'])
def test_actual_effects_preserve_all_outputs_and_ordered_reduction(geometry, mode):
    s = source(*geometry)
    original, rows, _ = run(s, mode, 'rows')
    actual, linear, receipt = run(s, mode, 'linear')
    assert np.array_equal(actual, original)
    assert receipt['spatial_iteration'] == 'linear'
    # Every original output has the identical sequence of K chunks and columns,
    # independent of how output rows are tiled together.
    def per_output(events):
        traces = {}
        for kind, a in events:
            if kind != 'compute':
                continue
            for row in range(a['rows']):
                for column in range(a['columns']):
                    key = (a['output_row']+row, a['output_column']+column)
                    traces.setdefault(key, []).append((a['depth'], bool(a['first_reduction']), a['bias']))
        return traces
    assert per_output(rows) == per_output(linear)
    assert sum(k == 'compute' for k, _ in linear) <= sum(k == 'compute' for k, _ in rows)


def test_defaults_preserve_original_event_bytes_and_refuse_unproven_windows():
    s = source((2, 5, 8, 3), (3, 2))
    plan = plan_convolution_schedule(s)
    original, explicit = [], []
    emit_convolution_schedule(plan, Integers(), lambda k, a: original.append((k, a)), synchronous_completion=True)
    emit_convolution_schedule(plan, Integers(), lambda k, a: explicit.append((k, a)),
                              synchronous_completion=True, spatial_iteration='rows')
    assert original == explicit
    for bad in ('guess', True, None):
        with pytest.raises(ValueError, match='spatial iteration'):
            emit_convolution_schedule(plan, SelectIntegers(), lambda *_: None,
                                      synchronous_completion=True, spatial_iteration=bad)
    with pytest.raises(ValueError, match='select builder'):
        emit_convolution_schedule(plan, Integers(), lambda *_: None,
                                  synchronous_completion=True, spatial_iteration='linear')
    padded = replace(s.plan, window=replace(s.plan.window, padding=(1, 0, 0, 0)))
    p = plan_convolution_schedule(replace(s, plan=padded))
    with pytest.raises(ValueError, match='unpadded'):
        emit_convolution_schedule(p, SelectIntegers(), lambda *_: None,
                                  synchronous_completion=True, spatial_iteration='linear')
    # Low-level callers cannot bind a convenient but different source geometry.
    gather = ConvolutionGatherRuns((1, 5, 8, 3), (1, 3, 7), (1, 1), (1, 1), (3, 2), 4)
    with pytest.raises(ValueError, match='exact original valid window'):
        emit_convolution_tile(s.plan, SelectIntegers(), lambda *_: None,
            output_row=0, output_column=0, rows=4, columns=4, accumulator_row=0,
            synchronous_completion=True, gather_runs=gather)

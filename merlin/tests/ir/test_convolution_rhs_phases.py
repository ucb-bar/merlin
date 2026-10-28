"""Capacity-derived phases preserve every original output and row lifetime."""
from dataclasses import replace

import numpy as np
import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_compute_epochs import epoch_law
from test_convolution_exploration import issue_law
from test_convolution_rhs_residency import rhs_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionRhsPhaseLaw,
    ConvolutionRhsPhases,
    ConvolutionScheduleChoice,
    emit_convolution_schedule,
    plan_convolution_rhs_phases,
    plan_convolution_rhs_residency,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionResidentInputRows,
    CountedConvolutionPlan,
    Window2D,
)


def fixture():
    w = Window2D((2, 5, 7, 5), (3, 3), padding=(1, 1, 1, 1))
    p = CountedConvolutionPlan(w, 9, 4, 4, 4, 4, 1024, 12, 0, 8, 0, True)
    source = ConvolutionExplorationSource(p, ((0, 4), (4, 4), (8, 1)), 'original-source-all-exits')
    schedule = plan_convolution_schedule(source, ConvolutionScheduleChoice(3, True, 4),
        retention_law=retention_law(), issue_law=replace(issue_law(), load_issue_budget=64), epoch_law=epoch_law())
    rows = ConvolutionResidentInputRows(w, source.identity, 4, 16, 16, 3, 16, 96, 4,
                                      halo_value=0, stage_channels=4)
    law = ConvolutionRhsPhaseLaw(source.identity, 'actual-whole-invocation-disjoint-output-proof',
        independent_output_columns=True, immutable_inputs=True, no_intermediate_observers=True,
        full_completion_before_reprepare=True)
    return schedule, rows, law


@pytest.mark.parametrize('capacity', [512, 256, 176])
@pytest.mark.parametrize('column_bound', [None, 1, 2])
def test_full_fit_phases_and_fallback_execute_all_original_outputs(capacity,column_bound):
    schedule, rows, law = fixture()
    rhs = plan_convolution_rhs_phases(schedule, operand_rows=capacity, law=rhs_law(), phase_law=law,
                                      input_rows=rows, max_column_blocks=column_bound)
    phases = len(rhs.phases) if type(rhs) is ConvolutionRhsPhases else 1
    assert phases == (1 if capacity==176 else (3 if column_bound==1 else
                      2 if capacity==256 or column_bound==2 else 1))
    assert (rhs is None) == (capacity == 176)
    image = (np.arange(350) % 11-5).astype(np.int8).reshape(2, 5, 7, 5)
    weights = (np.arange(405) % 7-3).astype(np.int8).reshape(45, 9)
    biases = np.arange(9, dtype=np.int32)-4
    sp, rhs_sources, acc, written, products = {}, {}, {}, {}, {}
    pending, boundaries = [], []
    staged_words = 0

    def execute(kind, a):
        nonlocal staged_words
        if kind in ('stage_input_rows', 'fill_input_rows'):
            for row in range(a['rows']):
                for col in range(a['columns']):
                    cell = a['local_row']+(col//4)*a['block_stride_rows']+row, col % 4
                    if kind == 'fill_input_rows':
                        assert a['value'] == 0
                        sp[cell] = 0
                    else:
                        first = a['first_element']+row*a['row_stride_elements']+col
                        assert 0 <= first < image.size
                        sp[cell] = int(image.ravel()[first])
                        staged_words += 1
        elif kind == 'load_weight':
            for row in range(a['rows']):
                rhs_sources[a['local_row']+row] = a['reduction_row']+row, a['output_column']
                for col in range(a['columns']):
                    sp[a['local_row']+row, col] = int(weights[a['reduction_row']+row, a['output_column']+col])
        elif kind == 'initialize_bias':
            acc[a['accumulator_row']] = np.broadcast_to(biases[a['output_column']:a['output_column']+a['columns']],
                                                        (a['rows'], a['columns'])).astype(np.int64).copy()
        elif kind == 'compute':
            reduction, column = rhs_sources[a['rhs_row']]
            assert column == a['output_column']
            products.setdefault((a['output_row'], column), []).append(reduction)
            lhs = np.array([[sp[a['lhs_row']+r, c] for c in range(a['depth'])] for r in range(a['rows'])])
            weight = np.array([[sp[a['rhs_row']+r, c] for c in range(a['columns'])] for r in range(a['depth'])])
            acc[a['accumulator_row']] += lhs @ weight
        elif kind == 'store':
            for row in range(a['rows']):
                for col in range(a['columns']):
                    key = a['output_row']+row, a['output_column']+col
                    assert key not in written
                    written[key] = acc[a['accumulator_row']][row, col]
        else:
            raise AssertionError(kind)

    def issue(kind, fields):
        if len(pending) == 2:
            execute(*pending.pop(0))
        pending.append((kind, fields))

    def complete(kind, fields):
        while pending:
            execute(*pending.pop(0))
        boundaries.append(kind)

    result = emit_convolution_schedule(schedule, Integers(), issue, complete=complete, rhs=rhs, input_rows=rows)
    assert not pending
    expected = np.broadcast_to(biases, (2, 5, 7, 9)).astype(np.int64).copy()
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1), (0, 0)))
    for ky in range(3):
        for kx in range(3):
            expected += padded[:, ky:ky+5, kx:kx+7, :].astype(np.int64) @ weights.reshape(3, 3, 5, 9)[ky, kx]
    actual = np.array([written[r, c] for r in range(70) for c in range(9)]).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    assert staged_words == image.size*phases
    original_k_order = [tap*5+channel for tap in range(9) for channel in (0, 4)]
    assert all(order == original_k_order for order in products.values())
    if phases > 1:
        assert boundaries.count('weight_preparation') == boundaries.count('rhs_phase_last_use') == phases
        assert result['rhs_phases']['coverage'].startswith('each original output column exactly once')
    elif rhs is not None:
        assert rhs == plan_convolution_rhs_residency(schedule, operand_rows=capacity, law=rhs_law(), rhs_base=112)
        assert 'rhs_phases' not in result


def test_explicit_phase_bound_retains_real_capacity_and_requires_independent_columns():
    s,rows,law=fixture()
    def make(bound,phase_law=law):
        return plan_convolution_rhs_phases(s,operand_rows=512,law=rhs_law(),phase_law=phase_law,
                                          input_rows=rows,max_column_blocks=bound)
    full=make(None)
    assert full==make(3)==make(9)
    narrow=make(2)
    assert [p.column_intervals for p in narrow.phases]==[((0,4),(4,4)),((8,1),)]
    assert all(p.layout.operand_rows==512 and p.layout.rhs_base==112 for p in narrow.phases)
    assert narrow.summary()['max_column_blocks']==2
    with pytest.raises(ValueError,match='separate'):make(2,None)
    for invalid in (0,-1,True,1<<63):
        with pytest.raises(ValueError,match='bound'):make(invalid)
    with pytest.raises(ValueError,match='bounded'):
        replace(narrow,max_column_blocks=1)


def test_exact_partition_source_permission_and_capacity_limits():
    s, rows, law = fixture()
    def make(**kw):
        return plan_convolution_rhs_phases(s, operand_rows=256, law=rhs_law(), input_rows=rows, **kw)
    rhs = make(phase_law=law)
    assert [p.column_intervals for p in rhs.phases] == [((0, 4), (4, 4)), ((8, 1),)]
    assert all(p.layout.rhs_base == 112 for p in rhs.phases)
    for changes in ({'phases': rhs.phases[::-1]}, {'phases': (rhs.phases[0], rhs.phases[0])},
                    {'law': replace(law, no_intermediate_observers=False)}):
        with pytest.raises(ValueError):
            replace(rhs, **changes)
    with pytest.raises(ValueError, match='separate'):
        make()
    with pytest.raises(ValueError, match='premises'):
        make(phase_law=replace(law, source_identity='foreign'))
    with pytest.raises(ValueError, match='overlap'):
        make(phase_law=law, rhs_base=96)
    # No single complete-K source interval fits: the unchanged ordinary route.
    assert plan_convolution_rhs_phases(s, operand_rows=176, law=rhs_law(), input_rows=rows) is None
    with pytest.raises(ValueError, match='partial RHS'):
        emit_convolution_schedule(s, Integers(), lambda *_: pytest.fail('partial emission'),
                                  complete=lambda *_: None, rhs=rhs.phases[0], input_rows=rows)
    with pytest.raises(ValueError, match='original input'):
        emit_convolution_schedule(s, Integers(), lambda *_: pytest.fail('wrong input'),
                                  complete=lambda *_: None, rhs=rhs)
    object.__setattr__(law, 'qualification_identity', 'changed')
    with pytest.raises(ValueError, match='changed'):
        rhs.verify_current()


def test_phase_registry_is_executable_and_capacity_fallback_deduplicates():
    import json

    from merlin.compile.scheduling.convolution_exploration import convolution_pass_space, explore_convolution_schedules
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_state import PipelineContext

    s, rows, law = fixture()
    def realize(schedule, *, rhs):
        events = []
        emit_convolution_schedule(schedule, Integers(), lambda k, a: events.append((k, a)),
            complete=lambda k, a: events.append(('drain', k)), rhs=rhs, input_rows=rows)
        return EmittedProgram(json.dumps(events, sort_keys=True).encode(), 'original-source-effects',
                              schedule.identity if rhs is None else rhs.identity)

    space = convolution_pass_space(s.source, verify_source=lambda _: None, realize=realize,
        retention_law=s.retention_law, issue_law=s.issue_law, epoch_law=s.epoch_law,
        live_tile_limits=(3,), issue_modes=(True,), compute_pair_limits=(4,), rhs_modes=(False, True, 'phased'),
        rhs_law=rhs_law(), private_operand_rows=256, rhs_base=112, rhs_phase_law=law, input_rows=rows,
        max_candidates=3)
    results = list(explore_convolution_schedules(s.source, space, context=PipelineContext()))
    assert len(results) == 3 and all(r.evaluation.status == 'compiled' for r in results)
    assert len({r.emission.data for r in results}) == 2
    assert sum(r.duplicate_of is not None for r in results) == 1

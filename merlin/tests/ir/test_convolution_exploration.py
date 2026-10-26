"""Reusable spatial registry consumes actual counted effects and source custody."""
import hashlib
import json
from dataclasses import replace

import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_load_issue import law as base_issue_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionScheduleChoice,
    convolution_pass_space,
    emit_convolution_schedule,
    explore_convolution_schedules,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan,
    Window2D,
    emit_counted_convolution,
)
from merlin.transforms.pass_exploration import EmittedProgram
from merlin.transforms.pipeline_state import PipelineContext


def issue_law():
    retained = retention_law()
    return replace(base_issue_law(), hardware_identity=retained.hardware_identity,
                   numerical_policy_identity=retained.numerical_policy_identity)


def source():
    p = CountedConvolutionPlan(Window2D((1, 3, 5, 3), (3, 2), padding=(1, 1, 1, 0)),
                               9, 4, 4, 4, 3, 32, 8, 0, 8, 0)
    return ConvolutionExplorationSource(p, ((0, 4), (4, 4), (8, 1)), 'original-signed-window')


def record(schedule):
    events = []
    emit_convolution_schedule(schedule, Integers(), lambda k, a: events.append((k, a)),
        complete=lambda k, a: events.append(('drain', dict(kind=k))), synchronous_completion=True)
    return json.dumps(events, sort_keys=True).encode()


def test_original_whole_invocation_and_capacity_fallback():
    s = source()
    baseline = plan_convolution_schedule(s)
    events = []
    emit_counted_convolution(s.plan, Integers(), lambda k, a: events.append((k, a)), synchronous_completion=True)
    assert record(baseline) == json.dumps(events, sort_keys=True).encode()
    bounded = plan_convolution_schedule(s, ConvolutionScheduleChoice(99), retention_law=retention_law())
    assert [len(block) for block in bounded.column_blocks] == [2, 1]
    assert [(t.accumulator_row, t.columns) for t in bounded.column_blocks[0]] == [(0, 4), (4, 4)]
    with pytest.raises(ValueError, match='retention premises'):
        plan_convolution_schedule(s, ConvolutionScheduleChoice(None))
    with pytest.raises(ValueError, match='real completion'):
        emit_convolution_schedule(baseline, Integers(), lambda *_: None)


def test_registry_realizes_only_source_bound_programs_and_deduplicates_equivalent_capacity():
    s = source()
    original = s.identity
    feedback = []

    def verify(current):
        assert current.identity == original

    def realize(schedule):
        return EmittedProgram(record(schedule), 'neutral-effect-stream', schedule.identity)

    space = convolution_pass_space(s, verify_source=verify, realize=realize,
        retention_law=retention_law(), issue_law=issue_law(), live_tile_limits=(1, 2, None), max_candidates=6)

    def observe(program, receipt):
        feedback.append(program.data)
        return {'program_sha256': hashlib.sha256(program.data).hexdigest(), 'cycles': None}

    results = list(explore_convolution_schedules(s, space, context=PipelineContext(), feedback=observe))
    assert len(results) == 6 and all(r.evaluation.status == 'compiled' for r in results)
    assert sum(r.duplicate_of is not None for r in results) == 2 and len(feedback) == 4
    assert s.identity == original
    for r in results:
        assert r.emission.schedule_sha256 and len(r.evaluation.result.receipt['passes']) == 4
        assert r.evaluation.result.receipt['host_pass_profile'][-1]['status'] == 'completed'
    # Source effects are actual emitted operations; no target score is synthesized.
    assert all(r.feedback is None or r.feedback['cycles'] is None for r in results)


def test_source_law_mutation_and_forged_emission_identity_refuse():
    s = source()
    mutable = retention_law()
    planned = plan_convolution_schedule(s, ConvolutionScheduleChoice(2), retention_law=mutable)
    object.__setattr__(mutable, 'source_identity', 'different')
    with pytest.raises(ValueError, match='changed after planning'):
        planned.verify_current()
    called = []
    space = convolution_pass_space(s, verify_source=lambda _: None,
        realize=lambda schedule: EmittedProgram(record(schedule), 'neutral', '0'*64),
        live_tile_limits=(1,), issue_modes=(False,))
    result = list(explore_convolution_schedules(s, space, context=PipelineContext()))[0]
    assert result.evaluation.status == 'failed' and 'not bound' in result.evaluation.error
    law = retention_law()
    space = convolution_pass_space(s, verify_source=lambda _: None,
        realize=lambda _: called.append(True), retention_law=law, live_tile_limits=(1,), issue_modes=(False,))
    object.__setattr__(law, 'source_identity', 'changed')
    result = list(explore_convolution_schedules(s, space, context=PipelineContext()))[0]
    assert result.evaluation.status == 'failed' and not called


def test_issue_budget_and_original_intervals_are_not_inferred():
    s = source()
    with pytest.raises(ValueError, match='cover every'):
        replace(s, column_intervals=((0, 4), (4, 4)))
    with pytest.raises(ValueError, match='bounded-issue premises'):
        plan_convolution_schedule(s, ConvolutionScheduleChoice(1, True))
    with pytest.raises(ValueError, match='different contracts'):
        plan_convolution_schedule(s, ConvolutionScheduleChoice(2, True), retention_law=retention_law(),
            issue_law=replace(issue_law(), hardware_identity='unselected'))

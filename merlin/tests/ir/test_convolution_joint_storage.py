"""Joint input/RHS ownership retains alternatives without static-score promotion."""
import json
from dataclasses import replace

import pytest
from test_convolution_rhs_residency import rhs_law
from test_convolution_spatial_output_groups import run, selected

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhaseLaw, convolution_storage_pass_space,
    explore_convolution_schedules, plan_convolution_storage_alternative,
    plan_convolution_schedule,
)
from merlin.transforms.pass_exploration import EmittedProgram
from merlin.transforms.pipeline_state import PipelineContext


def fixture(*,accumulator_rows=None):
    s, _, rows, _, array, readout = selected(slots=3, max_output_rows=1)
    if accumulator_rows is not None:
        source=replace(s.source,plan=replace(s.source.plan,accumulator_rows=accumulator_rows))
        s=plan_convolution_schedule(source,s.choice,retention_law=s.retention_law,
                                   issue_law=s.issue_law,epoch_law=s.epoch_law)
        rows=replace(rows,source_identity=source.identity)
    phase = ConvolutionRhsPhaseLaw(s.source.identity, 'TEST-ONLY checked original whole source',
        independent_output_columns=True, immutable_inputs=True, no_intermediate_observers=True,
        full_completion_before_reprepare=True)
    return s, (rows, replace(rows, slots=5)), phase, array, readout


def plan(s, rows, phase, **kwargs):
    return plan_convolution_storage_alternative(s, rows, operand_rows=320,
        rhs_law=rhs_law(), rhs_phase_law=phase, **kwargs)


def test_joint_storage_keeps_phase_and_row_group_tradeoff_and_every_numerical_exit():
    s, rows, phase, array, readout = fixture()
    choices = [plan(s, r, phase, max_output_rows=3) for r in rows]
    assert [v.summary()['structural_effects']['complete_input_traversals'] for v in choices] == [2, 3]
    assert [v.summary()['structural_effects']['input_private_rows'] for v in choices] == [90, 150]
    assert [[g.output_rows for g in v.output_group] for v in choices] == [[1, 1], [3, 3, 3]]
    for value in choices:
        assert value.summary()['structural_effects']['performance_score'] is None
        # Independent original tensor arithmetic; finite backpressure and fresh
        # weight stays, exact K order, tails and externally visible stores.
        _, _, report = run(s, value.rhs, value.input_rows, value.output_group,
            array, readout, pending_limit=2)
        assert len(report['phase_emissions']) == len(value.rhs.phases)


def test_narrower_rhs_phases_free_real_accumulators_for_more_original_rows():
    s,rows,phase,array,readout=fixture(accumulator_rows=64)
    choices=[plan_convolution_storage_alternative(s,rows[1],operand_rows=512,
        rhs_law=rhs_law(),rhs_phase_law=phase,max_output_rows=3,rhs_column_blocks=bound)
        for bound in (None,1)]
    group=choices[0].output_group
    assert group.output_rows==1
    assert [g.output_rows for g in choices[1].output_group]==[3,3,3]
    fresh_counts=[]
    for value in choices:
        _,fresh,_=run(s,value.rhs,value.input_rows,value.output_group,array,readout,pending_limit=2)
        fresh_counts.append(sum(fresh))
    assert fresh_counts[1]<fresh_counts[0]
    assert all(p.layout.operand_rows==512 for p in choices[1].rhs.phases)


def test_registry_emits_explicit_rhs_column_choices():
    s,rows,phase,array,readout=fixture();seen=[]
    def realize(value):
        seen.append(value.summary())
        events,fresh,_=run(s,value.rhs,value.input_rows,value.output_group,array,readout,pending_limit=2)
        return EmittedProgram(json.dumps((events,fresh),sort_keys=True).encode(),'actual-effects',value.identity)
    space=convolution_storage_pass_space(s,input_rows_choices=rows[1:],operand_rows=512,
        rhs_law=rhs_law(),rhs_phase_law=phase,verify_source=lambda _:None,realize=realize,
        output_row_limits=(3,),rhs_column_limits=(None,1),max_candidates=2)
    results=list(explore_convolution_schedules(s.source,space,context=PipelineContext()))
    assert len(seen)==len(results)==2 and all(r.evaluation.status=='compiled'for r in results)
    assert len({r.emission.data for r in results})==2


@pytest.mark.parametrize('pending_limit',[0,2])
def test_rhs_frontier_prefetch_waits_for_last_batch_and_spatial_consumer(pending_limit):
    from merlin.compile.scheduling.rolling_lookahead import RollingRhsFrontierAdmission
    s,rows,phase,array,readout=fixture(accumulator_rows=64)
    # Multiple batches force the last-use guard to retain every earlier user.
    source=replace(s.source,plan=replace(s.source.plan,
        window=replace(s.source.plan.window,input_shape=(2,*s.source.plan.window.input_shape[1:]))))
    s=plan_convolution_schedule(source,s.choice,retention_law=s.retention_law,
                               issue_law=s.issue_law,epoch_law=s.epoch_law)
    inp=replace(rows[1],window=source.plan.window,source_identity=source.identity)
    phase=replace(phase,source_identity=source.identity)
    value=plan_convolution_storage_alternative(s,inp,operand_rows=512,rhs_law=rhs_law(),
        rhs_phase_law=phase,max_output_rows=3,rhs_column_blocks=2)
    law=RollingRhsFrontierAdmission('neutral','source-i32','TEST-ONLY-exact-cross-lane-completion',
                                    True,True,True,True,True)
    old_trace=[];new_trace=[]
    _,old_fresh,old=run(s,value.rhs,inp,value.output_group,array,readout,pending_limit=pending_limit,
        grouped_transport=True,trace=old_trace)
    _,new_fresh,new=run(s,value.rhs,inp,value.output_group,array,readout,pending_limit=pending_limit,
        grouped_transport=True,trace=new_trace,rhs_phase_prefetch_law=law)
    assert old_fresh==new_fresh
    for effect in ('compute','store','load_weight','stage_input_rows','initialize_bias','weight_preparation',
                   'input_ready','input_last_use','spatial_group_last_use','rhs_phase_last_use'):
        assert sum(k==effect for k,a in old_trace)==sum(k==effect for k,a in new_trace)
    assert [a for k,a in old_trace if k=='compute']==[a for k,a in new_trace if k=='compute']
    assert new['phase_emissions'][0]['rolling_transport']['rhs_frontier_tiles']==18
    assert new['phase_emissions'][-1]['rolling_transport']['rhs_frontier_tiles']==0
    future_column=value.rhs.phases[1].column_intervals[0][0]
    future_positions=[i for i,(k,a)in enumerate(new_trace)if k=='load_weight'and a['output_column']==future_column]
    assert future_positions
    for i in future_positions:
        previous=next(a for k,a in reversed(new_trace[:i])if k=='compute')
        # Each future write follows the complete K/N/M run in the final
        # source batch and final spatial cohort, including a short cohort.
        oh,ow=source.plan.window.output_spatial
        group_rows=value.output_group[0].output_rows
        last_y=((oh-1)//group_rows)*group_rows
        assert previous['output_row'] >= (oh+last_y)*ow
    with pytest.raises(ValueError,match='WAR/RAW'):
        run(s,value.rhs,inp,value.output_group,array,readout,grouped_transport=True,
            rhs_phase_prefetch_law=replace(law,scratchpad_read_write_dependencies=False))


def test_registry_consumes_each_checked_joint_owner_and_deduplicates_actual_emission():
    s, rows, phase, array, readout = fixture()
    seen = []
    def realize(value):
        seen.append(value.summary())
        events, fresh, report = run(s, value.rhs, value.input_rows, value.output_group,
            array, readout, pending_limit=2)
        return EmittedProgram(json.dumps((events, fresh), sort_keys=True).encode(),
            'actual-checked-effects', value.identity)
    space = convolution_storage_pass_space(s, input_rows_choices=rows, operand_rows=320,
        rhs_law=rhs_law(), rhs_phase_law=phase, verify_source=lambda _: None,
        realize=realize, row_block_limits=(8, 16), output_row_limits=(1, 3), max_candidates=8)
    results = list(explore_convolution_schedules(s.source, space, context=PipelineContext()))
    assert len(seen) == 8 and all(r.evaluation.status == 'compiled' for r in results)
    assert len({r.emission.data for r in results}) < 8
    assert all(r.evaluation.result.receipt['passes'] for r in results)


@pytest.mark.parametrize('change', ['source', 'capacity', 'law', 'duplicate', 'bound'])
def test_invalid_joint_ownership_refuses_before_target_callback(change):
    s, rows, phase, _, _ = fixture()
    kwargs = dict(input_rows_choices=rows, operand_rows=320, rhs_law=rhs_law(),
        rhs_phase_law=phase, row_block_limits=(16,), output_row_limits=(1, 3))
    if change == 'source':
        kwargs['input_rows_choices'] = (replace(rows[0], source_identity='0'*64),)
    elif change == 'capacity':
        kwargs['operand_rows'] = 64
    elif change == 'law':
        kwargs['rhs_phase_law'] = replace(phase, immutable_inputs=False)
    elif change == 'duplicate':
        kwargs['input_rows_choices'] = (rows[0], rows[0])
    else:
        kwargs['output_row_limits'] = (True,)
    effects = []
    with pytest.raises(ValueError):
        space = convolution_storage_pass_space(s, verify_source=lambda _: None,
            realize=lambda value: effects.append(value), **kwargs)
        list(explore_convolution_schedules(s.source, space, context=PipelineContext()))
    assert not effects


def test_stale_original_layout_and_wrong_emitted_identity_refuse():
    s, rows, phase, _, _ = fixture()
    value = plan(s, rows[0], phase)
    object.__setattr__(rows[0], 'slots', 4)
    with pytest.raises(ValueError):
        value.verify_current()
    s, rows, phase, _, _ = fixture()
    space = convolution_storage_pass_space(s, input_rows_choices=rows[:1], operand_rows=320,
        rhs_law=rhs_law(), rhs_phase_law=phase, verify_source=lambda _: None,
        realize=lambda v: EmittedProgram(b'actual', 'test-only', s.identity))
    result, = explore_convolution_schedules(s.source, space, context=PipelineContext())
    assert result.evaluation.status == 'failed'
    assert 'bound to its actual joint' in result.evaluation.error


def test_repeated_convolution_geometry_reuses_search_but_rechecks_and_emits_each_source():
    from merlin.compile.scheduling.convolution_exploration import plan_convolution_schedule
    from merlin.transforms.pass_exploration import MappingProposalCache, MappingProposalKey
    import hashlib
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    s, rows, phase, array, readout = fixture()
    second = plan_convolution_schedule(replace(s.source, source_identity='second-independent-original'),
        s.choice, retention_law=s.retention_law, issue_law=s.issue_law, epoch_law=s.epoch_law)
    second_rows = tuple(replace(row, source_identity=second.source.identity) for row in rows)
    second_phase = replace(phase, source_identity=second.source.identity)
    domain = MappingProposalKey(sha(repr((s.source.plan.window.kernel,
        s.source.plan.window.stride, s.source.plan.window.padding))),
        ((1, 5, 11, 5), (3, 3, 5, 9), (1, 5, 11, 9)),
        ('NHWC', 'reduction_major', 'NHWC'), ('i8', 'i8', 'i32'), ('bias',),
        sha('selected neutral resources'), sha('current shared passes'), sha('exact original i32'))
    cache, validations, emissions, spaces = MappingProposalCache(), [], [], []
    for original, maps, law in ((s, rows, phase), (second, second_rows, second_phase)):
        def verify(current):
            assert current.identity == original.source.identity
            current.__post_init__()
            validations.append(current.source_identity)
        def realize(value):
            value.verify_current()
            events, fresh, _ = run(value.schedule, value.rhs, value.input_rows,
                value.output_group, array, readout, pending_limit=2)
            emissions.append(value.schedule.source.source_identity)
            return EmittedProgram(json.dumps((events, fresh), sort_keys=True).encode(),
                'actual-current-convolution-effects', value.identity)
        space = convolution_storage_pass_space(original, input_rows_choices=maps, operand_rows=320,
            rhs_law=rhs_law(), rhs_phase_law=law, verify_source=verify, realize=realize,
            output_row_limits=(1, 3), max_candidates=4, proposal_cache=cache, proposal_key=domain)
        spaces.append(space)
        results = list(explore_convolution_schedules(original.source, space, context=PipelineContext()))
        assert all(row.evaluation.status == 'compiled' for row in results)
    assert not spaces[0].proposal_cache_hit and spaces[1].proposal_cache_hit
    assert spaces[1].search_steps < spaces[0].search_steps
    for identity in (s.source.source_identity, second.source.source_identity):
        assert validations.count(identity) > 0 and emissions.count(identity) == 4
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)

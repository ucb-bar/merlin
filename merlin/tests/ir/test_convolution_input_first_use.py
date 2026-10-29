"""Pending input is staged at actual original readers and closes with a drain."""
from dataclasses import replace

import pytest
from test_convolution_spatial_output_groups import selected, run
from test_counted_convolution import Integers
from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
from merlin.compile.scheduling.convolution_input_first_use import (
    ConvolutionInputFirstUseAdmission, ConvolutionInputLead, PendingConvolutionInput,
)


def law():
    return ConvolutionInputFirstUseAdmission("neutral", "source-i32", "TEST-ONLY-actual-RAW",
        True, True, True, True, True, True)


@pytest.mark.parametrize("padding,shape,slots", [((0,0,0,0),(2,7,6,5),7), ((1,1,1,1),(2,5,4,5),7)])
@pytest.mark.parametrize("pending_limit", [0,2])
@pytest.mark.parametrize("stage_channels", [None,4])
@pytest.mark.parametrize("policy", [None,"prefix","bounded"])
def test_first_use_preserves_original_products_traffic_and_actual_completion(padding,shape,slots,pending_limit,stage_channels,policy):
    args=list(selected(shape,padding,slots=slots,max_output_rows=shape[1]))
    args[2]=replace(args[2],stage_channels=stage_channels)
    old_trace,new_trace=[],[]
    old,old_fresh,_=run(*args,pending_limit=pending_limit,trace=old_trace,grouped_transport=True)
    admission=law()
    if policy is not None:
        admission=replace(admission,lead=ConvolutionInputLead(policy,"neutral","TEST-ONLY-source-service",4,8,16,4,3))
    new,new_fresh,report=run(*args,pending_limit=pending_limit,trace=new_trace,grouped_transport=True,input_first_use_law=admission)
    assert new_fresh==old_fresh
    assert [a for k,a in old_trace if k=='compute']==[a for k,a in new_trace if k=='compute']
    def payload(trace):
        return sorted((k,tuple(sorted(a.items())))for k,a in trace if k in ('stage_input_rows','fill_input_rows','store'))
    assert payload(old_trace)==payload(new_trace)
    assert new.count('spatial_group_last_use')==old.count('spatial_group_last_use')==shape[0]
    assert new.count('input_last_use')==old.count('input_last_use')==shape[0]
    assert 'input_ready' not in new
    # At least one original compute precedes a later row's first DMA.
    compute=next(i for i,(k,a)in enumerate(new_trace)if k=='compute')
    assert any(k in ('stage_input_rows','fill_input_rows')for k,a in new_trace[compute+1:])
    assert report['rolling_transport']['schema']=='merlin.convolution_pending_input.v1'


@pytest.mark.parametrize("field",['scratchpad_raw_dependencies','ordered_configuration','finite_queue_progress',
    'immutable_input','dma_preserves_stationary_array','full_consumer_transport_completion','hardware_identity'])
def test_missing_actual_target_premise_refuses_before_effects(field):
    s,rhs,rows,group,array,readout=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    admission=replace(law(),**{field:'other' if field=='hardware_identity'else False})
    effects=[]
    with pytest.raises(ValueError):
        emit_convolution_schedule(s,Integers(),lambda *a:effects.append(a),complete=lambda *a:effects.append(a),
            rhs=rhs,input_rows=rows,output_group=group,array_law=array,readout_law=readout,
            grouped_transport=True,input_first_use_law=admission)
    assert effects==[]


def test_owner_requires_actual_reader_scope_and_real_final_completion():
    s,rhs,rows,group,array,readout=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    owner=PendingConvolutionInput(s,group,rows,law(),image=0)
    assert owner.first_uses==tuple((r,(0,r))for r in range(5))+((5,(1,4)),(6,(2,4)))
    effects=[]
    with pytest.raises(ValueError):owner.finish(lambda *a:effects.append(a),{})
    owner.start(s,group,rows,(0,4,8,12,16),(4,)*5,Integers(),lambda *a:effects.append(a))
    with pytest.raises(ValueError):owner.finish(lambda *a:effects.append(a),{})
    with pytest.raises(ValueError):owner.before_reader(Integers(),lambda *a:effects.append(a),0,0,0,5)
    assert not owner.closed and effects==[]
    with pytest.raises(ValueError):PendingConvolutionInput(s,replace(group,output_rows=4),rows,law(),image=0)


def test_pending_input_composes_with_original_dead_rhs_frontier():
    from test_convolution_rhs_residency import rhs_law
    from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsPhaseLaw,plan_convolution_rhs_phases
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
    from merlin.compile.scheduling.rolling_lookahead import RollingRhsFrontierAdmission
    s,_,rows,_,array,readout=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    phases=plan_convolution_rhs_phases(s,operand_rows=1024,law=rhs_law(),input_rows=rows,max_column_blocks=2,
        phase_law=ConvolutionRhsPhaseLaw(s.source.identity,"TEST-ONLY-complete-phase",True,True,True,True))
    groups=plan_rolling_convolution_output_group(s,phases,rows,max_output_rows=5)
    frontier=RollingRhsFrontierAdmission("neutral","source-i32","TEST-ONLY-WAR-RAW",True,True,True,True,True)
    trace=[]
    events,_,report=run(s,phases,rows,groups,array,readout,pending_limit=2,trace=trace,grouped_transport=True,
        rhs_phase_prefetch_law=frontier,input_first_use_law=law())
    assert len(report['phase_emissions'])==2
    assert events.count('rhs_phase_last_use')==2
    assert events.count('spatial_group_last_use')==4
    assert 'input_ready'not in events


@pytest.mark.parametrize("interval",[(1,4),(0,3),(4,4),(8,1)])
def test_partial_or_unowned_original_channel_transfer_refuses_before_effects(interval):
    _,_,rows,_,_,_=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    rows=replace(rows,stage_channels=4)
    effects=[]
    with pytest.raises(ValueError):
        rows.emit_stage(Integers(),{},lambda *a:effects.append(a),channel_interval=interval)
    assert effects==[]


@pytest.mark.parametrize("policy",["prefix","bounded"])
def test_pending_release_plan_retains_every_original_transfer_and_reader_deadline(policy):
    s,_,rows,group,_,_=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    rows=replace(rows,stage_channels=4)
    admission=replace(law(),lead=ConvolutionInputLead(policy,"neutral","TEST-ONLY-source-service",4,8,16,4,3))
    owner=PendingConvolutionInput(s,group,rows,admission,image=0)
    assert len(owner.releases)==7*2
    assert len({(row,interval)for release,first,row,interval in owner.releases})==14
    assert all(0<=release<=first for release,first,row,interval in owner.releases)
    assert any(release<first for release,first,row,interval in owner.releases)
    if policy=="prefix":assert all(release==0 for release,first,row,interval in owner.releases)
    else:assert any(release>0 for release,first,row,interval in owner.releases)


@pytest.mark.parametrize("change",[dict(load_entries=0),dict(reader_entries=0),dict(request_bytes=7),
    dict(beat_bytes=0),dict(startup_cycles=0),dict(compute_cycles_per_row=0),dict(policy="arbitrary"),dict(hardware_identity="other")])
def test_lead_needs_selected_finite_resources_and_service_scope(change):
    s,_,rows,group,_,_=selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5)
    lead=ConvolutionInputLead("bounded","neutral","TEST-ONLY-source-service",4,8,16,4,3)
    with pytest.raises(ValueError):PendingConvolutionInput(s,group,rows,replace(law(),lead=replace(lead,**change)),image=0)

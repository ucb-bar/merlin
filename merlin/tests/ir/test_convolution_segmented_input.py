"""Disjoint channel row segments retain original payloads and source lifetimes."""
from dataclasses import replace
import pytest
from test_convolution_spatial_output_groups import selected,run
from test_convolution_input_first_use import law
from test_convolution_rhs_residency import rhs_law
from test_counted_convolution import Integers
from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule,plan_convolution_storage_alternative
from merlin.compile.scheduling.convolution_input_first_use import ConvolutionInputLead


def segmented():
    args=list(selected((2,7,6,5),(0,0,0,0),slots=7,max_output_rows=5))
    args[2]=replace(args[2],stage_channels=4,private_rows=992,channel_segments=((0,4,32),(4,1,768)))
    return args


@pytest.mark.parametrize('policy',[None,'bounded','bank_advance'])
@pytest.mark.parametrize('pending_limit',[0,2])
def test_exact_original_ragged_channel_payloads_through_segmented_normal_consumer(policy,pending_limit):
    args=segmented();original=list(args);original[2]=replace(args[2],channel_segments=None)
    old_trace,new_trace=[],[]
    old,old_fresh,_=run(*original,pending_limit=pending_limit,trace=old_trace,grouped_transport=True,input_first_use_law=law())
    admission=law()
    if policy is not None:admission=replace(admission,lead=ConvolutionInputLead(policy,'neutral','TEST-ONLY-bank-resource',4,8,16,4,3,bank_rows=256,bank_count=4))
    new,new_fresh,report=run(*args,pending_limit=pending_limit,trace=new_trace,grouped_transport=True,input_first_use_law=admission)
    assert new_fresh==old_fresh
    assert new.count('spatial_group_last_use')==old.count('spatial_group_last_use')==2
    assert new.count('input_last_use')==old.count('input_last_use')==2
    def payload(trace):
        return sorted(tuple(sorted((k,v)for k,v in a.items()if k!='local_row'))for kind,a in trace if kind=='stage_input_rows')
    assert payload(old_trace)==payload(new_trace)
    rows=args[2]
    assert rows.physical_windows==((32,88),(768,824)) and rows.allocated_rows==112
    assert rows.end==824 and rows.disjoint_from(512,216)
    assert report['resident_input_rows']['private_row_windows']==rows.physical_windows
    cells=[rows.physical_cell(y*rows.rows_per_source_row*4+x*5+c)for y in range(7)for x in range(6)for c in range(5)]
    assert len(set(cells))==7*6*5
    assert all(32<=address<88 or 768<=address<824 for address,lane in cells)


def test_joint_storage_uses_actual_union_and_separately_owned_rhs_gap():
    s,_,rows,_,_,_=segmented()
    value=plan_convolution_storage_alternative(s,rows,operand_rows=768,rhs_base=512,rhs_law=rhs_law(),max_output_rows=5)
    value.verify_current()
    assert value.rhs.layout.rhs_base==512 and value.rhs.layout.rhs_rows==216
    assert value.summary()['structural_effects']['input_private_rows']==112


@pytest.mark.parametrize('segments',[
    ((0,4,32),),((0,4,32),(4,2,768)),((0,3,32),(3,2,768)),
    ((0,4,32),(4,1,64)),((0,4,32),(4,1,1000)),((0,4,32),(4,1,-1)),
    ((0,4,32),(4,1,'bad')),[]])
def test_incomplete_overlapping_unowned_or_transfer_cut_segments_refuse(segments):
    rows=segmented()[2]
    with pytest.raises(ValueError):replace(rows,channel_segments=segments)


def test_actual_rhs_overlap_refuses_before_target_preparation():
    s,rhs,rows,group,array,readout=segmented()
    rows=replace(rows,channel_segments=((0,4,32),(4,1,640)))
    effects=[]
    with pytest.raises(ValueError):
        emit_convolution_schedule(s,Integers(),lambda *a:effects.append(a),complete=lambda *a:effects.append(a),
            rhs=rhs,input_rows=rows,output_group=group,array_law=array,readout_law=readout,grouped_transport=True,input_first_use_law=law())
    assert effects==[]


def test_bank_deadlines_precede_late_first_bank_KY_order_without_prefilling_next_bank():
    from merlin.compile.scheduling.convolution_input_first_use import PendingConvolutionInput
    s,_,rows,group,_,_=segmented()
    admission=replace(law(),lead=ConvolutionInputLead('bank_advance','neutral','TEST-ONLY-bank-resource',4,8,16,4,3,bank_rows=256,bank_count=4))
    owner=PendingConvolutionInput(s,group,rows,admission,image=0)
    assert all(release==0 for release,first,row,interval in owner.releases if interval[0]==0)
    assert any(release>0 for release,first,row,interval in owner.releases if interval[0]==4)
    assert all(release<=first for release,first,row,interval in owner.releases)

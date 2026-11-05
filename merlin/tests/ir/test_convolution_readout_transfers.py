"""Exact source cells through source-admitted block-major narrow readouts."""
from dataclasses import replace

import pytest
from test_convolution_compute_epochs import plan
from test_convolution_readout_epochs import readout_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
from merlin.compile.scheduling.window_transfer import (
    ConvolutionOutputTile,
    ConvolutionReadoutTransferLaw,
    plan_convolution_readout_transfers,
)


def law(schedule, *, states=None, maximum=8):
    p=schedule.source.plan
    return ConvolutionReadoutTransferLaw('neutral','source-i32',p.identity,'TEST-ONLY-selected-narrow-block-map',
        p.column_tile,p.row_tile,maximum,states or ((0,p.output_channels,'a'*64),),
        scaled_i8=True,exact_block_column_map=True,original_numerical_state=True)


def cells(events, tile, step):
    output={}
    for kind,f in events:
        if kind!='store':
            continue
        for r in range(f['rows']):
            for c in range(f['columns']):
                key=f['output_row']+r,f['output_column']+c
                assert key not in output
                output[key]=(f['accumulator_row']+(c//tile)*step+r,c%tile)
    return output


@pytest.mark.parametrize('maximum,states,expected', [
    (8,None,2), (16,None,1), (8,((0,4,'a'*64),(4,5,'b'*64)),2),
])
def test_real_epoch_wide_readouts_publish_all_original_cells_once(maximum,states,expected):
    schedule=plan(4)
    original,actual,old_drains,new_drains=[],[],[],[]
    emit_convolution_schedule(schedule,Integers(),lambda k,a:original.append((k,dict(a))),
        complete=lambda k,a:old_drains.append((k,dict(a))),readout_law=readout_law())
    report=emit_convolution_schedule(schedule,Integers(),lambda k,a:actual.append((k,dict(a))),
        complete=lambda k,a:new_drains.append((k,dict(a))),readout_law=readout_law(),
        readout_transfer_law=law(schedule,states=states,maximum=maximum))
    p=schedule.source.plan
    assert cells(original,p.column_tile,p.row_tile)==cells(actual,p.column_tile,p.row_tile)
    assert len(cells(actual,p.column_tile,p.row_tile))==p.coverage['output_elements']
    assert [(k,a)for k,a in original if k!='store']==[(k,a)for k,a in actual if k!='store']
    assert new_drains==old_drains
    assert report['readout_transfers']['original_stores_per_spatial_tile']==3
    assert report['readout_transfers']['physical_stores_per_spatial_tile']==expected
    assert sum(k=='store'for k,_ in actual)*3==sum(k=='store'for k,_ in original)*expected


def test_partial_tiles_gaps_and_state_boundaries_keep_original_transfers():
    schedule=plan(4)
    p=schedule.source.plan
    tiles=(ConvolutionOutputTile(0,3,0),ConvolutionOutputTile(3,3,4),ConvolutionOutputTile(6,3,8))
    assert all(len(g)==1 for g in plan_convolution_readout_transfers(p,tiles,law(schedule),rows=4))
    tiles=(ConvolutionOutputTile(0,4,0),ConvolutionOutputTile(4,4,8))
    assert all(len(g)==1 for g in plan_convolution_readout_transfers(p,tiles,law(schedule),rows=4))


def test_row_tail_preserves_physical_block_spacing_without_inventing_padding():
    schedule=plan(4)
    p=schedule.source.plan
    tiles=(ConvolutionOutputTile(0,4,0),ConvolutionOutputTile(4,4,4))
    # Fewer live rows still use the original physical block stride.
    assert plan_convolution_readout_transfers(p,tiles,law(schedule),rows=2)==(tiles,)
    # Compact private slots cannot be reinterpreted as wider hardware blocks.
    selected=replace(law(schedule),block_rows=8)
    assert plan_convolution_readout_transfers(p,tiles,selected,rows=2)==tuple((t,)for t in tiles)


@pytest.mark.parametrize('changes', [
    {'scaled_i8':False},{'exact_block_column_map':False},{'original_numerical_state':False},
    {'source_identity':'wrong'},{'hardware_identity':'wrong'},{'numerical_policy_identity':'wrong'},
    {'tile_columns':3},{'block_rows':0},{'max_columns':True},{'max_columns':3},
    {'column_states':((0,8,'a'*64),)}, {'column_states':((0,4,'a'*64),(5,5,'b'*64))},
    {'column_states':((0,9,'not-a-source-identity'),)},
    {'column_states':((0,3,'a'*64),(3,6,'b'*64))},
])
def test_missing_or_changed_map_refuses_before_any_preparation(changes):
    schedule=plan(4)
    effects=[]
    with pytest.raises(ValueError):
        emit_convolution_schedule(schedule,Integers(),lambda *a:effects.append(a),
            complete=lambda *a:effects.append(a),readout_law=readout_law(),
            readout_transfer_law=replace(law(schedule),**changes))
    assert effects==[]


def test_map_does_not_grant_producer_completion_and_detects_callback_mutation():
    schedule=plan(4)
    selected=law(schedule)
    effects=[]
    with pytest.raises(ValueError,match='producer/readout epoch'):
        emit_convolution_schedule(schedule,Integers(),lambda *a:effects.append(a),
            complete=lambda *a:effects.append(a),readout_transfer_law=selected)
    assert effects==[]
    def mutate(kind,fields):
        if kind=='store':
            object.__setattr__(selected,'max_columns',4)
    with pytest.raises(ValueError,match='changed during emission'):
        emit_convolution_schedule(schedule,Integers(),mutate,complete=lambda *_:None,
            readout_law=readout_law(),readout_transfer_law=selected)

"""Spatial drains preserve complete retained source values and output coverage."""
from dataclasses import replace
import numpy as np
import pytest
from merlin.compile.scheduling.block_schedule import (
    BlockScheduleError,Contraction,Geometry,Knobs,Preload,Store,SpatialMaxReadout,
    schedule_contraction,with_column_bias,plan_spatial_max_readout,check_spatial_max_readout,
    execute_spatial_max_readout,
)

def build(*,padding=(0,0,0,0),bias=False):
    base=schedule_contraction(Contraction(2*3*5,7,6),Geometry(4,128,32,128,True),Knobs(lookahead_steps=0))
    if bias:base=with_column_bias(base,"bias")
    spec=SpatialMaxReadout((3,5),(2,2),(2,2),padding,pad_value=-128 if any(padding)else None)
    return plan_spatial_max_readout(base,spec,"source-readout"),base

@pytest.mark.parametrize("padding",[(0,0,0,0),(1,0,0,1),(1,1,1,1)])
@pytest.mark.parametrize("biased",[False,True])
def test_retained_tails_batch_padding_and_signed_readout(padding,biased):
    result,base=build(padding=padding,bias=biased);before=repr(base)
    a=(np.arange(210)%17-8).astype(np.int8).reshape(30,7)
    w=(np.arange(42)%13-6).astype(np.int8).reshape(7,6)
    bias=np.array([-(1<<31),(1<<31)-1,-9,0,11,-7],dtype=np.int32)
    def readout(x):return int(np.clip(np.rint(np.float32(x)*np.float32(.5)),-128,127))
    raw=(a.astype(np.int64)@w.astype(np.int64)+(bias if biased else 0)).astype(np.int32).reshape(2,3,5,6)
    quant=np.clip(np.rint(raw.astype(np.float32)*np.float32(.5)),-128,127).astype(np.int8)
    pt,pl,pb,pr=padding;oh,ow=result.source.output_hw
    padded=np.pad(quant,((0,0),(pt,pb),(pl,pr),(0,0)),constant_values=-128)
    expected=np.stack([padded[b,y*2:y*2+2,x*2:x*2+2].max(axis=(0,1)) for b in range(2)for y in range(oh)for x in range(ow)])
    got=execute_spatial_max_readout(result,a,w,readout,**({"bias":bias}if biased else {}))
    assert np.array_equal(got,expected)and repr(base)==before
    assert all(not isinstance(o,Store)for o in result.prefix)

@pytest.mark.parametrize("mutation",["early_overwrite","missing_store","wrong_window","duplicate_store","prefix","source","readout","capacity","budget","padding_unknown"])
def test_spatial_effect_mutations_refuse(mutation):
    result,base=build();source=result.source;readout=result.readout
    with pytest.raises(BlockScheduleError):
        if mutation=="capacity":
            plan_spatial_max_readout(replace(base,geometry=replace(base.geometry,accumulator_rows=4)),source,readout);return
        if mutation=="budget":plan_spatial_max_readout(base,source,readout,max_window_reads=1);return
        if mutation=="padding_unknown":SpatialMaxReadout((3,5),(2,2),(1,1),(1,0,0,0));return
        if mutation=="early_overwrite":
            ops=list(result.retained.ops);indices=[i for i,o in enumerate(ops)if isinstance(o,Preload)]
            ops[indices[-1]]=replace(ops[indices[-1]],accumulator_row=ops[indices[0]].accumulator_row)
            result=replace(result,retained=replace(result.retained,ops=tuple(ops)))
        if mutation=="missing_store":result=replace(result,stores=result.stores[:-1])
        if mutation=="duplicate_store":result=replace(result,stores=result.stores+(result.stores[0],))
        if mutation=="wrong_window":
            first=result.stores[0];windows=list(first.window_rows);windows[0]=(999,)+windows[0][1:]
            result=replace(result,stores=(replace(first,window_rows=tuple(windows)),)+result.stores[1:])
        if mutation=="prefix":result=replace(result,prefix=result.prefix[:-1])
        if mutation=="source":source=replace(source,stride=(1,1))
        if mutation=="readout":readout="different"
        check_spatial_max_readout(result,source=source,readout=readout)

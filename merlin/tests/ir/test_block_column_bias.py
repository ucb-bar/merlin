"""Explicit modular column seeds preserve finite contraction ownership and values."""
from dataclasses import replace
import numpy as np
import pytest
from merlin.compile.scheduling.block_schedule import (
    BlockScheduleError, ColumnBias, Compute, Contraction, ConvContraction, Geometry,
    InitializeAccumulator, Knobs, Preload, check_residency, execute,
    schedule_contraction, schedule_tiled_convolution, with_column_bias,
)


def original():
    return schedule_contraction(Contraction(5,7,6),Geometry(4,128,32,128,True),Knobs(lookahead_steps=0))


def test_tail_bias_exact_modular_i32_and_unchanged_raw_schedule():
    base=original();before=repr(base);changed=with_column_bias(base,'bias')
    a=(np.arange(35)%17-8).astype(np.int8).reshape(5,7)
    w=(np.arange(42)%13-6).astype(np.int8).reshape(7,6)
    b=np.array([-(1<<31),(1<<31)-1,0,11,-17,123],dtype=np.int32)
    got=execute(changed,a,w,bias=b)
    expected=(a.astype(np.int64)@w.astype(np.int64)+b).astype(np.int32)
    assert got.dtype==np.int32 and np.array_equal(got,expected)
    assert repr(base)==before and np.array_equal(execute(base,a,w),a.astype(np.int64)@w.astype(np.int64))
    assert changed.count(InitializeAccumulator)==4


def test_padded_convolution_bias_uses_every_source_tap_and_column():
    cv=ConvContraction(1,3,4,3,2,2,5,padding=(1,0,0,1))
    tiled=schedule_tiled_convolution(cv,Geometry(4,128,32,128,True),Knobs(lookahead_steps=0),candidate_budget=32,max_operations=2048)
    changed=with_column_bias(tiled.schedule,'bias',max_operations=2048)
    image=(np.arange(36)%9-4).astype(np.int8).reshape(1,3,4,3)
    weights=(np.arange(60)%7-3).astype(np.int8).reshape(2,2,3,5)
    bias=np.array([(1<<31)-1,-(1<<31),3,-5,11],dtype=np.int32)
    expected=np.zeros((cv.out_h,cv.out_w,5),dtype=np.int64)
    for oh in range(cv.out_h):
        for ow in range(cv.out_w):
            expected[oh,ow]=bias
            for kh in range(2):
                for kw in range(2):
                    ih,iw=oh+kh-1,ow+kw
                    if 0<=ih<3 and 0<=iw<4:
                        expected[oh,ow]+=image[0,ih,iw].astype(np.int64)@weights[kh,kw].astype(np.int64)
    assert np.array_equal(execute(changed,image.reshape(12,3),weights.reshape(12,5),bias=bias),expected.astype(np.int32).reshape(-1,5))


@pytest.mark.parametrize('mutation',['missing','duplicate','column','stride','address','first_overwrite','wrong_source','missing_declaration','budget'])
def test_bias_effect_mutations_refuse(mutation):
    base=original()
    if mutation=='budget':
        with pytest.raises(BlockScheduleError):with_column_bias(base,'bias',max_operations=len(base.ops))
        return
    changed=with_column_bias(base,'bias');ops=list(changed.ops);index=next(i for i,o in enumerate(ops)if isinstance(o,InitializeAccumulator))
    if mutation=='missing':ops.pop(index)
    if mutation=='duplicate':ops.insert(index,ops[index])
    if mutation=='column':ops[index]=replace(ops[index],dram_col=1)
    if mutation=='stride':ops[index]=replace(ops[index],source_row_stride=1)
    if mutation=='address':ops[index]=replace(ops[index],row=changed.geometry.accumulator_rows)
    if mutation=='first_overwrite':
        i=next(i for i,o in enumerate(ops)if isinstance(o,Preload));ops[i]=replace(ops[i],accumulate=False)
    if mutation=='wrong_source':changed=replace(changed,column_bias=ColumnBias('lhs',6))
    if mutation=='missing_declaration':changed=replace(changed,column_bias=None)
    with pytest.raises(BlockScheduleError):check_residency(replace(changed,ops=tuple(ops)))

from dataclasses import replace

import pytest

from xdsl.dialects import builtin
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_window import RollingWindowPlan
from merlin.compile.scheduling.rolling_lookahead import (
    RollingLookaheadAdmission, RollingRhsFrontier, materialize_lookahead_rolling_window,
    verify_lookahead_rolling_window, emit_lookahead_rolling_window,
)
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.xdsl_dialects.rolling_window import StageRowOp, StageRhsOp, CompleteOp, AdvanceConsumerOp
from merlin.compile.scheduling.convolution_output_groups import ConvolutionGroupContinuation


class IntegerBuilder:
    def op(self, name, a, b):
        return {"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"urem":lambda:(a%(1<<64))%b}[name]()

    def cmp(self, name, a, b):
        return a==b if name=="eq" else a%(1<<64)<b%(1<<64)

    def branch(self,condition,yes,no):
        return (yes if condition else no)()

    def loop(self, lo, hi, step, body):
        for i in range(lo,hi,step):body(i)

    def loop_carried(self, lo, hi, step, initial, body):
        value=initial
        for i in range(lo,hi,step):value=body(i,value)
        return value


def plan(shape=(2,11,17,5),kernel=(3,3),stride=(1,1),padding=(1,1,1,1),dilation=(1,1),slots=8):
    window=Window2D(shape,kernel,stride,dilation,padding)
    oh,ow=window.output_spatial
    return RollingWindowPlan(window,"a"*64,0,ow,0,oh,AddressWindow('input-cache',64,slots*shape[2]*shape[3]),slots,shape[2]*shape[3])


def admission():
    return RollingLookaheadAdmission("test-hardware","test-effects",True,True,True,True,True,True)


@pytest.mark.parametrize("p,group",[
    (plan(),3),
    (plan(shape=(1,10,7,9),padding=(0,0,0,0),slots=10),6),
    (plan(shape=(2,13,9,3),stride=(2,1),padding=(2,1,1,1),dilation=(2,1),slots=15),3),
    (plan(shape=(1,4,5,1),padding=(0,0,0,0),slots=4),4),
])
@pytest.mark.parametrize("runs", [None, 1, 17])
def test_actual_scf_preserves_original_taps_and_never_overwrites_current_union(p,group,runs):
    module=materialize_lookahead_rolling_window(p,output_rows=group,consumer_runs=runs)
    verified=verify_lookahead_rolling_window(module,p,output_rows=group,consumer_runs=runs)
    cache={};seen=[];staged=[];events=[];pending=set();counter=0

    def lower(kind,f):
        nonlocal counter
        events.append(kind)
        if kind=="begin_transport":
            counter+=1;pending.add(counter);return counter
        if kind=="stage_row":
            assert f['transfer']in pending
            # Assume every future write completes immediately, maximizing its
            # opportunity to corrupt the current consumer if slots overlap.
            cache[f['cache_byte']]=(f['image'],f['input_row']if f['valid']else None)
            staged.append((f['image'],f['input_row']))
        elif kind=="await_transport":
            pending.discard(f['transfer']);return f['transfer']
        elif kind in ("consume_group", "start_consumer"):
            assert f['resident']not in pending
            w=p.window
            def read(first=0,stop=1):
                for run in range(first,stop):
                    for y in range(f['output_y'],f['output_y']+f['output_rows']):
                        if run==0:seen.append((f['image'],y))
                        for ky in range(w.kernel[0]):
                            unbiased=y*w.stride[0]+ky*w.dilation[0]
                            iy=unbiased-w.padding[0]
                            key=p.cache.start+(unbiased%p.slots)*p.row_stride_bytes
                            assert cache[key]==(f['image'],iy if 0<=iy<w.input_shape[1]else None)
            if kind=="start_consumer":
                return ConvolutionGroupContinuation(f['consumer_runs'],read,lambda:f['resident'])
            read()
            return f['resident']
        elif kind=="advance_consumer":
            return f['consumer'].advance(f['first_run'],f['run_count'])
        elif kind=="finish_consumer":return f['consumer'].finish()
        elif kind=="complete":pending.clear()
        return 0

    result=emit_lookahead_rolling_window(module,p,IntegerBuilder(),lower,output_rows=group,admission=admission(),consumer_runs=runs)
    assert result['actual_ir_sha256']==verified['actual_ir_sha256']
    assert seen==[(b,y)for b in range(p.window.input_shape[0])for y in range(p.output_y_count)]
    expected=p.window.input_shape[0]*(p.span_rows+(p.output_y_count-1)*p.window.stride[0])
    assert len(staged)==expected==verified['dynamic_stage_rows']
    assert events.count('start_consumer'if runs else'consume_group')==events.count('complete')
    if runs and p.output_y_count>group:
        assert events.index('advance_consumer')<events.index('finish_consumer')
        between=events[events.index('advance_consumer'):events.index('finish_consumer')]
        assert 'stage_row'in between


def test_lookahead_requires_extra_owned_slots_and_real_effect_admission():
    with pytest.raises(ValueError,match="overlap"):
        materialize_lookahead_rolling_window(plan(slots=5),output_rows=3)
    p=plan();m=materialize_lookahead_rolling_window(p,output_rows=3)
    with pytest.raises(ValueError,match="admission"):
        emit_lookahead_rolling_window(m,p,IntegerBuilder(),lambda *a:0,output_rows=3,
                                     admission=RollingLookaheadAdmission('hw','unproved'))


@pytest.mark.parametrize('mutation',['address','completion'])
def test_reject_changed_actual_scf_addresses_or_completion(mutation):
    p=plan();m=materialize_lookahead_rolling_window(p,output_rows=3)
    if mutation=='address':
        stage=next(o for o in m.walk()if type(o)is StageRowOp)
        stage.properties['row_bytes']=builtin.IntegerAttr(p.row_bytes-1,64)
    else:
        complete=next(o for o in m.walk()if type(o)is CompleteOp)
        complete.parent.erase_op(complete)
    with pytest.raises(ValueError,match='SCF'):
        verify_lookahead_rolling_window(m,p,output_rows=3)


def test_split_consumer_refuses_skipped_products_and_early_readout():
    p=plan();m=materialize_lookahead_rolling_window(p,output_rows=3,consumer_runs=17)
    advance=next(o for o in m.walk()if type(o)is AdvanceConsumerOp)
    advance.properties['first_run']=builtin.IntegerAttr(1,64)
    with pytest.raises(ValueError,match='SCF'):
        verify_lookahead_rolling_window(m,p,output_rows=3,consumer_runs=17)
    consumer=ConvolutionGroupContinuation(17,lambda first,stop:None,lambda:None)
    with pytest.raises(ValueError,match='every original'):consumer.finish()
    with pytest.raises(ValueError,match='skips'):consumer.advance(1,4)
    consumer.advance(0,4)
    with pytest.raises(ValueError,match='skips'):consumer.advance(0,4)
    consumer.advance(4,13);consumer.finish()
    with pytest.raises(ValueError,match='skips'):consumer.advance(17,1)


@pytest.mark.parametrize('mutation',['address','columns','last_use_guard'])
def test_rhs_frontier_rejects_changed_actual_address_or_last_use_guard(mutation):
    from xdsl.dialects import arith
    p=plan()
    frontier=RollingRhsFrontier(p.source_identity,2048,4,11,3,((4,4),(8,3)))
    options=dict(output_rows=3,consumer_runs=18,rhs_frontier=frontier,future_input=False)
    m=materialize_lookahead_rolling_window(p,**options)
    stage=next(o for o in m.walk()if type(o)is StageRhsOp)
    if mutation=='address':
        stage.operands=(*stage.operands[:3],stage.reduction_row,stage.rows)
    elif mutation=='columns':
        stage.properties['columns']=builtin.IntegerAttr(1,64)
    else:
        guard=next(o for o in m.walk()if type(o)is arith.CmpiOp and o.predicate.value.data==0)
        guard.properties['predicate']=builtin.IntegerAttr(1,64)
    with pytest.raises(ValueError,match='SCF'):
        verify_lookahead_rolling_window(m,p,**options)


def test_rhs_frontier_refuses_incomplete_source_runs_and_growing_next_span():
    p=plan()
    frontier=RollingRhsFrontier(p.source_identity,2048,4,11,3,((4,4),(8,3)))
    for invalid,runs in ((frontier,17),(replace(frontier,current_column_blocks=1),18),
                         (replace(frontier,next_columns=((4,4),(7,3))),18)):
        with pytest.raises(ValueError):
            materialize_lookahead_rolling_window(p,output_rows=3,consumer_runs=runs,
                rhs_frontier=invalid,future_input=False)


@pytest.mark.parametrize("shape,kernel,stride,padding,dilation", [
    ((3,7,9,5),(3,3),(1,1),(0,0,0,0),(1,1)),
    ((2,9,7,3),(3,3),(2,1),(1,1,1,1),(2,1)),
    ((1,5,7,5),(1,3),(1,1),(0,0,0,0),(1,1)),
])
@pytest.mark.parametrize('image_windows',[1,2])
def test_batch_frontier_stages_each_original_row_after_its_final_source_read(
        shape,kernel,stride,padding,dilation,image_windows):
    from merlin.compile.scheduling.rolling_lookahead import RollingBatchFrontier, RollingBatchFrontierAdmission
    window=Window2D(shape,kernel,stride,dilation,padding)
    oh,ow=window.output_spatial
    slots=(kernel[0]-1)*dilation[0]+1+(oh-1)*stride[0]
    p=plan(shape,kernel,stride,padding,dilation,slots)
    if image_windows>1:
        p=replace(p,slots=slots*image_windows,image_windows=image_windows,
            cache=replace(p.cache,size=p.cache.size*image_windows))
    chunks=(shape[-1]+3)//4;runs=kernel[0]*kernel[1]*chunks
    frontier=RollingBatchFrontier(p.source_identity,4)
    options=dict(output_rows=oh,consumer_runs=runs,future_input=False,batch_frontier=frontier)
    module=materialize_lookahead_rolling_window(p,**options)
    cache={};staged=[];seen=[];completed=[];events=[];counter=0
    def lower(kind,f):
        nonlocal counter
        events.append(kind)
        if kind=='begin_transport':counter+=1;return counter
        if kind=='stage_row':
            if image_windows>1 and f['image']>=image_windows:
                assert f['image']-image_windows in completed
            cache[f['cache_byte']]=(f['image'],f['input_row']if f['valid']else None)
            staged.append((f['image'],f['input_row']))
        elif kind=='await_transport':return f['transfer']
        elif kind=='start_consumer':
            def advance(first,stop,after_run=None):
                for run in range(first,stop):
                    ky=run//(kernel[1]*chunks)
                    for y in range(oh):
                        unbiased=y*stride[0]+ky*dilation[0];iy=unbiased-padding[0]
                        slot=(f['image']%image_windows)*p.image_slots+unbiased%p.image_slots
                        key=p.cache.start+slot*p.row_stride_bytes
                        assert cache[key]==(f['image'],iy if 0<=iy<shape[1]else None)
                    if run==0:seen.append(f['image'])
                    if after_run:after_run(run,0,min(4,shape[-1]-(run%chunks)*4))
            return ConvolutionGroupContinuation(runs,advance,lambda:completed.append(f['image']))
        elif kind=='advance_consumer':
            return f['consumer'].advance(f['first_run'],f['run_count'],after_run=f.get('after_run'))
        elif kind=='finish_consumer':f['consumer'].finish()
        return 0
    law=RollingBatchFrontierAdmission('test-hardware','test-numerics','TEST-only original input effects',
                                     True,True,True,True,True,True)
    result=emit_lookahead_rolling_window(module,p,IntegerBuilder(),lower,
        admission=admission(),batch_frontier_admission=law,**options)
    assert seen==completed==list(range(shape[0]))
    assert len(staged)==shape[0]*slots==result['dynamic_stage_rows']
    assert len(set(staged))==len(staged)
    assert events.count('complete')==shape[0]
    assert events.count('await_transport')==shape[0]+1
    if shape[0]>1:
        assert 'stage_row'in events[events.index('advance_consumer'):events.index('finish_consumer')]


def test_batch_frontier_refuses_partial_source_and_changed_actual_row_write():
    from merlin.compile.scheduling.rolling_lookahead import RollingBatchFrontier
    p=plan(shape=(2,7,9,5),padding=(0,0,0,0),slots=7)
    frontier=RollingBatchFrontier(p.source_identity,4)
    with pytest.raises(ValueError,match='complete original'):
        materialize_lookahead_rolling_window(p,output_rows=4,consumer_runs=18,
            batch_frontier=frontier,future_input=False)
    options=dict(output_rows=5,consumer_runs=18,batch_frontier=frontier,future_input=False)
    m=materialize_lookahead_rolling_window(p,**options)
    stage=next(o for o in m.walk()if type(o)is StageRowOp)
    stage.properties['row_bytes']=builtin.IntegerAttr(p.row_bytes-1,64)
    with pytest.raises(ValueError,match='SCF'):
        verify_lookahead_rolling_window(m,p,**options)

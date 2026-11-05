"""Real Q1 completion, bounded private translation, original residual phases."""
from dataclasses import replace
import sys
from merlin.common.paths import repo_root
import pytest

sys.path.insert(0,str(repo_root()/'merlin/tests/ir'))
from test_quantized_sum_source import fixture as source_fixture
from test_dense_retained_reduction import inputs
from test_compact_dense import ADAPTER
from merlin.compile.scheduling.block_schedule import Store
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout,TiledIntegerStore
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.dense_loaded_sum_composition import (
    CheckedLoadedSumView, QuantizedSumCompositionAdmission,compose_dense_loaded_sum,_compose_kernel)
from merlin.compile.scheduling.compact_loaded_sum import LoadedSumContract,LoadedSumLayout,LoadedSumEffect
from merlin.compile.scheduling.loaded_sum_epochs import LoadedSumEpochLaw,plan_loaded_sum_epochs
from merlin.llvmlower.quantized_affine_pair import prove_ordered_loaded_sum
from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_source
from merlin.sched.ir import Call,Const,Kernel,Loop,TensorArg,Var,select,instances,evaluate
from merlin.compile.scheduling.compact_dense import _digest


def selected(rows=7,columns=16,*,prefetch=False):
    p,r,s,b,admission=inputs((rows,6,columns))
    readout=DenseTiledIntegerReadout.from_source_scales(0.5,columns=p.n,tile_columns=p.tile_n)
    def encode(op):
        if type(op)is TiledIntegerStore:
            raw=ADAPTER.encode(Store(op.dram_row,op.dram_col,op.rows,op.cols,op.accumulator_row))
            return replace(raw,instr=readout.instruction_name,args=(*raw.args,('readout_tile',op.readout_tile)))
        return ADAPTER.encode(op)
    def project(call):
        if call.instr==readout.instruction_name:
            raw=ADAPTER.project(replace(call,instr='store',args=tuple((k,v)for k,v in call.args if k!='readout_tile')))
            return TiledIntegerStore(raw.dram_row,raw.dram_col,raw.rows,raw.cols,raw.accumulator_row,readout.identity,call.arg('readout_tile'))
        return ADAPTER.project(call)
    adapter=replace(ADAPTER,encode=encode,project=project)
    transforms=()
    if prefetch:
        from merlin.compile.scheduling.compact_dense import emit_compact_dense
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_operand_hoists
        admission=replace(admission,max_issued_readouts=b.m*b.n,pending_readouts=2,
            readout_configuration='ordered_snapshot_before_store_issue')
        raw=emit_compact_dense(p,b,s,adapter,name='q1',dtypes=('i8','i8','i32'),
            outer_order=('n','m'),output_readout=readout)
        transforms=dense_retained_operand_hoists(raw,adapter,role='weight')
    plan=make_dense_schedule_plan(p,r,s,adapter,source_identity='a'*64,name='q1',
        dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(b.m,b.k,b.n),outer_order=('n','m'),
        output_readout=readout,pipeline_kind='retained_full_k',pipeline_prefetch=False,loop_transforms=transforms)
    producer=schedule_dense_pipeline(p,r,s,adapter,name='q1',dtypes=('i8','i8','i32'),
        source_identity='a'*64,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=admission,
        batch_readouts=False,output_readout=readout,readout_qualification_identity='TEST-original-Q1')
    if prefetch:
        from merlin.compile.scheduling.dense_panel_pipeline import DenseReadoutConfigurationSnapshot,DenseReadyReadoutAdmission
        from merlin.compile.scheduling.dense_ready_readout import schedule_dense_ready_readout
        ready=DenseReadyReadoutAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
            'TEST-original-Q1-producer-RAW','TEST-original-Q1-ordered-store',b.m*b.n,2,
            configuration_snapshot=DenseReadoutConfigurationSnapshot(adapter.configuration_identity,
                adapter.numerical_policy_identity,'TEST-original-scale-snapshot',readout.identity),
            disjoint_load_qualification_identity='TEST-selected-disjoint-LD-EX-ST')
        producer=schedule_dense_ready_readout(producer,p,b,r,s,adapter,ready,
            input_pipeline_row_offset=b.m*b.k*p.tile_m,static_budget=256)
    source=plan_quantized_sum_source(*source_fixture(rows,p.n))
    views=tuple(consumer(h,b.n*p.tile_n)for h in sorted({min(rows,b.m*p.tile_m),rows%(b.m*p.tile_m)}-{0}))
    law=QuantizedSumCompositionAdmission(source.identity,'TEST-independent-effect-law',True,True,True,True,True,True,True,
        consumer_preserves_prefetched_inputs=prefetch)
    result=compose_dense_loaded_sum(source,producer,p,b,project=project,group_path=('outer_n','outer_m'),
        row_group='outer_m',column_group='outer_n',consumers=views,admission=law)
    return result,project


def consumer(rows,cols):
    proof=prove_ordered_loaded_sum(1.0,1.0,1.0,load_lhs_scale=1.0,load_rhs_scale=1.0,store_scale=1.0,relu=True)
    layout=LoadedSumLayout(rows,cols,2,4,0,16,1024)
    parameters=((('scale',1),),(('scale',1),),(('scale',1),('relu',1)))
    contract=LoadedSumContract(_digest(proof),parameters,'TEST-sum-effect',True,True)
    mt,nt=(rows+1)//2,(cols+3)//4
    fields=(('row',Var('m')*2),('column',Var('n')*4),('rows',select('m',mt-1,(rows-1)%2+1,2)),
        ('columns',select('n',nt-1,(cols-1)%4+1,4)),('local',Const(0)))
    calls=tuple(Call(k,fields+tuple((name,Const(v))for name,v in param))for k,param in zip(('initialize','accumulate','store'),parameters))
    kernel=Kernel('sum',tuple(TensorArg(name,(rows,cols),'i8',access)for name,access in
        (('lhs','read'),('rhs','read'),('output','write'))),(Loop('m',mt,(Loop('n',nt,calls),)),))
    def project(call):
        i=('initialize','accumulate','store').index(call.instr);f=dict(call.args)
        return LoadedSumEffect(call.instr,('lhs','rhs','output')[i],f['row'],f['column'],f['rows'],f['columns'],f['local'],16,_digest(proof),
            tuple((name,f[name].value)for name,_ in parameters[i]),'synchronous')
    law=LoadedSumEpochLaw(contract.implementation_identity,contract.scalar_proof_sha256,'TEST-complete-sum',True,True,True,True,True,True)
    plan=plan_loaded_sum_epochs(kernel,layout,contract,project,scalar_proof=proof,target_law=law,max_live_tiles=64,flatten_rows=True)
    return CheckedLoadedSumView(plan,kernel,project,proof)


@pytest.mark.parametrize('rows,prefetch',[(r,False)for r in [7,8,9,10,13,15]]+[(r,True)for r in [7,8,9,10,13,15]])
def test_translation_preserves_every_original_readout_and_exact_group_cover(rows,prefetch):
    result,project=selected(rows,prefetch=prefetch)
    result.revalidate()
    original=[]
    for call,env in instances(result.producer.kernel):
        if call.instr=='store_tiled_i8':original.append(tuple(evaluate(v,env)for _,v in call.args))
    actual=[];groups=[];pending=False
    for call,env in instances(result.kernel):
        if call.instr=='connected_q1_readout':
            actual.append(tuple(evaluate(v,env)for _,v in call.args[:-2]))
            f={k:evaluate(v,env)for k,v in call.args};assert 0<=f['workspace_row']<4 and 0<=f['workspace_column']<8
            pending=True
        elif call.instr=='merlin_dense_panel_drain':pending=False
        elif call.instr=='connected_loaded_sum_group':
            assert not pending
            f={k:evaluate(v,env)for k,v in call.args}
            groups.extend((f['output_row']+i,f['output_column']+j)for i in range(f['rows'])for j in range(f['columns']))
    assert sorted(actual)==sorted(original)
    assert len(groups)==len(set(groups))==rows*16
    assert set(groups)=={(i,j)for i in range(rows)for j in range(16)}
    assert result.kernel.tensor('quantized_workspace').shape==(4,8)
    if prefetch:
        # Every original staged input, compute, store and real completion
        # remains in the same order. Only the Q2 consumer is inserted after
        # each original complete producer rectangle.
        def commands(kernel,composed):
            for call,env in instances(kernel):
                if call.instr=='connected_loaded_sum_group':continue
                if composed and call.instr=='connected_q1_readout':
                    call=replace(call,instr='store_tiled_i8',args=call.args[:-2])
                yield call.instr,tuple((k,evaluate(v,env))for k,v in call.args)
        assert list(commands(result.kernel,True))==list(commands(result.producer.kernel,False))
        with pytest.raises(ValueError,match='prefetched inputs'):
            compose_dense_loaded_sum(result.source,result.producer,result.problem,result.block,
                project=project,group_path=result.group_path,row_group=result.row_group,column_group=result.column_group,
                consumers=result.consumers,admission=replace(result.admission,consumer_preserves_prefetched_inputs=False))
    with pytest.raises(ValueError,match='source, proposal'):
        replace(result,kernel=result.producer.kernel).revalidate()
    with pytest.raises(ValueError,match='admission'):
        replace(result,admission=replace(result.admission,real_full_completion=False)).revalidate()


def test_changed_source_Q1_or_Q2_refuses_before_emission():
    result,_=selected()
    result.source.rewrite.entries['producer']['acc_scale']=0.25
    result.source.rewrite.programs['producer']['entry']['acc_scale']=0.25
    with pytest.raises(ValueError,match='source, stages|original source'):
        result.revalidate()


def test_wrong_group_origin_or_short_source_group_cannot_address_workspace():
    result,project=selected()
    args=dict(group_path=('outer_n','outer_m'),row_group='outer_m',column_group='outer_n',rows=7,columns=16,
        tile_rows=2,tile_columns=4,block_rows=2,block_columns=2)
    kernel=result.producer.kernel
    with pytest.raises(ValueError):_compose_kernel(kernel,project,**dict(args,block_columns=1))
    with pytest.raises(ValueError):_compose_kernel(kernel,project,**dict(args,rows=9))


def test_peeled_group_cannot_read_unwritten_tail_workspace():
    from merlin.sched.ir.expr import free_vars

    result,project=selected(9)
    changed=0
    def corrupt(body):
        nonlocal changed
        output=[]
        for statement in body:
            if type(statement) is Loop:
                output.append(replace(statement,body=corrupt(statement.body)))
            elif statement.instr=='store_tiled_i8' and 'outer_m' not in free_vars(project(statement).dram_row):
                # The original peeled group has only one logical row. A
                # physical two-row store fits the capacity but exceeds its
                # actual initialized Q1 view and must be refused.
                output.append(replace(statement,args=tuple((name,Const(2) if name=='rows' else value)
                                                          for name,value in statement.args)))
                changed+=1
            else:
                output.append(statement)
        return tuple(output)
    kernel=replace(result.producer.kernel,body=corrupt(result.producer.kernel.body))
    assert changed
    with pytest.raises(ValueError,match='private quantized workspace'):
        _compose_kernel(kernel,project,group_path=('outer_n','outer_m'),row_group='outer_m',
            column_group='outer_n',rows=9,columns=16,tile_rows=2,tile_columns=4,block_rows=2,block_columns=2)


def test_eliminated_single_column_group_has_an_exact_zero_origin_only():
    from merlin.compile.scheduling.dense_panel_pipeline import _remap

    result,project=selected(9,8)
    source=result.producer.kernel
    loops=[op for op in source.body if type(op) is Loop and op.var=='outer_n']
    assert len(loops)==1 and loops[0].extent==1
    stripped=replace(source,body=tuple(child for op in source.body for child in
        (_remap(op.body,'outer_n',None,1,0) if op is loops[0] else (op,))))
    args=dict(group_path=('outer_n','outer_m'),row_group='outer_m',column_group='outer_n',
        rows=9,columns=8,tile_rows=2,tile_columns=4,block_rows=2,block_columns=2)
    composed=_compose_kernel(stripped,project,**args)
    original=[tuple(evaluate(value,env)for _,value in call.args)for call,env in instances(stripped)
              if call.instr=='store_tiled_i8']
    actual=[tuple(evaluate(value,env)for _,value in call.args[:-2])for call,env in instances(composed)
            if call.instr=='connected_q1_readout']
    assert actual==original
    groups=[{name:evaluate(value,env)for name,value in call.args}for call,env in instances(composed)
            if call.instr=='connected_loaded_sum_group']
    assert [(v['output_row'],v['output_column'],v['rows'],v['columns'])for v in groups]==[
        (0,0,4,8),(4,0,4,8),(8,0,1,8)]
    with pytest.raises(ValueError,match='one exact original counted loop'):
        _compose_kernel(stripped,project,**dict(args,columns=16))


def test_prefetched_tail_epoch_cannot_publish_an_unwritten_extra_row():
    result,project=selected(9,prefetch=True)
    changed=0
    def corrupt(body):
        nonlocal changed
        output=[]
        for statement in body:
            if type(statement) is Loop:
                output.append(replace(statement,body=corrupt(statement.body)))
            elif statement.instr=='store_tiled_i8' and type(statement.arg('rows')) is not Const:
                output.append(replace(statement,args=tuple((name,Const(2)if name=='rows'else value)
                                                          for name,value in statement.args)))
                changed+=1
            else:output.append(statement)
        return tuple(output)
    invalid=replace(result.producer.kernel,body=corrupt(result.producer.kernel.body))
    assert changed
    with pytest.raises(ValueError,match='exact original panel'):
        _compose_kernel(invalid,project,group_path=result.group_path,row_group=result.row_group,
            column_group=result.column_group,rows=9,columns=16,tile_rows=2,tile_columns=4,
            block_rows=2,block_columns=2,completed_epochs=True)


@pytest.mark.parametrize('rows,prefetch',[(r,p)for r in [7,9,13]for p in [False,True]])
def test_emission_waits_for_real_Q1_writes_then_completes_every_sum_phase(rows,prefetch):
    from test_loaded_sum_epochs import Integers
    import numpy as np
    result,_=selected(rows,prefetch=prefetch)
    q1=(np.arange(rows*16).reshape(rows,16)%256-128).astype(np.int8)
    skip=np.flip(q1,axis=0).copy()
    workspace=np.full((4,8),99,dtype=np.int8)
    output=np.full((rows,16),-99,dtype=np.int8)
    pending=[];acc={};writes=[];phases=[]
    def complete():
        for action in reversed(pending):action()
        pending.clear()
    def producer(kind,f):
        if kind!='connected_q1_readout':return
        pending.append(lambda:workspace.__setitem__((slice(f['workspace_row'],f['workspace_row']+f['rows']),
            slice(f['workspace_column'],f['workspace_column']+f['cols'])),
            q1[f['row']:f['row']+f['rows'],f['col']:f['col']+f['cols']]))
    def consumer(kind,f):
        if kind in ('begin_consumer','end_consumer'):
            assert not pending
            phases.append(kind)
            return
        def finish():
            for i in range(f['rows']):
                for j in range(f['columns']):
                    cell=f['local_byte']+i*f['accumulator_row_bytes']+j*4
                    row,col=f['output_row']+i,f['output_column']+j
                    if kind=='initialize':acc[cell]=int(workspace[f['row']+i,f['column']+j])
                    elif kind=='accumulate':acc[cell]+=int(skip[row,col])
                    else:
                        output[row,col]=min(127,max(0,acc[cell]));writes.append((row,col))
        pending.append(finish)
    report=result.emit(Integers(),producer,consumer,complete)
    assert not pending and phases==['begin_consumer','end_consumer']*(2*((rows+3)//4))
    assert len(writes)==len(set(writes))==rows*16
    np.testing.assert_array_equal(output,np.clip(q1.astype(np.int32)+skip.astype(np.int32),0,127))
    assert report['workspace_shape']==[4,8]


def test_same_checked_consumer_accepts_the_exact_ordinary_command_source():
    from test_quantized_sum_source import command_fixture
    from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_commands
    result,_=selected();source=plan_quantized_sum_commands(command_fixture())
    changed=replace(result,source=source,admission=replace(result.admission,source_identity=source.identity))
    # Admission is source-bound in the original owner identity as well.
    with pytest.raises(ValueError,match='source, proposal'):changed.revalidate()
    actual=compose_dense_loaded_sum(source,result.producer,result.problem,result.block,project=result.project,
        group_path=result.group_path,row_group=result.row_group,column_group=result.column_group,
        consumers=result.consumers,admission=replace(result.admission,source_identity=source.identity))
    actual.revalidate()

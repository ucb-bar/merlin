"""Current tensor observers and the two original quantization stages stay bound."""
from copy import deepcopy

import pytest
from xdsl.dialects import func
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.device_offload import DeviceRewrite
from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_source


def fixture(rows=7, columns=16):
    source = f'''module {{
      func.func @forward(%x:tensor<{rows}x6xi8>,%w:tensor<6x{columns}xi8>,%s:tensor<{rows}x{columns}xi8>)->tensor<{rows}x{columns}xi8> {{
        %e=tensor.empty():tensor<{rows}x{columns}xi8>
        %p=func.call @producer(%x,%w,%e):(tensor<{rows}x6xi8>,tensor<6x{columns}xi8>,tensor<{rows}x{columns}xi8>)->tensor<{rows}x{columns}xi8>
        %f=tensor.empty():tensor<{rows}x{columns}xi8>
        %y=func.call @sum(%p,%s,%f):(tensor<{rows}x{columns}xi8>,tensor<{rows}x{columns}xi8>,tensor<{rows}x{columns}xi8>)->tensor<{rows}x{columns}xi8>
        return %y:tensor<{rows}x{columns}xi8>
      }}
      func.func private @producer(tensor<{rows}x6xi8>,tensor<6x{columns}xi8>,tensor<{rows}x{columns}xi8>)->tensor<{rows}x{columns}xi8>
      func.func private @sum(tensor<{rows}x{columns}xi8>,tensor<{rows}x{columns}xi8>,tensor<{rows}x{columns}xi8>)->tensor<{rows}x{columns}xi8>
    }}'''
    m = parse_mlir_text(source)
    entries = dict(producer=dict(op='matmul',operand_dtype='int8',epilogue=['acc_scale'],acc_scale=0.5),
        sum=dict(op='residual_add',operand_dtype='int8',epilogue=['relu'],source_scales=[1.0]*3))
    r = DeviceRewrite('test-only-original-source',entries=entries,
        programs={s:dict(schema='group_program_v1',entry=deepcopy(e))for s,e in entries.items()},
        arg_access={s:('read','read','write')for s in entries},
        call_buffers={s:[dict(shape=list(t),dtype='i8')for t in shapes]for s,shapes in
            dict(producer=((rows,6),(6,columns),(rows,columns)),sum=((rows,columns),)*3).items()})
    p,c=[o for o in m.walk()if isinstance(o,func.CallOp)]
    return m,r,p,c


def test_bound_current_edge_retains_both_original_numerical_entries():
    m,r,p,c=fixture()
    bound=plan_quantized_sum_source(m,r,p,c)
    bound.revalidate()
    assert bound.receipt['shape']==[7,16]
    assert bound.receipt['producer_result_operand']==0
    assert bound.receipt['entries']==[r.entries['producer'],r.entries['sum']]
    r.entries['producer']['acc_scale']=0.25
    with pytest.raises(ValueError,match='numerical stages'):
        bound.revalidate()


@pytest.mark.parametrize('mutation',['observer','program','format','borrowed_output','stale_call'])
def test_changed_observer_source_format_or_freshness_refuses(mutation):
    m,r,p,c=fixture()
    if mutation=='observer':
        c.parent.insert_op_before(func.CallOp('sum',[p.res[0],c.arguments[1],c.arguments[2]],[c.res[0].type]),c.next_op)
    elif mutation=='program':r.programs['sum']['entry']['source_scales']=[2.0]*3
    elif mutation=='format':r.call_buffers['producer'][0]['dtype']='u8'
    elif mutation=='borrowed_output':c.operands=[p.res[0],c.arguments[1],c.arguments[1]]
    elif mutation=='stale_call':p=p.clone()
    with pytest.raises(ValueError):
        plan_quantized_sum_source(m,r,p,c)


def command_fixture(rows=7,columns=16):
    return dict(abi_version='0.1',target='neutral',tensors=dict(
        A=dict(shape=[rows,6],dtype='i8',role='input'),W=dict(shape=[6,columns],dtype='i8',role='weight'),
        Q1=dict(shape=[rows,columns],dtype='i8',role='output'),S=dict(shape=[rows,columns],dtype='i8',role='input'),Y=dict(shape=[rows,columns],dtype='i8',role='output')),
        commands=[dict(opcode='RES_PACK',operands=dict(src='W',dst='R'),attributes=dict(layout='packed_rhs')),
            dict(opcode='MATMUL_RESIDENT',operands=dict(lhs='A',rhs='R',dst='ACC'),attributes={}),
            dict(opcode='COMMIT',operands=dict(src='ACC',dst='Q1'),attributes=dict(epilogue=['acc_scale'],acc_scale=0.5,output_dtype='i8')),
            dict(opcode='EVICT',operands=dict(handle='R'),attributes={}),
            dict(opcode='RESIDUAL_ADD',operands=dict(lhs='Q1',rhs='S',dst='Y'),attributes=dict(epilogue=['relu'],lhs_scale=1.0,rhs_scale=1.0,source_scales=[1.0]*3,output_dtype='i8'))])


def test_complete_ordinary_command_request_retains_both_source_stages_and_literal_split():
    from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_commands,split_quantized_sum_commands
    cb=command_fixture();source=plan_quantized_sum_commands(cb);source.revalidate()
    p,q=split_quantized_sum_commands(source)
    assert p['commands']==cb['commands'][:-1] and q['commands']==cb['commands'][-1:]
    assert source.receipt['shape']==[7,16]
    assert source.receipt['entries'][0]['acc_scale']==0.5
    assert source.receipt['entries'][1]['source_scales']==[1.0]*3
    cb['commands'][2]['attributes']['acc_scale']=0.25
    with pytest.raises(ValueError,match='source or numerical stages changed'):source.revalidate()


@pytest.mark.parametrize('change',['extra_command','output_alias','edge_twice','wrong_type','missing_source_scales','wrong_resident'])
def test_unclosed_or_changed_command_program_refuses(change):
    from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_commands
    cb=command_fixture()
    if change=='extra_command':cb['commands'].append(deepcopy(cb['commands'][-1]))
    elif change=='output_alias':cb['commands'][-1]['operands']['dst']='Q1'
    elif change=='edge_twice':cb['commands'][-1]['operands']['rhs']='Q1'
    elif change=='wrong_type':cb['tensors']['Q1']['dtype']='i32'
    elif change=='missing_source_scales':cb['commands'][-1]['attributes'].pop('source_scales')
    elif change=='wrong_resident':cb['commands'][-2]['operands']['handle']='other'
    with pytest.raises(ValueError):plan_quantized_sum_commands(cb)

"""The ordinary paired request retains Q1 and the original residual policy."""
from copy import deepcopy
import pytest
from merlin.targetgen import corpus_spec as CS,capsule_golden as CG
from merlin.targetgen.contract.schemas import validate_capsule
from merlin.targetgen.contract.interface_emit import parse_interface_mlir


def source(bias=3,skip=1,side=0):
    metadata=dict(kind='op',source_role='handauthored_compiler_test',source_reference='public exact source test',label='public')
    producer=dict(metadata,name='producer',op='conv2d',Himg=2,Wimg=2,ci=1,N=1,kh=1,kw=1,
        stride=[1,1],padding=[0,0,0,0],epilogue=['bias_add','acc_scale'],acc_scale=0.5,output_dtype='i8')
    residual=dict(metadata,name='sum',op='residual_add',M=4,N=1,epilogue=['relu'],lhs_scale=1.0,rhs_scale=1.0,source_scales=[1.0]*3,bound_lsb=1)
    skip_name='sum_'+('X1','X0')[side]
    entry=dict(metadata,name='pair',kind='model_slice',schema='connected_quantized_sum_entry_v1',op='quantized_producer_sum',
        components=[producer,residual],producer_result_operand=side,input_palette=dict(schema='merlin.component_input_palette.v1',inputs=[
            dict(name=n,axis='linear',values=[v],offset=0)for n,v in [('producer_IFM',0),('producer_W',0),('producer_B',bias),(skip_name,skip)]]))
    binding=CS.CorpusBinding('fixture',16,'int8','i32',True,['L0','L1','L2','L3'],'exact_int',readout_inputs={})
    return entry,binding


@pytest.mark.parametrize('bias,skip,expected,side',[(3,1,3,0),(257,-128,0,0),(-257,127,0,1),(3,-1,1,1)])
def test_public_builder_and_independent_original_stage_evaluators(bias,skip,expected,side):
    entry,binding=source(bias,skip,side);original=deepcopy(entry)
    cap,interface=CS.build(entry,binding);validate_capsule(cap)
    cb=parse_interface_mlir(interface)
    from merlin.llvmlower.quantized_sum_source import plan_quantized_sum_commands,split_quantized_sum_commands
    bound_source=plan_quantized_sum_commands(cb);bound_source.revalidate();p,q=split_quantized_sum_commands(bound_source)
    assert p['commands']==cb['commands'][:-1] and q['commands']==cb['commands'][-1:]
    conv=next(c for c in cb['commands']if c['opcode']=='CONV2D');residual=cb['commands'][-1]
    assert residual['operands'][('lhs','rhs')[side]]==conv['operands']['dst']
    assert conv['attributes']['acc_scale']==0.5 and residual['attributes']['source_scales']==[1.0]*3
    assert CG.golden(cap)=={'sum_Y0':[[expected]]*4}
    original_final,_=CS.build(entry['components'][1],binding)
    assert cap['numeric_policy']==original_final['numeric_policy']
    assert entry==original

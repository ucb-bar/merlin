"""Ordinary source grouping retains current edges, shared callees and Q1/Q2."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest
from xdsl.dialects import func,tensor
from merlin.llvmlower.device_offload import Routed
from merlin.llvmlower.quantized_sum_route import QuantizedSumRoute,freeze_route,require_route,group_quantized_sum,quantized_sum_interface
from merlin.xdsl_dialects._common import text
from test_quantized_sum_source import fixture
from merlin.llvmlower.quantized_sum_source import discover_quantized_sum_sources


def selected(shared=False,side=0):
    module,rewrite,producer,consumer=fixture()
    rewrite=replace(rewrite,routed=tuple(Routed(s,(),(),('i8','i8','i8'))for s in ('producer','sum')),
        signatures={'producer':(),'sum':()},granularity='group')
    if side:consumer.operands=[consumer.arguments[1],consumer.arguments[0],consumer.arguments[2]]
    if shared:
        empty=tensor.EmptyOp([],consumer.res[0].type)
        extra=func.CallOp('sum',[consumer.res[0],consumer.arguments[1-side],empty.tensor],[consumer.res[0].type])
        consumer.parent.insert_ops_before([empty,extra],consumer.next_op)
        rewrite=replace(rewrite,routed=(*rewrite.routed,Routed('sum',(),(),('i8','i8','i8'))))
    return module,rewrite,QuantizedSumRoute((('producer','sum'),))


@pytest.mark.parametrize('shared,side',[(False,0),(True,0),(False,1)])
def test_detached_group_preserves_public_ABI_both_source_stages_and_other_calls(shared,side):
    module,rewrite,route=selected(shared,side);before=text(module,generic=True)
    result=group_quantized_sum(module,rewrite,route)
    assert text(module,generic=True)==before
    result.module.verify()
    row=result.receipt['groups'][0];entry=result.rewrite.entries[row['symbol']]
    assert entry['components']==[rewrite.entries['producer'],rewrite.entries['sum']]
    assert entry['producer_result_operand']==side
    assert result.rewrite.arg_access[row['symbol']]==('read','read','read','write')
    assert len(result.rewrite.call_buffers[row['symbol']])==4
    assert ('sum'in result.rewrite.entries)==shared
    assert'producer'not in result.rewrite.entries
    original=next(o for o in module.body.block.ops if isinstance(o,func.FuncOp))
    current=next(o for o in result.module.body.block.ops if isinstance(o,func.FuncOp))
    assert original.function_type==current.function_type
    assert result.rewrite.programs[row['symbol']]['components']==[rewrite.programs['producer'],rewrite.programs['sum']]
    assert row['external_intermediate_observers']==[]
    assert len([o for o in result.module.walk()if isinstance(o,func.CallOp)])==1+shared


def test_original_route_is_frozen_and_opaque_late_rewrites_refuse():
    route=QuantizedSumRoute((('producer','sum'),))
    device=SimpleNamespace(quantized_sum=route,granularity='group')
    frozen=freeze_route(device);assert require_route(device,frozen)is route
    device.quantized_sum=QuantizedSumRoute(route.source_pairs)
    with pytest.raises(ValueError,match='changed'):require_route(device,frozen)
    device.quantized_sum=route;device.post_offload_transform=lambda:None
    with pytest.raises(ValueError,match='without opaque'):freeze_route(device)
    assert freeze_route(SimpleNamespace())is None


def test_observed_intermediate_and_wrong_original_pair_refuse_without_partial_mutation():
    module,rewrite,route=selected();before=text(module,generic=True)
    with pytest.raises(ValueError):group_quantized_sum(module,rewrite,QuantizedSumRoute((('sum','producer'),)))
    assert text(module,generic=True)==before


@pytest.mark.parametrize('shared,side', [(False,0), (True,0), (False,1)])
def test_discovered_current_edges_feed_the_ordinary_route_without_names(shared,side):
    module,rewrite,_=selected(shared,side)
    before=text(module,generic=True)
    proposals=discover_quantized_sum_sources(module,rewrite)
    assert len(proposals)==1
    source=proposals[0]
    assert source.receipt['producer_result_operand']==side
    route=QuantizedSumRoute(tuple(tuple(p.receipt['components'])for p in proposals))
    result=group_quantized_sum(module,rewrite,route)
    assert len(result.receipt['groups'])==1
    assert text(module,generic=True)==before
    rewrite.programs['sum']['entry']['source_scales']=[2.0]*3
    with pytest.raises(ValueError):source.revalidate()
    assert discover_quantized_sum_sources(module,rewrite)==()


def test_discovery_keeps_observed_intermediates_unfused():
    module,rewrite,_=selected()
    producer=next(o for o in module.walk()if isinstance(o,func.CallOp))
    consumer=next(o for o in module.walk()if isinstance(o,func.CallOp)and o is not producer)
    observer=consumer.clone()
    producer.parent.insert_op_before(observer,consumer)
    assert discover_quantized_sum_sources(module,rewrite)==()


def test_discovery_does_not_choose_an_instance_of_a_shared_producer_symbol():
    module,rewrite,_=selected()
    producer=next(o for o in module.walk()if isinstance(o,func.CallOp))
    extra=producer.clone()
    producer.parent.insert_op_before(extra,producer)
    assert discover_quantized_sum_sources(module,rewrite)==()


def test_original_interfaces_keep_two_readouts_and_only_the_final_observation(monkeypatch):
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir,parse_interface_mlir
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.llvmlower import whole_program
    p=dict(abi_version='0.1',target='neutral',tensors=dict(A=dict(shape=[7,6],dtype='i8',role='input'),W=dict(shape=[6,16],dtype='i8',role='weight'),Y=dict(shape=[7,16],dtype='i8',role='output')),
        commands=[dict(opcode='RES_PACK',operands=dict(src='W',dst='R'),attributes=dict(layout='packed_rhs')),
            dict(opcode='MATMUL_RESIDENT',operands=dict(lhs='A',rhs='R',dst='ACC'),attributes={}),
            dict(opcode='COMMIT',operands=dict(src='ACC',dst='Y'),attributes=dict(epilogue=['acc_scale'],output_dtype='i8',acc_scale=0.5)),
            dict(opcode='EVICT',operands=dict(handle='R'),attributes={})])
    q=dict(abi_version='0.1',target='neutral',tensors={n:dict(shape=[7,16],dtype='i8',role='output'if n=='Y'else'input')for n in ('L','S','Y')},commands=[dict(opcode='RESIDUAL_ADD',operands=dict(lhs='L',rhs='S',dst='Y'),attributes=dict(epilogue=['relu'],source_scales=[1.0]*3,lhs_scale=1.0,rhs_scale=1.0,bound_lsb=0,output_dtype='i8'))])
    original=[deepcopy(p),deepcopy(q)]
    monkeypatch.setattr(whole_program,'group_interface',lambda e,binder,**kwargs:emit_interface_mlir(p if e['op']=='matmul'else q))
    interface=quantized_sum_interface(dict(schema='connected_quantized_sum_entry_v1',producer_result_operand=0,components=[dict(op='matmul'),dict(op='residual_add')]),None,source_reference='original source')
    cb=parse_interface_mlir(interface)
    assert cb['commands'][-1]['operands']['lhs']=='producer_Y'
    assert cb['commands'][-1]['attributes']==q['commands'][0]['attributes']
    assert cb['commands'][2]['attributes']==p['commands'][2]['attributes']
    buffers,reason=logical_buffers(interface,observed_outputs=('sum_Y',));assert buffers is not None,reason
    assert[(b.name,b.kind)for b in buffers]==[('producer_A','input'),('producer_W','input'),('sum_S','input'),('sum_Y','output')]
    assert[p,q]==original


@pytest.mark.parametrize('early',[False,True])
def test_normal_preparation_updates_exact_source_sidecar_and_pins(tmp_path,monkeypatch,early):
    from merlin.llvmlower import device_preparation as P,device_offload as O
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.frontends.linalg_mlir import parse_mlir_file
    module,rewrite,route=selected(shared=True)
    source=tmp_path/'host.mlir';source.write_text(text(module,generic=True))
    rewrite.write_sidecar(tmp_path)
    monkeypatch.setattr(O,'rewrite_prepared_file',lambda *args,**kwargs:rewrite)
    device=DeviceRouting('neutral',tmp_path,'i8','i32',granularity='group',quantized_sum=route)
    frozen=P.freeze_device_offload_selection(device)
    result=P.prepare_device_route(device,source,tmp_path,early_complete_groups=early,selection=frozen)
    calls=[o.callee.root_reference.data for o in parse_mlir_file(result).walk()if isinstance(o,func.CallOp)]
    assert'sum'in calls and len(calls)==2
    sidecar=O.load_sidecar(tmp_path);assert set(sidecar['signatures'])==set(calls)
    assert frozen.sidecar[0].is_file() and result in [p for p,_ in frozen.sources]
    P.require_device_offload_selection(device,frozen)
    with pytest.raises(ValueError,match='opaque prepared-model'):
        P.require_supported_device_preparation(frozen,object())
    changed=replace(device,quantized_sum=replace(route))
    with pytest.raises(ValueError,match='selection changed'):P.require_device_offload_selection(changed,frozen)


def effects_factory_for_snapshot(source,built):
    raise AssertionError('snapshot admission must not invoke the effects owner')


def test_descriptor_snapshot_retains_route_identity_without_inventing_effects(tmp_path):
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.runtime.device_descriptor_results import DeviceDescriptorResults
    route=QuantizedSumRoute((('producer','sum'),))
    device=DeviceRouting('neutral',tmp_path,'i8','i32',granularity='group',quantized_sum=route,
        descriptor_results=True,descriptor_effects=effects_factory_for_snapshot)
    guard=DeviceDescriptorResults(device);guard.revalidate()
    assert guard.identities==()
    object.__setattr__(route,'source_pairs',(('other','sum'),))
    with pytest.raises(ValueError):guard.revalidate()

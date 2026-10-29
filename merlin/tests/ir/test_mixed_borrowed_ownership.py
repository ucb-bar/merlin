"""Independent plural and ordinary result owners share the real native arena."""

import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import MemRefType

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.arena_bind import bind_arena
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCall, BorrowedBufferCalls, BorrowedBufferEffects, from_writers
from merlin.llvmlower.borrowed_call_composition import compose_borrowed_calls
from merlin.llvmlower.descriptor_result_identity import DescriptorResultIdentity
from merlin.llvmlower.device_shim import LogicalBuffer, _memref_typedef, emit_logical_translation_unit
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang
from merlin.xdsl_dialects._common import text


SOURCE = '''module {
 func.func private @pair(%a:tensor<2x3xi8> {bufferization.access="read"},
 %w:tensor<3x4xi8> {bufferization.access="read"},
 %x:tensor<2x4xi32> {bufferization.access="write"},%y:tensor<2x4xi32> {bufferization.access="write"})
 ->(tensor<2x4xi32>,tensor<2x4xi32>) attributes {llvm.emit_c_interface}
 func.func private @head(%a:tensor<1x2x4xi8> {bufferization.access="read"},
 %w:tensor<1x4x5xi8> {bufferization.access="read"},
 %o:tensor<1x2x5xi32> {bufferization.access="write"})->tensor<1x2x5xi32> attributes {llvm.emit_c_interface}
 func.func @forward(%a:tensor<2x3xi8>,%w:tensor<3x4xi8>,%hw:tensor<1x4x5xi8>)->tensor<1x2x5xi32> {
 %e0=tensor.empty():tensor<2x4xi32>
 %e1=tensor.empty():tensor<2x4xi32>
 %p:2=func.call @pair(%a,%w,%e0,%e1):(tensor<2x3xi8>,tensor<3x4xi8>,tensor<2x4xi32>,tensor<2x4xi32>)->(tensor<2x4xi32>,tensor<2x4xi32>)
 %qe=tensor.empty():tensor<2x4xi8>
 %q=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
 ins(%p#0,%p#1:tensor<2x4xi32>,tensor<2x4xi32>) outs(%qe:tensor<2x4xi8>) {
 ^bb0(%x:i32,%y:i32,%unused:i8):
 %sum=arith.addi %x,%y:i32
 %v=arith.trunci %sum:i32 to i8
 linalg.yield %v:i8
 }->tensor<2x4xi8>
 %q3=tensor.expand_shape %q [[0,1],[2]] output_shape [1,2,4]:tensor<2x4xi8> into tensor<1x2x4xi8>
 %he=tensor.empty():tensor<1x2x5xi32>
 %h=func.call @head(%q3,%hw,%he):(tensor<1x2x4xi8>,tensor<1x4x5xi8>,tensor<1x2x5xi32>)->tensor<1x2x5xi32>
 %oe=tensor.empty():tensor<1x2x5xi32>
 %o=linalg.generic {indexing_maps=[affine_map<(b,i,j)->(b,i,j)>,affine_map<(b,i,j)->(b,i,j)>],iterator_types=["parallel","parallel","parallel"]}
 ins(%h:tensor<1x2x5xi32>) outs(%oe:tensor<1x2x5xi32>) {
 ^bb0(%x:i32,%unused:i32):
 %one=arith.constant 1:i32
 %v=arith.addi %x,%one:i32
 linalg.yield %v:i32
 }->tensor<1x2x5xi32>
 return %o:tensor<1x2x5xi32>
 }}'''


def prepare(tmp_path, source=SOURCE):
    plural_effects = BorrowedBufferEffects('independent neutral full-writing plural implementation', True, True, True)
    ordinary_effects = BorrowedBufferEffects('independent neutral synchronous ranked batch implementation', True, True, True)
    writer = FreshTensorWriterContract('pair', 2, (2, 3), 'pair_borrowed', 64,
                                      borrowed_effects=plural_effects, result_arguments=(2, 3))
    module = parse_mlir_text(source)
    rewrite_fresh_tensor_writers(module, (writer,))
    original = tmp_path/'source.mlir'
    original.write_text(text(module, generic=True))
    head = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == 'head')
    types = tuple(head.function_type.inputs)
    call = [dict(role=role, shape=list(typ.get_shape()), dtype=str(typ.get_element_type()))
            for role,typ in zip(('input_0','weight_0','out_0'),types,strict=True)]
    buffers = tuple(LogicalBuffer(name='v'+str(i),role=row['role'],shape=tuple(row['shape']),dtype=row['dtype'],
                                  kind='output' if i == 2 else 'input') for i,row in enumerate(call))
    unit = emit_logical_translation_unit('neutral', {'head':(buffers,call)},
                                         kernel_symbol_for=lambda _: 'batch_product', descriptor_results=True)
    assert not unit.skipped and len(unit.result_identities) == 1
    identity = replace(unit.result_identities[0], borrowed_effects=ordinary_effects)
    implementation = tmp_path/'implementation.c'
    implementation.write_text(unit.text + _memref_typedef(2) + r'''
void batch_product(void *aa,void *bb,void *yy){
 const int8_t*a=aa,*b=bb;int32_t*y=yy;
 for(int i=0;i<2;i++)for(int j=0;j<5;j++){
  int32_t s=0;for(int k=0;k<4;k++)s+=(int32_t)a[i*4+k]*b[k*5+j];y[i*5+j]=s;
 }
}
void _mlir_ciface_pair_borrowed(const merlin_lmemref_2d*a,const merlin_lmemref_2d*w,
 const merlin_lmemref_2d*x,const merlin_lmemref_2d*y){
 if(x->aligned==y->aligned)__builtin_trap();
 for(int i=0;i<2;i++)for(int j=0;j<4;j++){
  int32_t s=0;for(int k=0;k<3;k++)s+=(int32_t)((int8_t*)a->aligned)[a->offset+i*a->strides[0]+k*a->strides[1]]*
   ((int8_t*)w->aligned)[w->offset+k*w->strides[0]+j*w->strides[1]];
  ((int32_t*)x->aligned)[x->offset+i*x->strides[0]+j*x->strides[1]]=s;
  ((int32_t*)y->aligned)[y->offset+i*y->strides[0]+j*y->strides[1]]=s+1;
 }
}
''')
    pins=((implementation,sha256_file(implementation)),)
    states=[{'valid':True},{'valid':True}]
    def plural_valid():
        if not states[0]['valid']:raise ValueError('plural original owner changed')
    def ordinary_valid():
        if not states[1]['valid']:raise ValueError('ordinary original owner changed')
    plural=from_writers(original,(writer,),pins,plural_valid)
    ordinary=BorrowedBufferCalls(original,sha256_file(original),sha256_file(original),
        (BorrowedBufferCall(identity.wrapper_symbol,identity.borrowed_symbol,
         tuple(str(MemRefType(t.get_element_type(),t.get_shape())) for t in types),2,ordinary_effects),),pins,ordinary_valid)
    return original, identity, ordinary, plural, states, implementation


@pytest.mark.parametrize('inline_helpers',[False,True])
def test_real_mixed_rank_plural_and_descriptor_owners_enable_native_arena(tmp_path,upstream_host_tools,inline_helpers):
    original,identity,ordinary,plural,states,implementation=prepare(tmp_path)
    lowered=lower_model(original.read_text(),tmp_path/'lower',targets=(),
        descriptor_result_identities=(identity,),descriptor_result_effects=ordinary,borrowed_buffer_calls=plural,
        index_bits=64,data_layout=_query(clang(),('-O2',)),static_arena=False,hoist_static_allocs=False,
        features=frozenset(('inline_tensor_helpers_before_bufferization',)) if inline_helpers else frozenset())
    assert lowered.stats['descriptor_result_identity']['rewrites']==1
    report=json.loads((tmp_path/'lower/descriptor-result-identity.json').read_text())
    assert report['borrowed_calls']==ordinary.record() and report['additional_borrowed_calls']==plural.record()
    assert len(lowered.stats['borrowed_buffer_lifetime']['accepted_symbols'])==(2 if inline_helpers else 4)
    selected,arena=bind_arena(lowered.ll_path.read_text(),borrowed_lifetime=lowered.borrowed_arena_lifetime)
    assert arena.bound>=3 and arena.refused==(0 if inline_helpers else 2)
    for label,ir in (('heap',lowered.ll_path.read_text()),('arena',selected)):
        ll,so=tmp_path/(label+'.ll'),tmp_path/(label+'.so');ll.write_text(ir)
        compiled=subprocess.run([str(clang()),'-O2','-shared','-fPIC',str(ll),str(implementation),'-lm','-o',str(so)],capture_output=True,text=True)
        assert compiled.returncode==0,compiled.stderr
        invoke=HostModel.load(str(so),image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
        for seed in (1,7):
            a=(np.arange(6)*19-55+seed).astype(np.int8).reshape(2,3)
            w=(np.arange(12)*5-24+seed).astype(np.int8).reshape(3,4)
            hw=(np.arange(20)*3-19+seed).astype(np.int8).reshape(1,4,5)
            p=a.astype(np.int32)@w.astype(np.int32)
            expected=(2*p+1).astype(np.int8).astype(np.int32)[None,:,:]@hw.astype(np.int32)+1
            guarded=np.full(18,-7654321,np.int32);out=guarded[4:-4].reshape(1,2,5)
            inputs=tuple(v.copy() for v in (a,w,hw))
            invoke([(v.ctypes.data,v.shape) for v in (a,w,hw,out)])
            np.testing.assert_array_equal(out,expected)
            assert np.all(guarded[:4]==-7654321) and np.all(guarded[-4:]==-7654321)
            for value,before in zip((a,w,hw),inputs,strict=True):
                np.testing.assert_array_equal(value,before)
    for i,state in enumerate(states):
        state['valid']=False
        with pytest.raises(ValueError,match='original owner changed'):
            bind_arena(selected,borrowed_lifetime=lowered.borrowed_arena_lifetime)
        state['valid']=True
    with pytest.raises(ValueError,match='distinct|symbols'):
        compose_borrowed_calls((ordinary,ordinary))
    with pytest.raises(ValueError,match='source epoch'):
        compose_borrowed_calls((ordinary,replace(plural,input_sha256='0'*64)))
    with pytest.raises(ValueError,match='live original'):
        compose_borrowed_calls((ordinary,plural.record()))


def test_mixed_device_selection_keeps_typed_source_and_effect_requirements(tmp_path):
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
    from merlin.runtime.device_descriptor_results import DeviceDescriptorResults
    from test_device_descriptor_result_build import neutral_effects_owner

    effect=BorrowedBufferEffects('explicit conditional neutral source owner',True,True,True)
    source=HeadBatchSourceRoute('0'*64,effect,shared_input_siblings=True,whole_batch=True)
    route=DeviceRouting('neutral',tmp_path,'i8','i32',granularity='group',sibling_source=source,
                        descriptor_results=True,descriptor_effects=neutral_effects_owner)
    guard=DeviceDescriptorResults(route)
    assert guard.selected and guard.options()=={}  # No identities/effects before actual build custody.
    with pytest.raises(ValueError,match='group route|typed logical sibling'):
        DeviceDescriptorResults(replace(route,granularity='contraction'))
    with pytest.raises(ValueError,match='qualified borrowed-buffer'):
        DeviceDescriptorResults(replace(route,descriptor_effects=None))


def test_layout_selection_composes_with_mixed_owners_and_native_storage(tmp_path, upstream_host_tools):
    """A connected host layout change must survive both independent call owners."""
    import re
    import yaml
    from merlin.llvmlower.host_graph_policy import read_host_graph_policy
    from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA

    source = SOURCE.replace(' %qe=tensor.empty()', '''
 %te0=tensor.empty():tensor<4x2xi32>
 %te1=tensor.empty():tensor<4x2xi32>
 %t0=linalg.transpose ins(%p#0:tensor<2x4xi32>) outs(%te0:tensor<4x2xi32>) permutation=[1,0]
 %t1=linalg.transpose ins(%p#1:tensor<2x4xi32>) outs(%te1:tensor<4x2xi32>) permutation=[1,0]
 %qe=tensor.empty()''')
    q_start = source.index(' %qe=')
    q_end = source.index(' %q3=')
    quantizer = source[q_start:q_end].replace('2x4', '4x2').replace('%p#0,%p#1', '%t0,%t1')
    quantizer = quantizer.replace('%q=', '%qt=') + '''
 %qempty=tensor.empty():tensor<2x4xi8>
 %q=linalg.transpose ins(%qt:tensor<4x2xi8>) outs(%qempty:tensor<2x4xi8>) permutation=[1,0]
'''
    source = source[:q_start] + quantizer + source[q_end:]
    software = tmp_path/'software.yaml'
    software.write_text(yaml.safe_dump(dict(schema='merlin.software_spec.v1', status='reviewed',
        target='neutral', host_optimization=dict(schema=POLICY_SCHEMA,
            permitted_stages=['logical_host'], permitted_optimizations=['propagate_layout'],
            independent_element_order_unobserved=True))))
    policy = read_host_graph_policy(software, target='neutral', entry='forward')
    counts = {}
    for variant in ('original', 'layout'):
        work = tmp_path/variant
        work.mkdir()
        original, identity, ordinary, plural, states, implementation = prepare(work, source)
        lowered = lower_model(original.read_text(), work/'lower', targets=(),
            descriptor_result_identities=(identity,), descriptor_result_effects=ordinary,
            borrowed_buffer_calls=plural, index_bits=64, data_layout=_query(clang(), ('-O2',)),
            static_arena=False, hoist_static_allocs=False,
            features=frozenset(('inline_tensor_helpers_before_bufferization',)),
            graph_optimizations=policy if variant == 'layout' else None)
        assert lowered.stats['descriptor_result_identity']['rewrites'] == 1
        assert len(lowered.stats['borrowed_buffer_lifetime']['accepted_symbols']) == 2
        selected, arena = bind_arena(lowered.ll_path.read_text(), borrowed_lifetime=lowered.borrowed_arena_lifetime)
        assert arena.refused == 0
        counts[variant] = dict(bound=arena.bound, aligned_bound_bytes=arena.naive_total_bytes,
            requested_bytes=sum(map(int, re.findall(r'call ptr @malloc\(i64 (\d+)\)', lowered.ll_path.read_text()))),
            arena_bytes=arena.arena_bytes)
        if variant == 'layout':
            report = lowered.stats['graph_optimizations']
            assert report['automatic_selection']['status'] == 'applied'
            assert report['layout']['before_transpose_bytes'] > report['layout']['after_transpose_bytes']
        for memory, lltext in (('heap', lowered.ll_path.read_text()), ('arena', selected)):
            ll = work/(memory+'.ll');ll.write_text(lltext)
            so = work/(memory+'.so')
            compiled = subprocess.run([str(clang()), '-O2', '-shared', '-fPIC', str(ll), str(implementation),
                '-lm', '-o', str(so)], capture_output=True, text=True)
            assert compiled.returncode == 0, compiled.stderr
            invoke = HostModel.load(str(so), image_policy=PrivateHostImagePolicy(work.resolve()))
            a = (np.arange(6)*19-55).astype(np.int8).reshape(2,3)
            w = (np.arange(12)*5-24).astype(np.int8).reshape(3,4)
            hw = (np.arange(20)*3-19).astype(np.int8).reshape(1,4,5)
            p = a.astype(np.int32)@w.astype(np.int32)
            expected = (2*p+1).astype(np.int8).astype(np.int32)[None,:,:]@hw.astype(np.int32)+1
            guarded = np.full(18, -7654321, np.int32); out=guarded[4:-4].reshape(1,2,5)
            invoke([(v.ctypes.data,v.shape) for v in (a,w,hw,out)])
            np.testing.assert_array_equal(out, expected)
            assert np.all(guarded[:4] == -7654321) and np.all(guarded[-4:] == -7654321)
    assert all(counts['layout'][field] < counts['original'][field]
               for field in ('bound', 'requested_bytes', 'aligned_bound_bytes', 'arena_bytes'))
    (tmp_path/'composition-counts.json').write_text(json.dumps(counts, indent=2)+'\n')

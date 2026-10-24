"""Storage views need typed trace identity and may not change visible aliases."""
import hashlib
import json

import pytest
from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import IndexType, IntegerType, MemRefType

from merlin.frontends.bitcast_views import lower_captured_bitcast_views
from merlin.frontends.linalg_mlir import parse_mlir_text


def fixture(tmp_path,*,source='f32',destination='i32',dtype='int32',escape=False,target='aten.view.dtype'):
    client='func.return %v : tensor<4xDST>' if escape else '''%c=tensor.cast %v : tensor<4xDST> to tensor<4xDST>
    %e=tensor.empty() : tensor<4xDST>
    %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
      ins(%c:tensor<4xDST>) outs(%e:tensor<4xDST>) { ^bb0(%a:DST,%b:DST): linalg.yield %a:DST } -> tensor<4xDST>
    func.return %r:tensor<4xDST>'''
    text='''module {
    func.func private @view(tensor<4xSRC>) -> tensor<4xDST>
    func.func @forward(%x:tensor<4xSRC>) -> tensor<4xDST> {
      %v=func.call @view(%x) {prov.source_node_ids=["p0"]} : (tensor<4xSRC>) -> tensor<4xDST>
      CLIENT
    }}'''.replace('CLIENT',client).replace('SRC',source).replace('DST',destination)
    node=dict(id='p0',op='call_function',target=target,args=[dict(node_id='input'),dict(kind='dtype',value='torch.'+dtype)])
    raw=json.dumps({'graphs':{'prepared':{'nodes':[node]}}}).encode();path=tmp_path/'trace.json';path.write_bytes(raw)
    return parse_mlir_text(text),dict(frontend_trace=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))


@pytest.mark.parametrize('source,destination,dtype',[('f16','i16','int16'),('bf16','i16','int16'),('f32','i32','int32'),('f64','i64','int64'),('i32','f32','float32')])
def test_equal_width_value_clients_get_a_bitcast(tmp_path,source,destination,dtype):
    module,meta=fixture(tmp_path,source=source,destination=destination,dtype=dtype)
    assert lower_captured_bitcast_views(module,meta)==1
    module.verify();assert sum(op.name=='arith.bitcast' for op in module.walk())==1
    assert not any(op.name=='func.call' for op in module.walk())


@pytest.mark.parametrize('case',['escape','width','missing_trace','wrong_target','wrong_dtype','changed_trace'])
def test_unproved_views_stay_opaque(tmp_path,case):
    module,meta=fixture(tmp_path,escape=case=='escape',destination='i16' if case=='width' else 'i32',dtype='float32' if case=='wrong_dtype' else 'int16' if case=='width' else 'int32',target='aten.clone.default' if case=='wrong_target' else 'aten.view.dtype')
    if case=='missing_trace':meta={}
    if case=='changed_trace':
        meta['frontend_trace']['sha256']='0'*64
        with pytest.raises(ValueError,match='digest changed'):lower_captured_bitcast_views(module,meta)
    else:assert lower_captured_bitcast_views(module,meta)==0
    assert sum(op.name=='func.call' for op in module.walk())==1


def test_effectful_tensor_clients_cannot_hide_a_storage_alias(tmp_path):
    module,meta=fixture(tmp_path)
    function=next(op for op in module.walk() if op.name=='func.func' and op.sym_name.data=='forward')
    block=function.body.block;pointer=block.insert_arg(MemRefType(IntegerType(32),[1]),len(block.args))
    function.update_function_type()
    generic=next(op for op in module.walk() if op.name=='linalg.generic')
    zero=arith.ConstantOp.from_int_and_width(0,IndexType())
    block.insert_op_before(zero,generic)
    store=memref.StoreOp.get(generic.body.block.args[0],pointer,[zero.result])
    generic.body.block.insert_op_before(store,generic.body.block.last_op)
    module.verify()
    assert lower_captured_bitcast_views(module,meta)==0
    assert sum(op.name=='func.call' for op in module.walk())==1

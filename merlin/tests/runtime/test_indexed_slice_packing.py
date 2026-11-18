"""Affine input packing keeps bytes and rejects invalid maps and spans."""
import ctypes as ct
from dataclasses import replace
from math import prod
import subprocess

import numpy as np
import pytest

from merlin.compile.layout_c import bounded_dense_read_map, render_indexed_slice_copy
from merlin.semantic_compiler.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps


def source_view(shape, result_shape, coefficients, offsets, dtype='bf16'):
    source = SemanticNode('input', 'input', (), TensorType(shape, dtype, 'raw-bit-pattern'), effect='input')
    rank = len(result_shape)
    view = SemanticNode('view', 'gather', (source.id,), replace(source.type, shape=result_shape),
        index_maps=(IndexMap(rank, coefficients, offsets), pointwise_identity_maps(rank, 1)[0]))
    return source, view


CASES = [
    ((7,), (2,5,7), ((0,0,1),), (0,)),
    ((2,3,10), (2,3,5), ((1,0,0),(0,1,0),(0,0,1)), (0,0,5)),
    ((3,11), (3,5), ((1,0),(0,-2)), (0,10)),
    ((5,7), (7,5), ((0,1),(1,0)), (0,0)),
    ((13,), (3,4), ((2,1),), (2,)),
    ((), (2,3), (), ()),
]


@pytest.mark.parametrize('shape,result_shape,coefficients,offsets', CASES)
@pytest.mark.parametrize('dtype,width', [('bf16',2), ('f32',4), ('i8',1), ('f64',8)])
def test_affine_bytes_slices_and_unchanged_guards(tmp_path, shape, result_shape, coefficients, offsets, dtype, width):
    source, view = source_view(shape, result_shape, coefficients, offsets, dtype)
    text = render_indexed_slice_copy(source, view, 'pack')
    text += '\nint call(void *d,const void *s,size_t first,size_t count){return pack(d,s,first,count);}\n'
    (tmp_path/'pack.c').write_text(text)
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    'pack.c','-o','pack.so'], cwd=tmp_path, check=True, capture_output=True)
    call = ct.CDLL(str(tmp_path/'pack.so')).call
    call.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_size_t]
    count = prod(result_shape)
    raw = np.random.default_rng(9381).integers(0,256,size=prod(shape)*width,dtype=np.uint8).tobytes()
    array = np.frombuffer(raw,dtype=f'V{width}').reshape(shape)
    expected = b''.join(array[tuple(offset + sum(c*d for c,d in zip(row,index,strict=True))
        for row,offset in zip(coefficients,offsets,strict=True))].tobytes() for index in np.ndindex(result_shape))
    original = ct.create_string_buffer(raw)
    output = ct.create_string_buffer(bytes([0xa5])*(count*width+32))
    for first,length in ((0,count),(1,count-2),(count-1,1),(count//3,count//2)):
        ct.memset(output,0xa5,len(output))
        assert call(ct.addressof(output)+16,original,first,length)==0
        assert output.raw[16:16+length*width]==expected[first*width:(first+length)*width]
        assert output.raw[:16]==bytes([0xa5])*16
        assert output.raw[16+length*width:]==bytes([0xa5])*(len(output)-16-length*width)
        assert original.raw[:-1]==raw
    for destination,operand,first,length in (
        (output,original,0,0),(output,original,count,1),(output,original,count-1,2),
        (output,original,2**64-1,1),(original,original,0,1),
        (ct.addressof(original)+width-1,original,0,1),
        (output,None,0,1),(None,original,0,1),(output,ct.c_void_p(2**64-1),0,1)):
        before = output.raw,original.raw
        assert call(destination,operand,first,length)==-1
        assert (output.raw,original.raw)==before


def test_refuse_unbounded_mutating_or_changed_layouts():
    source,view = source_view((7,), (2,5,7), ((0,0,1),), (0,))
    for changed in (
        replace(view,attrs=(('unknown',True),)),
        replace(view,type=replace(view.type,dtype='f32')),
        replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,)),view.index_maps[1])),
        replace(view,index_maps=(replace(view.index_maps[0],coefficients=((0,0,-1),)),view.index_maps[1])),
        replace(view,index_maps=(view.index_maps[0],replace(view.index_maps[1],offsets=(0,0,1)))),
        replace(view,op='transpose')):
        assert bounded_dense_read_map(source,changed) is None
        with pytest.raises(ValueError): render_indexed_slice_copy(source,changed,'pack')
    huge = replace(source,type=replace(source.type,shape=(2**63,)))
    with pytest.raises(ValueError,match='size domain'): render_indexed_slice_copy(huge,view,'pack')
    scalar = replace(view,type=replace(view.type,shape=()))
    assert bounded_dense_read_map(source,scalar) is None
    for name in ('', '_pack', 'pack;evil', 'λpack'):
        with pytest.raises(ValueError): render_indexed_slice_copy(source,view,name)

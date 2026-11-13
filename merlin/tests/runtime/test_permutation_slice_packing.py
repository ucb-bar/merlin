"""Consumer slices retain raw codes and reject unsafe spans before writing."""
import ctypes
from dataclasses import replace
from math import prod
import subprocess

import numpy as np
import pytest

from merlin.compile.layout_c import render_permutation_slice_copy
from merlin.semantic_compiler.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps


def source_view(shape, axes, dtype):
    source = SemanticNode('input', 'input', (), TensorType(shape, dtype, 'raw-bit-pattern'), effect='input')
    result_shape = tuple(shape[axes.index(i)] for i in range(len(shape)))
    read = IndexMap(len(shape), tuple(tuple(int(j == axis) for j in range(len(shape))) for axis in axes),
                    (0,) * len(shape))
    view = SemanticNode('view', 'transpose', (source.id,), replace(source.type, shape=result_shape),
                        index_maps=(read, pointwise_identity_maps(len(shape), 1)[0]))
    return source, view


@pytest.mark.parametrize('shape,axes', [((5,7),(1,0)), ((2,3,5,7),(0,2,1,3)),
    ((2,3,5),(2,0,1)), ((1,9,1),(1,0,2)), ((35,),(0,))])
@pytest.mark.parametrize('dtype,width', [('bf16',2), ('f32',4), ('i8',1), ('f64',8)])
def test_compiled_slice_bytes_bounds_and_overlap(tmp_path, shape, axes, dtype, width):
    source, view = source_view(shape, axes, dtype)
    text = render_permutation_slice_copy(source, view, 'pack')
    text += '\nint call(void *d,const void *s,size_t first,size_t count) { return pack(d,s,first,count); }\n'
    (tmp_path/'packing.c').write_text(text)
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    str(tmp_path/'packing.c'),'-o',str(tmp_path/'packing.so')], check=True, capture_output=True)
    library = ctypes.CDLL(str(tmp_path/'packing.so'))
    call = library.call
    call.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    count = prod(shape)
    raw = np.random.default_rng(1234).integers(0,256,size=count*width,dtype=np.uint8).tobytes()
    original = ctypes.create_string_buffer(raw)
    output = ctypes.create_string_buffer(bytes([0xa5])*(count*width+32))
    expected = np.frombuffer(raw,dtype=f'V{width}').reshape(shape).transpose(np.argsort(axes)).tobytes()
    for first,length in ((0,count), (1,count-2), (count-1,1), (count//3,count//2)):
        ctypes.memset(ctypes.addressof(output),0xa5,len(output))
        assert call(ctypes.addressof(output)+16,ctypes.addressof(original),first,length)==0
        assert output.raw[16:16+length*width] == expected[first*width:(first+length)*width]
        assert output.raw[:16] == bytes([0xa5])*16
        assert output.raw[16+length*width:] == bytes([0xa5])*(len(output)-16-length*width)
        assert original.raw[:len(raw)] == raw
    for first,length in ((0,0),(count,1),(count+1,1),(count-1,2),((1<<64)-1,1)):
        before = output.raw
        assert call(output,original,first,length)==-1
        assert output.raw == before
    for destination,operand in ((original,original), (ctypes.addressof(original)+width,original),
                                (output,None),(None,original), (output,ctypes.c_void_p((1<<64)-1))):
        before = output.raw,original.raw
        assert call(destination,operand,0,1)==-1
        assert (output.raw,original.raw)==before


def test_invalid_maps_types_and_names_refuse():
    source,view=source_view((3,5),(1,0),'bf16')
    for changed in (replace(view,attrs=(('unknown',True),)),
                    replace(view,type=replace(view.type,dtype='f32')),
                    replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,0)),view.index_maps[1]))):
        with pytest.raises(ValueError): render_permutation_slice_copy(source,changed,'pack')
    for name in ('', '_pack', 'pack;evil', 'λpack'):
        with pytest.raises(ValueError): render_permutation_slice_copy(source,view,name)

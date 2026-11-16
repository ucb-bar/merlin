"""Prove repeated storage from affine indexing, independently of any target."""
from dataclasses import replace
from itertools import product
from math import prod

import pytest

from merlin.compile.layout_c import repeated_dense_read_period
from merlin.semantic_compiler.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps


def read(source_shape, shape, coefficients, offsets=None, dtype='i32'):
    source = SemanticNode('base','input',(),TensorType(source_shape,dtype,'raw-bit-pattern'),effect='input')
    mapping = IndexMap(len(shape),coefficients,offsets or (0,)*len(source_shape))
    view = SemanticNode('view','gather',('base',),TensorType(shape,dtype,'raw-bit-pattern'),
                        index_maps=(mapping,pointwise_identity_maps(len(shape),1)[-1]))
    return source,view


@pytest.mark.parametrize('source_shape,shape,coefficients', [
    ((4,),(3,4),((0,1),)), ((2,3),(5,2,3),((0,1,0),(0,0,1))),
    ((1,4),(3,1,4),((0,0,0),(0,0,1))),
    ((1,4),(3,4),((0,0),(0,1))), ((1,),(3,4),((0,0),)),
])
@pytest.mark.parametrize('dtype', ['bf16','u8','i64'])
def test_period_matches_every_address(source_shape,shape,coefficients,dtype):
    source,view = read(source_shape,shape,coefficients,dtype=dtype)
    period = repeated_dense_read_period(source,view)
    assert period == prod(source_shape)
    strides = [prod(source_shape[i+1:]) for i in range(len(source_shape))]
    addresses = []
    for coordinates in product(*(range(n) for n in shape)):
        address = sum(sum(c*x for c,x in zip(row,coordinates,strict=True))*stride
                      for row,stride in zip(coefficients,strides,strict=True))
        addresses.append(address)
    assert addresses == [index % period for index in range(prod(shape))]


@pytest.mark.parametrize('source_shape,shape,coefficients,offsets', [
    ((4,),(4,),((1,),),(0,)),  # no repetition
    ((8,),(3,4),((0,1),),(1,)),  # slice, not the whole physical sequence
    ((4,),(3,4),((0,-1),),(3,)),
    ((8,),(3,4),((0,2),),(0,)),
    ((2,3),(5,3,2),((0,0,1),(0,1,0)),(0,0)),
    ((4,),(3,4),((1,1),),(0,)),  # leaves bounds
])
def test_nonperiodic_or_changed_storage_is_not_admitted(source_shape,shape,coefficients,offsets):
    assert repeated_dense_read_period(*read(source_shape,shape,coefficients,offsets)) is None


def test_changed_attributes_dtype_or_write_map_refuse():
    source,view = read((4,),(3,4),((0,1),))
    for changed in (replace(view,attrs=(('unknown',True),)),
                    replace(view,type=replace(view.type,dtype='f32')),
                    replace(view,index_maps=(view.index_maps[0],IndexMap(2,((0,1),(1,0)),(0,0))))):
        assert repeated_dense_read_period(source,changed) is None

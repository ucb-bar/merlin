"""Dense alias proofs agree with an independently enumerated address stream."""
from dataclasses import replace
from itertools import product

import pytest

from merlin.compiler.lower.c.layout import dense_read_preserves_order
from merlin.compiler.semantic.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps


def read(source_shape, shape, coefficients, offsets=None, dtype='i32'):
    source = SemanticNode('base','input',(),TensorType(source_shape,dtype,'raw-bit-pattern'),effect='input')
    mapping = IndexMap(len(shape),coefficients,offsets or (0,)*len(source_shape))
    view = SemanticNode('view','gather',('base',),TensorType(shape,dtype,'raw-bit-pattern'),
                        index_maps=(mapping,pointwise_identity_maps(len(shape),1)[-1]))
    return source,view


@pytest.mark.parametrize('dtype', ['bf16', 'u8', 'i64'])
@pytest.mark.parametrize('source_shape,shape,coefficients', [
    ((6,), (2, 3), ((3, 1),)),
    ((2, 3), (1, 2, 3), ((0, 1, 0), (0, 0, 1))),
    ((1, 6), (6, 1), ((0, 1), (1, 0))),
    ((1,), (1, 1), ((-17, 23),)),
])
def test_complete_dense_address_order_is_recognized(source_shape, shape, coefficients, dtype):
    source, view = read(source_shape, shape, coefficients, dtype=dtype)
    assert dense_read_preserves_order(source, view)


def test_small_affine_maps_match_explicit_address_enumeration():
    for shape in ((6,), (2, 3), (3, 2), (1, 6), (6, 1)):
        for coefficients in product(range(-2, 4), repeat=len(shape)):
            for offset in range(6):
                source, view = read((6,), shape, (coefficients,), (offset,))
                addresses = [offset + sum(c * x for c, x in zip(coefficients, point, strict=True))
                             for point in product(*(range(extent) for extent in shape))]
                assert dense_read_preserves_order(source, view) == (addresses == list(range(6)))


def test_nontrivial_permutation_repetition_and_slices_do_not_alias():
    for source_shape, shape, coefficients, offsets in (
        ((2, 3), (3, 2), ((0, 1), (1, 0)), (0, 0)),
        ((6,), (2, 3), ((0, 1),), (0,)),
        ((6,), (6,), ((-1,),), (5,)),
        ((6,), (3,), ((1,),), (1,)),
        ((3,), (2, 3), ((0, 1),), (0,)),
    ):
        assert not dense_read_preserves_order(*read(source_shape, shape, coefficients, offsets))


def test_transpose_legality_encoding_and_write_maps_remain_required():
    source, view = read((1, 6), (6, 1), ((0, 1), (1, 0)))
    assert dense_read_preserves_order(source, replace(view, op='transpose'))
    for changed in (
        replace(view, attrs=(('unknown', True),)),
        replace(view, type=replace(view.type, dtype='f32')),
        replace(view, type=replace(view.type, numerical_policy='different')),
        replace(view, index_maps=(view.index_maps[0], IndexMap(2, ((0, 1), (1, 0)), (0, 0)))),
        replace(view, op='input', inputs=(), effect='input', index_maps=()),
    ):
        assert not dense_read_preserves_order(source, changed)
    # An affine reshape-shaped gather is valid, but rank-changing transpose is not.
    source, view = read((6,), (2, 3), ((3, 1),))
    assert dense_read_preserves_order(source, view)
    assert not dense_read_preserves_order(source, replace(view, op='transpose'))

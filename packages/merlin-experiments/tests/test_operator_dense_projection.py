"""Streaming intake preserves the original exclusion, not initializer numerics."""

import hashlib
import io

import pytest
import test_connected_integer_bounds as B
from merlin_experiments.phase0 import operator_dense_projection as P
from merlin_experiments.phase0 import operator_typed_exclusion as T

LIMIT = 1 << 20


def source(literal, typ="tensor<4xi8>"):
    return B.program().replace("%z =", f"%weights = arith.constant dense<{literal}> : {typ}\n    %z =").encode()


def project(raw, **kw):
    return P.project(io.BytesIO(raw), max_input_bytes=4*LIMIT, max_compact_bytes=LIMIT, **kw)


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 31, 65536])
@pytest.mark.parametrize("literal,typ", [
    ('"0x00807fff"', "tensor<4xi8>"),
    ('"0x000000000000803f0000807f0000c07f"', "tensor<4xf32>"),
    ('[[1, -2], [3, 4]]', "tensor<2x2xi32>"),
    ('0', "tensor<4xi8>"), ('-0.0', "tensor<4xf32>"),
])
def test_exact_canonical_identity_matches_existing_registered_parser(literal, typ, chunk):
    raw = source(literal, typ)
    expected = T.signatures(raw, max_bytes=LIMIT)
    projected = project(raw, chunk_bytes=chunk)
    assert projected.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert projected.source_bytes == len(raw) and projected.projected_constants == 1
    assert T.signatures(projected.compact, max_bytes=LIMIT) == expected
    assert P.MARKER.encode() in projected.compact and b'dense_resource<' in projected.compact


def test_large_hex_payload_is_not_retained_or_expanded_and_full_hash_is_checked():
    raw = source('"0x' + '8f' * LIMIT + '"', f"tensor<{LIMIT}xi8>")
    projected = project(raw, chunk_bytes=4096)
    assert len(projected.compact) < 8192
    assert projected.hexadecimal_payload_bytes == LIMIT
    assert projected.source_sha256 == hashlib.sha256(raw).hexdigest()
    # Existing affordable parser independently reconstructs the same identities.
    assert T.signatures(projected.compact, max_bytes=LIMIT) == T.signatures(raw, max_bytes=3*LIMIT)


def test_constant_used_by_the_actual_integer_contraction_retains_exact_identity():
    raw = source('"0x010203040506"', 'tensor<3x2xi8>')
    raw = raw.replace(b', %w: tensor<3x2xi8>', b'').replace(b'ins(%x, %w', b'ins(%x, %weights')
    expected = T.signatures(raw, max_bytes=LIMIT)
    assert len(expected) == 1
    assert T.signatures(project(raw).compact, max_bytes=LIMIT) == expected
    # Deliberate conservative exclusion: original initializer values are not a
    # means of making a protected contraction eligible for publication.
    changed = raw.replace(b'010203040506', b'080706050403')
    assert T.signatures(project(changed).compact, max_bytes=LIMIT) == expected


@pytest.mark.parametrize('operation,inputs,weights,outputs', [
    ('linalg.matmul', '2x3', '3x2', '2x2'),
    ('linalg.batch_matmul', '2x3x4', '2x4x2', '2x3x2'),
    ('linalg.conv_2d_nhwc_hwcf', '1x5x5x2', '3x3x2x4', '1x3x3x4'),
])
def test_registered_named_float_contractions_with_live_constant_weights(operation, inputs, weights, outputs):
    count = 1
    for dim in weights.split('x'):
        count *= int(dim)
    extra = (' {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}'
             if operation.startswith('linalg.conv') else '')
    raw = (f'module {{\n func.func @f(%x: tensor<{inputs}xf32>) -> tensor<{outputs}xf32> {{\n'
           f'%w = arith.constant dense<"0x{"0000803f" * count}"> : tensor<{weights}xf32>\n'
           f'%z = arith.constant 0.0 : f32\n %e = tensor.empty() : tensor<{outputs}xf32>\n'
           f'%i = linalg.fill ins(%z : f32) outs(%e : tensor<{outputs}xf32>) -> tensor<{outputs}xf32>\n'
           f'%r = {operation}{extra} ins(%x, %w : tensor<{inputs}xf32>, tensor<{weights}xf32>) '
           f'outs(%i : tensor<{outputs}xf32>) -> tensor<{outputs}xf32>\n'
           f'return %r : tensor<{outputs}xf32>\n }}\n}}').encode()
    if operation == 'linalg.batch_matmul':
        # The retained parser lacks this custom named form. Streaming must
        # preserve that refusal; generic batched contractions remain separate.
        with pytest.raises(Exception):
            T.signatures(raw, max_bytes=LIMIT)
        with pytest.raises(Exception):
            project(raw)
    else:
        assert T.signatures(project(raw).compact, max_bytes=LIMIT) == T.signatures(raw, max_bytes=LIMIT)


@pytest.mark.parametrize("literal,typ", [
    ('"0x123"', "tensor<2xi8>"), ('"0xgg"', "tensor<1xi8>"),
    ('"0x00"', "tensor<2xi8>"), ('[1, 2, 3]', "tensor<2xi8>"),
    ('[999999999999999999999999999999999]', "tensor<1xi8>"),
    ('not_a_number', "tensor<1xf32>"), ('[]', "tensor<1xf32>"),
])
def test_malformed_or_incomplete_original_initializers_do_not_gain_an_identity(literal, typ):
    with pytest.raises(Exception):
        project(source(literal, typ))


def test_scalar_region_initializers_are_never_elided_even_when_unused():
    raw = B.program().replace("%p = arith.muli", '%private = arith.constant dense<"0x01"> : tensor<1xi8>\n'
                             '      %p = arith.muli').encode()
    with pytest.raises(ValueError, match="immediate function-body"):
        project(raw)


def test_reserved_marker_comments_strings_and_invalid_original_regions_refuse():
    raw = source('"0x01020304"')
    for changed in [raw + b'\n// ' + P.HANDLE.encode(),
                    raw.replace(b'func.func @', b'"unknown.invalid" @')]:
        with pytest.raises(Exception):
            project(changed, chunk_bytes=7)
    commented = raw.replace(b'%weights = arith.constant', b'// %weights = arith.constant')
    assert project(commented).projected_constants == 0


def test_arithmetic_and_shape_mutations_remain_distinct_after_streaming():
    raw = source('"0x01020304"')
    baseline = T.signatures(project(raw).compact, max_bytes=LIMIT)
    for changed in [raw.replace(b'arith.muli', b'arith.addi'),
                    raw.replace(b'tensor<2x3xi8>', b'tensor<2x4xi8>').replace(b'tensor<3x2xi8>', b'tensor<4x2xi8>')]:
        assert T.signatures(project(changed).compact, max_bytes=LIMIT) != baseline
    with pytest.raises(Exception):
        project(raw.replace(b'(m,k)', b'(k,m)'))


def test_every_independent_input_and_allocation_limit_refuses_without_partial_projection():
    raw = source('"0x01020304"')
    for kwargs in [dict(max_input_bytes=1, max_compact_bytes=LIMIT),
                   dict(max_input_bytes=4*LIMIT, max_compact_bytes=1),
                   dict(max_input_bytes=4*LIMIT, max_compact_bytes=LIMIT, max_constants=1)]:
        trial = (raw if 'max_constants' not in kwargs else raw.replace(
            b'%z =', b'%second = arith.constant dense<0> : tensor<1xi8>\n %z ='))
        with pytest.raises(ValueError):
            P.project(io.BytesIO(trial), **kwargs)
    with pytest.raises(ValueError, match='literal budget'):
        project(source('[1, 2, 3, 4]'), max_literal_bytes=2)
    with pytest.raises(ValueError, match='allocation budget'):
        project(source('[1, 2, 3, 4]'), max_retained_literal_bytes=2)
    with pytest.raises(ValueError, match='memory budget'):
        project(source('[1, 2, 3, 4]'), max_decoded_bytes=2)

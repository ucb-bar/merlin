"""Rank-independent source proofs reject unsafe floating-to-integer producer chains."""
from dataclasses import asdict

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_composite_math import (
    recognize_static_composite_math_body as recognize,
)
from merlin.frontends.linalg_composite_math import (
    validate_serialized_static_composite_math_pattern as reopen,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def source(shape='2x7', lower=-100., upper=111., rounding='math.roundeven'):
    dims = ', '.join(f'd{i}' for i, _ in enumerate(shape.split('x')))
    mapping = f'affine_map<({dims}) -> ({dims})>'
    iterators = ', '.join('#linalg.iterator_type<parallel>' for _ in shape.split('x'))

    def generic(name, operand, init, output_type, body):
        return f'''%{name} = "linalg.generic"(%{operand}, %{init}) <{{
          indexing_maps = [{mapping}, {mapping}], iterator_types = [{iterators}],
          operandSegmentSizes = array<i32: 1, 1>}}> ({{
          ^bb0(%v: f32, %old: {output_type}):
            {body}
        }}) : (tensor<{shape}xf32>, tensor<{shape}x{output_type}>) -> tensor<{shape}x{output_type}>'''

    rounded = generic('rounded', 'x', 'finit', 'f32', f'''
        %r = "{rounding}"(%v) : (f32) -> f32
        "linalg.yield"(%r) : (f32) -> ()''')
    clamped = generic('clamped', 'rounded', 'finit', 'f32', f'''
        %lo = "arith.constant"() <{{value = {lower:.8e} : f32}}> : () -> f32
        %a = "arith.maximumf"(%v, %lo) : (f32, f32) -> f32
        %hi = "arith.constant"() <{{value = {upper:.8e} : f32}}> : () -> f32
        %b = "arith.minimumf"(%a, %hi) : (f32, f32) -> f32
        "linalg.yield"(%b) : (f32) -> ()''')
    cast = generic('cast', 'clamped', 'iinit', 'i8', '''
        %r = "arith.fptosi"(%v) : (f32) -> i8
        "linalg.yield"(%r) : (i8) -> ()''')
    text = f'''builtin.module {{
      func.func @f(%x: tensor<{shape}xf32>, %finit: tensor<{shape}xf32>,
                   %iinit: tensor<{shape}xi8>) -> tensor<{shape}xi8> {{
        {rounded}
        {clamped}
        {cast}
        func.return %cast : tensor<{shape}xi8>
      }}
    }}'''
    return text


@pytest.mark.parametrize('shape', ['5', '2x7', '1x3x5', '2x1x3x4'])
def test_finite_intervals_and_safe_casts_generalize_across_shapes(shape):
    ops = tuple(mq.walk(mq.parse(source(shape)), 'linalg.generic'))
    clamp, cast = recognize(ops[1]), recognize(ops[2])
    assert clamp.operation == 'clamp_literal_interval_f32'
    assert cast.operation == 'fptosi_i8_of_rounded_literal_interval'
    assert cast.literal_bits == clamp.literal_bits
    assert reopen(asdict(cast)) == cast


@pytest.mark.parametrize('lower,upper', [(-129., 127.), (-128., 128.), (-0.5, 100.), (2., 1.)])
def test_unsafe_cast_intervals_refuse(lower, upper):
    ops = tuple(mq.walk(mq.parse(source(lower=lower, upper=upper)), 'linalg.generic'))
    with pytest.raises(InvalidLinalgPattern):
        recognize(ops[2])


def test_direct_cast_or_wrong_rounding_cannot_claim_safe_integer_domain():
    text = source().replace('"linalg.generic"(%clamped, %iinit)', '"linalg.generic"(%x, %iinit)')
    with pytest.raises(InvalidLinalgPattern):
        recognize(tuple(mq.walk(mq.parse(text), 'linalg.generic'))[2])
    with pytest.raises(InvalidLinalgPattern):
        recognize(tuple(mq.walk(mq.parse(source(rounding='math.floor')), 'linalg.generic'))[2])


def test_serialized_proof_cannot_inflate_legal_cast_range():
    pattern = recognize(tuple(mq.walk(mq.parse(source()), 'linalg.generic'))[2])
    forged = asdict(pattern)
    forged['literal_bits'] = ('0xc3000000', '0x43000000')  # -128, +128
    with pytest.raises(InvalidLinalgPattern, match='signed i8'):
        reopen(forged)

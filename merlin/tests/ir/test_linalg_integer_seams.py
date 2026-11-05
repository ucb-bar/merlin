"""Closed integer seams must not admit nearby arithmetic or changed dataflow."""
import pytest
from merlin.common import mlir_query as mq
from merlin.frontends.linalg_integer_seams import recognize, InvalidLinalgPattern


def source(xor=False):
    dtype = 'i8' if xor else 'i32'
    constants = '%mask = arith.constant 1 : i8' if xor else '%lo = arith.constant -128 : i32\n%hi = arith.constant 127 : i32'
    body = '%v = arith.xori %x, %mask : i8' if xor else '%a = arith.maxsi %x, %lo : i32\n%b = arith.minsi %a, %hi : i32\n%v = arith.trunci %b : i32 to i8'
    return f'''builtin.module {{ func.func @f(%input: tensor<2x3x{dtype}>, %init: tensor<2x3xi8>) -> tensor<2x3xi8> {{
    {constants}
    %out = linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>], iterator_types=["parallel","parallel"]}} ins(%input:tensor<2x3x{dtype}>) outs(%init:tensor<2x3xi8>) {{
    ^bb0(%x:{dtype}, %unused:i8):
    {body}
    linalg.yield %v : i8
    }} -> tensor<2x3xi8>
    func.return %out : tensor<2x3xi8>
    }} }}'''


def pattern(text):
    return recognize(next(mq.walk(mq.parse(text), 'linalg.generic')))


@pytest.mark.parametrize('xor',[False,True])
def test_original_typed_seams(xor):
    result=pattern(source(xor))
    assert result.operation == ('xor_low_bit_i8' if xor else 'saturating_i32_to_i8')
    assert result.shape == (2,3)


@pytest.mark.parametrize('xor,before,after',[
    (False,'constant -128','constant -127'),(False,'constant 127','constant 126'),
    (False,'arith.maxsi','arith.maxui'),(False,'arith.minsi %a','arith.minsi %x'),
    (False,'arith.trunci %b','arith.trunci %x'),(True,'constant 1','constant 2'),
    (True,'arith.xori','arith.ori'),(True,'arith.xori %x','arith.xori %unused')])
def test_close_but_different_programs_refuse(xor,before,after):
    with pytest.raises(InvalidLinalgPattern):
        pattern(source(xor).replace(before,after))

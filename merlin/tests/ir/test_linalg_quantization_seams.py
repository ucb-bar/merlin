"""Typed quantization proofs refuse changed math, maps and initialization reads."""
import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_quantization_seams import recognize
from merlin.transforms.dense_ir import InvalidLinalgPattern, checked_static_linalg_shell

SOURCE = '''builtin.module {
 func.func @forward(%data:tensor<2x3xf32>, %scale:tensor<f32>, %zero:tensor<i64>, %init:tensor<2x3xi8>) -> tensor<2x3xi8> {
 %empty = tensor.empty() : tensor<f32>
 %recip = linalg.generic {indexing_maps=[affine_map<()->()>,affine_map<()->()>],iterator_types=[]} ins(%scale:tensor<f32>) outs(%empty:tensor<f32>) {
 ^bb0(%s:f32,%unused:f32):
 %one = arith.constant 1.0 : f32
 %inverse = arith.divf %one, %s : f32
 linalg.yield %inverse : f32
 } -> tensor<f32>
 %lo = arith.constant -128.0 : f32
 %hi = arith.constant 127.0 : f32
 %out = linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->()>,affine_map<(d0,d1)->()>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]} ins(%data,%recip,%zero:tensor<2x3xf32>,tensor<f32>,tensor<i64>) outs(%init:tensor<2x3xi8>) {
 ^bb0(%x:f32,%r:f32,%z:i64,%old:i8):
 %product = arith.mulf %x, %r : f32
 %rounded = math.roundeven %product : f32
 %zf = arith.sitofp %z : i64 to f32
 %shifted = arith.addf %rounded, %zf : f32
 %low = arith.maximumf %shifted, %lo : f32
 %high = arith.minimumf %low, %hi : f32
 %byte = arith.fptosi %high : f32 to i8
 linalg.yield %byte : i8
 } -> tensor<2x3xi8>
 func.return %out : tensor<2x3xi8>
 } }
'''


def ops(text=SOURCE):
    return tuple(mq.walk(mq.parse(text), 'linalg.generic'))


def test_exact_two_stage_source_and_rank_zero_opt_in():
    reciprocal, quant = ops()
    assert recognize(reciprocal).shape == ()
    assert recognize(quant).input_shapes == ((2, 3), (), ())
    assert recognize(quant).literal_bits == ('0xc3000000', '0x42fe0000')
    with pytest.raises(InvalidLinalgPattern):
        checked_static_linalg_shell(reciprocal)


@pytest.mark.parametrize('index,before,after', [
    (0,'constant 1.0','constant 2.0'),
    (0,'arith.divf %one, %s','arith.divf %one, %unused'),
    (1,'constant -128.0','constant -127.0'),
    (1,'constant 127.0','constant 126.0'),
    (1,'math.roundeven','math.floor'),
    (1,'arith.mulf %x, %r','arith.addf %x, %r'),
    (1,'arith.minimumf %low','arith.minimumf %shifted'),
    (1,'arith.fptosi %high','arith.fptosi %rounded'),
    (1,'linalg.yield %byte','linalg.yield %old'),
    (1,'arith.maximumf','arith.maxnumf'),
    (1,'affine_map<(d0,d1)->(d0,d1)>','affine_map<(d0,d1)->(d1,d0)>'),
])
def test_nearby_programs_are_not_admitted(index, before, after):
    with pytest.raises(InvalidLinalgPattern):
        recognize(ops(SOURCE.replace(before, after))[index])

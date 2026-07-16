"""lower_quant_ext is granularity-general: per_tensor / per_channel / per_group dequant all lower to a
generic dequant linalg.generic (target-agnostic → f32), not just per_channel."""
from __future__ import annotations

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import lower_quant_ext


def _mod(kind, attrs, wty, sty, zty, oty):
    return parse_mlir_text(f'''module {{
  func.func @dq(%w: {wty}, %s: {sty}, %z: {zty}) -> {oty} {{
    %r = "quant_ext.dequantize_{kind}"(%w, %s, %z) <{{{attrs}}}> : ({wty}, {sty}, {zty}) -> {oty}
    return %r : {oty}
  }}
}}''')


CASES = {
    "per_tensor": (dict(attrs='input_dtype = "i8"', wty="tensor<2x8xi8>", sty="tensor<f32>",
                        zty="tensor<i32>", oty="tensor<2x8xf32>"),
                   "affine_map<(d0, d1) -> ()>"),
    "per_channel": (dict(attrs='axis = 0 : i64, input_dtype = "i8"', wty="tensor<2x8xi8>",
                         sty="tensor<2xf32>", zty="tensor<2xi32>", oty="tensor<2x8xf32>"),
                    "affine_map<(d0, d1) -> (d0)>"),
    "per_group": (dict(attrs='group_size = 4 : i64, axis = 1 : i64, input_dtype = "i8"',
                       wty="tensor<2x8xi8>", sty="tensor<2x2xf32>", zty="tensor<2x2xi32>",
                       oty="tensor<2x8xf32>"),
                  "affine_map<(d0, d1) -> (d0, (d1 floordiv 4))>"),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_dequant_granularity_lowers_generically(kind):
    args, expected_scale_map = CASES[kind]
    mod = _mod(kind, **args)
    n = lower_quant_ext(mod)
    assert n == 1, f"{kind} not lowered"
    mod.verify()                                   # xDSL structural verification passes
    text = str(mod)
    assert "quant_ext." not in text                # fully rewritten to standard dialects
    gen = next(o for o in mod.walk() if o.name == "linalg.generic")
    # scale operand (2nd input) indexing map encodes the granularity
    assert str(gen.indexing_maps.data[1]) == expected_scale_map
    # dequant body: (sitofp(w) - sitofp(zp)) * scale
    body_ops = [o.name for o in gen.body.blocks[0].ops]
    assert body_ops == ["arith.sitofp", "arith.sitofp", "arith.subf", "arith.mulf", "linalg.yield"]


def test_multiple_granularities_in_one_module():
    # a module mixing per_channel + per_group lowers both
    mod = parse_mlir_text('''module {
  func.func @m(%w0: tensor<2x8xi8>, %s0: tensor<2xf32>, %z0: tensor<2xi32>,
               %w1: tensor<2x8xi8>, %s1: tensor<2x2xf32>, %z1: tensor<2x2xi32>)
               -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    %a = "quant_ext.dequantize_per_channel"(%w0, %s0, %z0) <{axis = 0 : i64}> : (tensor<2x8xi8>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x8xf32>
    %b = "quant_ext.dequantize_per_group"(%w1, %s1, %z1) <{group_size = 4 : i64, axis = 1 : i64}> : (tensor<2x8xi8>, tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x8xf32>
    return %a, %b : tensor<2x8xf32>, tensor<2x8xf32>
  }
}''')
    assert lower_quant_ext(mod) == 2
    mod.verify()
    assert "quant_ext." not in str(mod)

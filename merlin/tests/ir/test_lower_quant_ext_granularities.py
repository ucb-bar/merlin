# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""lower_quant_ext is granularity-general: per_tensor / per_channel / per_group dequant all lower to a
generic dequant linalg.generic (target-agnostic → f32), not just per_channel."""

from __future__ import annotations

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import lower_quant_ext, prune_dead_pure_tensor_ops


def _mod(kind, attrs, wty, sty, zty, oty):
    return parse_mlir_text(f"""module {{
  func.func @dq(%w: {wty}, %s: {sty}, %z: {zty}) -> {oty} {{
    %r = "quant_ext.dequantize_{kind}"(%w, %s, %z) <{{{attrs}}}> : ({wty}, {sty}, {zty}) -> {oty}
    return %r : {oty}
  }}
}}""")


CASES = {
    "per_tensor": (
        dict(
            attrs='input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<f32>",
            zty="tensor<i32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> ()>",
    ),
    "per_channel": (
        dict(
            attrs='axis = 0 : i64, input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<2xf32>",
            zty="tensor<2xi32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> (d0)>",
    ),
    "per_group": (
        dict(
            attrs='group_size = 4 : i64, axis = 1 : i64, input_dtype = "i8"',
            wty="tensor<2x8xi8>",
            sty="tensor<2x2xf32>",
            zty="tensor<2x2xi32>",
            oty="tensor<2x8xf32>",
        ),
        "affine_map<(d0, d1) -> (d0, (d1 floordiv 4))>",
    ),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_dequant_granularity_lowers_generically(kind):
    args, expected_scale_map = CASES[kind]
    mod = _mod(kind, **args)
    n = lower_quant_ext(mod)
    assert n == 1, f"{kind} not lowered"
    mod.verify()  # xDSL structural verification passes
    text = str(mod)
    assert "quant_ext." not in text  # fully rewritten to standard dialects
    gen = next(o for o in mod.walk() if o.name == "linalg.generic")
    assert gen.attributes["prov.transforms"].data == f"dequant_{kind}"
    # scale operand (2nd input) indexing map encodes the granularity
    assert str(gen.indexing_maps.data[1]) == expected_scale_map
    # Per-channel promotion/subtraction precedes conversion in the signed-i8/i32/f32 contract.
    body_ops = [o.name for o in gen.body.blocks[0].ops]
    if kind == "per_channel":
        assert body_ops == ["arith.extsi", "arith.subi", "arith.sitofp", "arith.mulf", "linalg.yield"]
    else:
        assert body_ops == ["arith.sitofp", "arith.sitofp", "arith.subf", "arith.mulf", "linalg.yield"]


def test_multiple_granularities_in_one_module():
    # a module mixing per_channel + per_group lowers both
    mod = parse_mlir_text("""module {
  func.func @m(%w0: tensor<2x8xi8>, %s0: tensor<2xf32>, %z0: tensor<2xi32>,
               %w1: tensor<2x8xi8>, %s1: tensor<2x2xf32>, %z1: tensor<2x2xi32>)
               -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    %a = "quant_ext.dequantize_per_channel"(%w0, %s0, %z0) <{axis = 0 : i64}> : (tensor<2x8xi8>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x8xf32>
    %b = "quant_ext.dequantize_per_group"(%w1, %s1, %z1) <{group_size = 4 : i64, axis = 1 : i64}> : (tensor<2x8xi8>, tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2x8xf32>
    return %a, %b : tensor<2x8xf32>, tensor<2x8xf32>
  }
}""")
    assert lower_quant_ext(mod) == 2
    mod.verify()
    assert "quant_ext." not in str(mod)


@pytest.mark.parametrize(
    "kind,attrs,scale_type",
    [
        ("per_tensor", "", "tensor<f32>"),
        ("per_channel", "axis = 0 : i64", "tensor<2xf32>"),
    ],
)
def test_dequant_f32_scale_to_bf16_result_verifies(kind, attrs, scale_type):
    # A real BF16 model carries f32 quantization scales. Its output type must not
    # turn an f32 scale into the rhs of a BF16 arith.mulf.
    mod = _mod(
        kind,
        attrs,
        "tensor<2x8xi8>",
        scale_type,
        "tensor<i16>" if kind == "per_tensor" else "tensor<2xi16>",
        "tensor<2x8xbf16>",
    )
    assert lower_quant_ext(mod) == 1
    mod.verify()
    body = next(op for op in mod.walk() if op.name == "linalg.generic").body.blocks[0]
    assert [op.name for op in body.ops] == [
        "arith.sitofp",
        "arith.sitofp",
        "arith.subf",
        "arith.extf",
        "arith.mulf",
        "arith.truncf",
        "linalg.yield",
    ]
    mul = next(op for op in body.ops if op.name == "arith.mulf")
    assert all(str(value.type) == "f64" for value in mul.operands)


def test_per_tensor_quantize_lowers_to_exact_standard_qdq_expression():
    mod = parse_mlir_text("""module {
  func.func @q(%x: tensor<2x8xf32>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x8xi8> {
    %r = "quant_ext.quantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<2x8xf32>, tensor<f32>, tensor<i64>) -> tensor<2x8xi8>
    return %r : tensor<2x8xi8>
  }
}""")
    assert lower_quant_ext(mod) == 1
    mod.verify()
    assert "quant_ext." not in str(mod)
    generics = [o for o in mod.walk() if o.name == "linalg.generic"]
    assert len(generics) == 2
    reciprocal, gen = generics
    assert [o.name for o in reciprocal.body.blocks[0].ops] == ["arith.constant", "arith.divf", "linalg.yield"]
    assert [o.name for o in gen.body.blocks[0].ops] == [
        "arith.mulf",
        "math.roundeven",
        "arith.sitofp",
        "arith.addf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.fptosi",
        "linalg.yield",
    ]
    assert str(gen.indexing_maps.data[1]) == "affine_map<(d0, d1) -> ()>"
    assert str(gen.indexing_maps.data[2]) == "affine_map<(d0, d1) -> ()>"


def test_per_tensor_quantize_promotes_bf16_activation_to_f32():
    # PT2E quantize_per_tensor first converts BF16 activations to f32. The
    # reciprocal scale is already f32; multiplying it by BF16 is invalid MLIR.
    mod = parse_mlir_text("""module {
  func.func @q(%x: tensor<2x8xbf16>, %s: tensor<f32>, %z: tensor<i64>) -> tensor<2x8xi8> {
    %r = "quant_ext.quantize_per_tensor"(%x, %s, %z)
      <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<2x8xbf16>, tensor<f32>, tensor<i64>) -> tensor<2x8xi8>
    return %r : tensor<2x8xi8>
  }
}""")
    assert lower_quant_ext(mod) == 1
    mod.verify()
    body = [op for op in mod.walk() if op.name == "linalg.generic"][-1].body.blocks[0]
    names = [op.name for op in body.ops]
    assert names[:3] == ["arith.extf", "arith.mulf", "math.roundeven"]
    mul = next(op for op in body.ops if op.name == "arith.mulf")
    assert all(str(value.type) == "f32" for value in mul.operands)


def test_dead_qdq_cone_is_pruned_without_touching_unknown_side_effects():
    mod = parse_mlir_text("""module {
  func.func @f(%x: tensor<2x8xf32>, %w: tensor<2x8xi8>, %s: tensor<f32>, %z: tensor<i32>) -> tensor<2x8xf32> {
    %dead = "quant_ext.dequantize_per_tensor"(%w, %s, %z) : (tensor<2x8xi8>, tensor<f32>, tensor<i32>) -> tensor<2x8xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.splat %zero : tensor<2x8xf32>
    return %out : tensor<2x8xf32>
  }
}""")
    # The default reader leaves quant_ext unregistered. Its spelling cannot
    # supply an effect contract to DCE; the existing explicit lowering first
    # gives this value-only expression typed scalar/tensor semantics.
    assert prune_dead_pure_tensor_ops(mod) == 0
    assert lower_quant_ext(mod) == 1
    assert prune_dead_pure_tensor_ops(mod) == 2
    mod.verify()
    assert "quant_ext." not in str(mod)
    assert "tensor.splat" in str(mod)


def test_prune_preserves_constants_captured_by_a_live_linalg_region():
    mod = parse_mlir_text("""module {
  func.func @f(%x: tensor<2xf32>) -> tensor<2xf32> {
    %c = arith.constant 3.0 : f32
    %e = tensor.empty() : tensor<2xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<2xf32>) outs(%e : tensor<2xf32>) {
    ^bb0(%v: f32, %o: f32):
      %p = arith.mulf %v, %c : f32
      linalg.yield %p : f32
    } -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
}""")
    assert prune_dead_pure_tensor_ops(mod) == 0
    mod.verify()
    assert "arith.constant 3.000000e+00" in str(mod)


def test_the_captured_per_channel_spelling_lowers():
    """The op as the frontend actually emits it (provenance attributes and all), not a hand spelling."""
    from merlin.common.paths import merlin_dir

    text = (merlin_dir() / "tests" / "data" / "quant_ext" / "captured_dequantize_per_channel.mlir").read_text()
    mod = parse_mlir_text(text)
    assert lower_quant_ext(mod) == 1
    mod.verify()
    assert "quant_ext." not in str(mod)
    gen = next(o for o in mod.walk() if o.name == "linalg.generic")
    assert gen.attributes["prov.transforms"].data == "dequant_per_channel"
    assert str(gen.indexing_maps.data[1]) == "affine_map<(d0, d1) -> (d1)>"


def test_half_dequant_refuses_unproved_wide_zero_point():
    mod = _mod("per_tensor", "", "tensor<2xi8>", "tensor<f32>", "tensor<i64>", "tensor<2xf16>")
    with pytest.raises(ValueError, match="exact-product proof"):
        lower_quant_ext(mod)


@pytest.mark.parametrize("output", ["f16", "bf16"])
def test_half_dequant_literal_byte_zero_point_has_exact_product_proof(output):
    mod = parse_mlir_text(f"""module {{
      func.func @f(%w: tensor<2xi8>, %s: tensor<f32>) -> tensor<2x{output}> {{
        %z = arith.constant dense<127> : tensor<i64>
        %r = "quant_ext.dequantize_per_tensor"(%w, %s, %z) :
          (tensor<2xi8>, tensor<f32>, tensor<i64>) -> tensor<2x{output}>
        return %r : tensor<2x{output}>
      }}
    }}""")
    assert lower_quant_ext(mod) == 1
    mod.verify()
    mul = next(op for op in mod.walk() if op.name == "arith.mulf")
    assert str(mul.result.type) == "f64"
    assert sum(op.name == "arith.truncf" for op in mod.walk()) == 1


def test_f32_product_rounding_can_change_a_half_tie():
    import struct
    from fractions import Fraction

    # q=3 and this canonical f32 scale straddle a half midpoint: f32
    # multiplication rounds to the midpoint before the old second rounding.
    scale = struct.unpack("<f", struct.pack("<f", 0.3338215947151184))[0]
    exact = Fraction.from_float(scale) * 3
    midpoint = Fraction(1) + Fraction(3, 2048)
    assert exact < midpoint
    intermediate = struct.unpack("<f", struct.pack("<f", float(exact)))[0]
    assert Fraction.from_float(intermediate) == midpoint
    assert struct.unpack("<H", struct.pack("<e", float(exact)))[0] == 0x3C01
    assert struct.unpack("<H", struct.pack("<e", intermediate))[0] == 0x3C02


def test_unsigned_dequantization_reads_byte_codes_without_sign_extension():
    module = _mod(
        "per_tensor", 'input_dtype = "uint8"', "tensor<256xi8>", "tensor<f32>", "tensor<i16>", "tensor<256xf16>"
    )
    lower_quant_ext(module)
    module.verify()
    body = next(op for op in module.walk() if op.name == "linalg.generic").body.block
    assert list(body.ops)[0].name == "arith.uitofp"
    assert list(body.ops)[1].name == "arith.sitofp"
    assert all(str(v.type) == "f64" for op in body.ops if op.name == "arith.mulf" for v in op.operands)


def test_unsigned_per_channel_codes_promote_by_zero_extension_before_subtracting():
    module = _mod(
        "per_channel",
        'axis = 0 : i64, input_dtype = "uint8"',
        "tensor<2x8xi8>",
        "tensor<2xf32>",
        "tensor<2xi32>",
        "tensor<2x8xf32>",
    )
    lower_quant_ext(module)
    module.verify()
    body = next(op for op in module.walk() if op.name == "linalg.generic").body.block
    assert [op.name for op in body.ops] == ["arith.extui", "arith.subi", "arith.sitofp", "arith.mulf", "linalg.yield"]

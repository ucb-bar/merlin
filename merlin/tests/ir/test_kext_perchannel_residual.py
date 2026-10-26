"""Source-axis binding and exact residual coefficients for K-extension readout."""

from fractions import Fraction

import numpy as np
import pytest
from test_composition_fusion import _module
from test_output_fusion import _path

from merlin.common.mlir_query import parse
from merlin.targetgen import model_coverage
from merlin.targetgen.exact_arithmetic import round_binary
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF

WEIGHT_SCALES = (0.25,) * 4 + (0.5,) * 4
RESIDUAL_SCALES = (0.125,) * 4 + (0.25,) * 4


def source(*, scales=RESIDUAL_SCALES, weights=WEIGHT_SCALES, zeros=(0,) * 8, axis=1):
    text = _module(residual=True, weight_scale=weights, bias=(0,) * 8, relu=False)
    start, end = text.index("    %rs ="), text.index("    %e_sum =")
    scale_values = ", ".join(repr(float(np.float32(v))) for v in scales)
    zero_values = ", ".join(str(v) for v in zeros)
    return (
        text[:start]
        + f"""    %rs = arith.constant dense<[{scale_values}]> : tensor<{len(scales)}xf32>
    %rz = arith.constant dense<[{zero_values}]> : tensor<{len(zeros)}xi32>
    %rd = "quant_ext.dequantize_per_channel"(%rq, %rs, %rz) <{{axis = {axis} : i64}}>
        : (tensor<4x8xi8>, tensor<{len(scales)}xf32>, tensor<{len(zeros)}xi32>) -> tensor<4x8xf32>
"""
        + text[end:]
    )


def chain_for(text):
    module = parse(text)
    module.verify()
    root = next(op for op in module.walk() if CG.classify(op) and CG.classify(op).kind == CG.CONTRACTION)
    return module, OF.source_chain(root, OF.Values())


def decide(text, *, extension=True, strategy="f32_product_rne", path=None):
    module, chain = chain_for(text)
    path = _path() if path is None else path
    path["exits"]["accumulator"] = False
    return OF.decide(
        chain,
        path,
        OF.Values(),
        model_coverage._contraction_extents(module),
        readout_strategy=strategy,
        integer_residual_extension=extension,
    )


def residual(chain):
    return next(s for s in chain.stages if s.kind == "residual")


@pytest.mark.parametrize("axis", [1, -1])
def test_per_channel_residual_is_accepted_and_proved(axis):
    _, chain = chain_for(source(axis=axis))
    stage = residual(chain)
    table, scales = OF._residual_values(chain, stage, OF.Values(), np.float32)
    assert OF._residual_coefficients(chain, stage, scales) == {Fraction(2)}
    assert set(table) == set(range(-128, 128))
    for code, value in table.items():
        expected = np.float32(code) * np.asarray(RESIDUAL_SCALES, dtype=np.float32)
        np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
    result = decide(source(axis=axis))
    assert result.cut == "requant", result.refusals
    assert result.proof["requires_integer_residual_extension"]
    assert result.proof["definitional"] == "f32_product_rne"
    assert result.proof["accumulator_bound"] == [-2304, 2302]
    assert result.stages == [s.kind for s in chain.stages]


def test_per_channel_table_keeps_source_f32_multiply_rounding():
    _, chain = chain_for(source(scales=(0.3,) * 4 + (0.7,) * 4))
    table, _ = OF._residual_values(chain, residual(chain), OF.Values(), np.float32)
    scales = np.array((0.3,) * 4 + (0.7,) * 4, dtype=np.float32)
    for r in range(-128, 128):
        exact = [round_binary(Fraction(r) * Fraction(float(scale))) for scale in scales]
        np.testing.assert_array_equal(table[r].view(np.uint32), np.asarray(exact, dtype=np.float32).view(np.uint32))
    # The independent rational boundary rounds each product before addition.
    assert table[127].dtype == np.float32


@pytest.mark.parametrize(
    "axis,scales,zeros",
    [
        (0, (0.125,) * 4, (0,) * 4),
        (0, RESIDUAL_SCALES, (0,) * 8),
        (2, RESIDUAL_SCALES, (0,) * 8),
        (-3, RESIDUAL_SCALES, (0,) * 8),
    ],
)
def test_wrong_axis_scales_are_refused(axis, scales, zeros):
    result = decide(source(axis=axis, scales=scales, zeros=zeros))
    assert result.cut == "bare"
    assert any("axis" in r["reason"] or "output channels" in r["reason"] for r in result.refusals)


@pytest.mark.parametrize(
    "scales,zeros",
    [
        ((0.125,), (0,) * 8),
        (RESIDUAL_SCALES, (0,)),
        (RESIDUAL_SCALES, (0,) * 7 + (1,)),
        ((0.125,) * 7 + (0.0,), (0,) * 8),
        ((0.125,) * 7 + (-0.125,), (0,) * 8),
    ],
)
def test_bad_vectors_and_nonzero_channel_zero_points_are_refused(scales, zeros):
    assert decide(source(scales=scales, zeros=zeros)).cut == "bare"


@pytest.mark.parametrize(
    "weights,scales",
    [
        (WEIGHT_SCALES, (0.03125,) * 4 + (0.0625,) * 4),  # 1/2
        (WEIGHT_SCALES, (-0.03125,) * 4 + (-0.0625,) * 4),
        ((0.5,) * 8, (float(np.float32(0.3)),) * 8),
        (WEIGHT_SCALES, (0.125,) * 8),  # integral, but different coefficients
        (WEIGHT_SCALES, RESIDUAL_SCALES[:-1] + (float(np.nextafter(np.float32(0.25), np.float32(1))),)),
    ],
)
def test_noninteger_or_different_residual_coefficients_still_refuse(weights, scales):
    result = decide(source(weights=weights, scales=scales))
    assert result.cut != "requant", result.proof
    assert any("exact integer" in r["reason"] or "positive and finite" in r["reason"] for r in result.refusals)


def test_per_channel_ordinary_residual_uses_each_matching_accumulator_unit():
    text = source(scales=(0.0625,) * 4 + (0.125,) * 4)
    text = text.replace("quant_min = -128", "quant_min = -8").replace("quant_max = 127", "quant_max = 7")
    path = _path()
    next(s for s in path["readout"] if s["kind"] == "requant")["clamp"] = [-8, 7]
    result = decide(text, extension=False, strategy=None, path=path)
    assert result.cut == "requant", result.refusals
    assert result.proof["offsets"] == [0] * 8


def test_per_tensor_readout_and_residual_table_stay_unchanged():
    text = _module(residual=True, weight_scale=(0.25,), bias=(0,) * 8, relu=False)
    _, chain = chain_for(text)
    table, scale = OF._residual_values(chain, residual(chain), OF.Values(), np.float32)
    assert scale == 0.125 and type(scale) is float
    assert all(np.isscalar(v) and v == np.float32(r * 0.125) for r, v in table.items())
    result = decide(text)
    assert result.cut == "requant", result.refusals
    assert result.proof["requires_integer_residual_extension"]


def test_residual_layout_cannot_transpose_the_channel_in_the_add():
    text = source().replace("tensor<4x8", "tensor<8x8")
    # Square shapes do not make a row-wise access a channel broadcast.
    start = text.index("%sum =")
    text = text[:start] + text[start:].replace(
        "affine_map<(d0, d1) -> (d0, d1)>", "affine_map<(d0, d1) -> (d1, d0)>", 1
    )
    assert decide(text).cut == "bare"


def test_residual_axis_follows_matching_transposes():
    text = source()
    start = text.index("    %e_sum =")
    transpose = """    %ae = tensor.empty() : tensor<8x4xf32>
    %at = linalg.transpose ins(%ba : tensor<4x8xf32>) outs(%ae : tensor<8x4xf32>) permutation = [1, 0]
    %re = tensor.empty() : tensor<8x4xf32>
    %rt = linalg.transpose ins(%rd : tensor<4x8xf32>) outs(%re : tensor<8x4xf32>) permutation = [1, 0]
"""
    text = text[:start] + transpose + text[start:].replace("tensor<4x8", "tensor<8x4").replace("%ba, %rd", "%at, %rt")
    text = text.replace("-> tensor<4x8xi8> {", "-> tensor<8x4xi8> {")
    assert decide(text).cut == "requant"


@pytest.mark.parametrize("mutation", ["unsigned", "scale_dtype", "zero_dtype", "result_dtype"])
def test_unmodeled_residual_formats_refuse(mutation):
    text = source()
    if mutation == "unsigned":
        text = text.replace("axis = 1 : i64", 'axis = 1 : i64, input_dtype = "uint8"')
    elif mutation == "scale_dtype":
        start, end = text.index("    %rs ="), text.index("    %e_sum =")
        text = text[:start] + text[start:end].replace("tensor<8xf32>", "tensor<8xf64>") + text[end:]
    elif mutation == "zero_dtype":
        text = text.replace("tensor<8xi32>", "tensor<8xf32>").replace("dense<[0,", "dense<[0.0,")
        # Use valid floating literals for all zero points.
        text = text.replace("0, 0, 0, 0, 0, 0, 0]", "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
    else:
        # Rather than inventing mixed-precision arithmetic, inspect the dequant
        # theorem directly; the surrounding f32 chain remains the selected model.
        _, chain = chain_for(source())
        op = residual(chain).detail["operand"].owner
        from xdsl.dialects.builtin import TensorType, f64
        from xdsl.rewriter import Rewriter

        replacement = Rewriter.replace_value_with_new_type(op.results[0], TensorType(f64, [4, 8]))
        residual(chain).detail["operand"] = replacement
        with pytest.raises(OF._Refused, match="signed i8 to f32"):
            OF._residual_values(chain, residual(chain), OF.Values(), np.float32)
        return
    assert decide(text).cut == "bare"


def test_negative_integer_per_channel_coefficient_is_proved():
    text = source(scales=(-0.125,) * 4 + (-0.25,) * 4)
    _, chain = chain_for(text)
    stage = residual(chain)
    _, scales = OF._residual_values(chain, stage, OF.Values(), np.float32)
    assert OF._residual_coefficients(chain, stage, scales) == {Fraction(-2)}
    result = decide(text)
    assert result.cut == "requant", result.refusals
    assert result.proof["accumulator_bound"] == [-2302, 2304]


@pytest.mark.parametrize("scale", [-43307.0, 43307.0, float(np.nextafter(np.float32(43307.0), np.float32(0)))])
def test_original_f32_coefficient_refusals_are_exact(scale):
    # The retained noninteger smoke uses source gain f32(1/6). Nearby f32
    # residual scales must never be rounded back to coefficient 259842.
    text = source(weights=(float(np.float32(1 / 6)),) * 8, scales=(scale,) * 8)
    text = text.replace("%sin = arith.constant 2.500000e-01", "%sin = arith.constant 1.000000e+00")
    _, chain = chain_for(text)
    stage = residual(chain)
    _, scales = OF._residual_values(chain, stage, OF.Values(), np.float32)
    assert any(c.denominator != 1 for c in OF._residual_coefficients(chain, stage, scales))
    assert decide(text).cut == "bare"


def test_square_contraction_output_permutation_does_not_hide_a_wrong_axis():
    text = source().replace("tensor<4x8", "tensor<8x8")
    text = text.replace("affine_map<(m,n,k)->(m,n)>", "affine_map<(m,n,k)->(n,m)>", 1)
    result = decide(text)
    assert result.cut == "bare"
    assert any("output-channel layout" in r["reason"] for r in result.refusals)

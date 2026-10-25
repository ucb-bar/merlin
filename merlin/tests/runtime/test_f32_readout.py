"""One IEEE f32 readout, independent of simulator macros and exact host policy."""

from fractions import Fraction

import numpy as np
import pytest

from merlin.runtime.commandbuffer import apply_declared_acc_scale
from merlin.runtime.exact_readout import (
    f32_readout,
    f32_readout_scalar,
    form_multiplier,
    pred_pow2_vectors,
    prove_fp32_products,
    requantize_int32,
)
from merlin.runtime.tensor import Tensor


@pytest.mark.parametrize(
    "acc,multiplier,expected",
    [
        (664, form_multiplier(1, 166), 4),
        (-664, form_multiplier(1, 166), -4),
        (2**24 - 1, 2**-18, 64),
        (-(2**24 - 1), 2**-18, -64),
        (2**25 + 1, 2**-26, 0),
        (-(2**25 + 1), 2**-26, 0),
        (2**25 + 3, 2**-26, 1),
        (-(2**25 + 3), 2**-26, -1),
    ],
)
def test_conversion_product_and_pred_power_rounding(acc, multiplier, expected):
    assert f32_readout_scalar(acc, multiplier, "i8") == expected
    assert f32_readout([acc], multiplier, np.int8).tolist() == [expected]


def test_ties_even_negative_multipliers_and_saturation():
    values = [-1000, -257, -255, -253, -7, -5, -3, -1, 0, 1, 3, 5, 7, 253, 255, 257, 1000]
    expected = [-128, -128, -128, -126, -4, -2, -2, 0, 0, 0, 2, 2, 4, 126, 127, 127, 127]
    assert f32_readout(values, 0.5, "i8").tolist() == expected
    assert f32_readout([-x for x in values], -0.5, "i8").tolist() == expected
    assert f32_readout(values, 0.5, "u8").tolist() == [min(255, max(0, round(x / 2))) for x in values]
    assert f32_readout([-100, 100], -1, "i8", relu=True).tolist() == [0, -100]
    assert f32_readout([2**31 - 1, -(2**31)], 1, "i32").tolist() == [2**31 - 1, -(2**31)]
    assert f32_readout([2**31 - 1, -(2**31)], np.finfo(np.float32).max, "i8").tolist() == [127, -128]
    assert f32_readout([1, 3], 0.5, "i8", zero_point=1, qmin=1, qmax=2).tolist() == [1, 2]


def test_multiplier_rounds_exact_rational_once_including_halfway_and_subnormal():
    from merlin.runtime.commandbuffer import declared_acc_scale

    half = Fraction(1) + Fraction(1, 2**24)
    above = half + Fraction(1, 2**80)
    next_one = np.nextafter(np.float32(1), np.float32(np.inf))
    assert form_multiplier(half) == 1
    assert form_multiplier(above) == next_one
    assert form_multiplier(-above) == -next_one
    assert np.float32(float(above)) == 1  # An intermediate binary64 loses the deciding bit.
    assert form_multiplier(Fraction(1) + Fraction(3, 2**24)) == np.float32(1 + 2**-22)
    assert form_multiplier(1, 2**150) == 0
    assert form_multiplier(3, 2**150) == np.float32(2**-148)
    numerator, denominator = above.as_integer_ratio()
    assert form_multiplier(str(numerator), str(denominator)) == next_one
    assert declared_acc_scale({"acc_scale": above}, op="test") == next_one
    assert form_multiplier([1, 3], 2).tolist() == [0.5, 1.5]
    with pytest.raises(ValueError, match="denominators"):
        form_multiplier(1, 0)
    with pytest.raises(ValueError, match="overflows"):
        form_multiplier(2**128)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32])
def test_random_sweeps_against_numpy_float32(dtype):
    rng = np.random.default_rng(427)
    values = rng.integers(-(2**31), 2**31, size=(8192, 1))
    # Include arbitrary bit patterns, both signs, subnormals and near-overflow products.
    bits = rng.integers(0, 2**32, 128, dtype=np.uint32)
    scales = bits.view(np.float32)
    scales = np.concatenate((scales[np.isfinite(scales)], np.array([0, 0.5, 2**-26, 2**-18], np.float32)))
    with np.errstate(over="ignore"):
        product = values.astype(np.float32) * scales
    limits = np.iinfo(dtype)
    expected = np.clip(np.rint(product).astype(np.float64), limits.min, limits.max).astype(dtype)
    actual = f32_readout(values, scales, dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scale", [form_multiplier(1, 166), 2**-18, -(2**-18), 0.5, 2**-26])
def test_pred_generator_and_subsampled_exact_integer_domain(scale):
    vectors = pred_pow2_vectors(scale, -(2**31), 2**31 - 1, range(-2, 24))
    assert vectors.size and np.all(vectors[:-1] < vectors[1:])
    np.testing.assert_array_equal(vectors, pred_pow2_vectors(scale, -(2**31), 2**31 - 1, range(-2, 24)))
    if scale == form_multiplier(1, 166):
        assert 664 in vectors and -664 in vectors
    if abs(scale) == 2**-18:
        assert 2**24 - 1 in vectors and -(2**24 - 1) in vectors
    values = np.concatenate((np.arange(-(2**24), 2**24 + 1, 257), vectors))
    expected = np.clip(np.rint(values.astype(np.float32) * np.float32(scale)).astype(np.float64), -128, 127)
    np.testing.assert_array_equal(f32_readout(values, scale, "i8"), expected.astype(np.int8))


def test_pred_generator_finds_both_ends_of_all_reachable_preimages():
    values = np.arange(-2048, 2049)
    for scale in (form_multiplier(1, 166), form_multiplier(-1, 166), form_multiplier(3, 8)):
        vectors = set(pred_pow2_vectors(scale, values[0], values[-1], range(-1, 5)))
        products = values.astype(np.float32) * scale
        for exponent in range(-1, 5):
            power = np.float32(2**exponent)
            for magnitude in (np.nextafter(power, np.float32(0)), power - 0.5, power + 0.5):
                for sign in (-1, 1):
                    matches = values[products == sign * magnitude]
                    if matches.size:
                        assert matches[0] in vectors and matches[-1] in vectors


def test_exact_bridge_certifies_ieee_pred_power_result_and_rejects_rounding():
    proof = prove_fp32_products(Fraction(1, 2**18), -(2**24), 2**24)
    assert proof["readout_policy"] == "f32_product_rne"
    assert proof["equals_policy"] == "exact_rational_host"
    assert proof["integer_rounding"] == "half_to_even"
    pred = Fraction(2**24 - 1, 2**18)
    assert np.float32(pred) == np.nextafter(np.float32(64), np.float32(0))
    assert f32_readout_scalar(2**24 - 1, 2**-18) == round(pred) == 64
    with pytest.raises(ValueError, match="conversion"):
        prove_fp32_products(Fraction(1, 2**26), 0, 2**25 + 1)
    with pytest.raises(ValueError, match="scale"):
        prove_fp32_products(Fraction(1, 166), 0, 664)
    with pytest.raises(ValueError, match="mantissa"):
        prove_fp32_products(form_multiplier(1, 166), 0, 664)


def test_explicit_policy_wins_over_parameter_spelling_and_legacy_stays_exact():
    tensor = Tensor((1, 2), [2**25 + 1, -(2**25 + 1)], "i32")
    attrs = dict(scale_numerator="1", scale_denominator=str(2**26), quant_min=-128, quant_max=127)
    assert apply_declared_acc_scale(tensor, attrs, op="test").data == [1, -1]
    assert apply_declared_acc_scale(tensor, {**attrs, "readout_strategy": "exact_rational_host"}, op="test").data == [
        1,
        -1,
    ]
    assert apply_declared_acc_scale(tensor, {**attrs, "readout_strategy": "f32_product_rne"}, op="test").data == [0, 0]
    assert tensor.requant_acc_scale(2**-26).data == [0, 0]
    with pytest.raises(ValueError, match="strategy"):
        apply_declared_acc_scale(tensor, {**attrs, "readout_strategy": "unknown"}, op="test")


def test_v1_per_tensor_power_of_two_exact_host_bytes_are_unchanged():
    rng = np.random.default_rng(71)
    values = np.concatenate(([2**25 + 1, -(2**25 + 1), -(2**31), 2**31 - 1], rng.integers(-(2**31), 2**31, 4096)))
    tensor = Tensor(values.shape, values.tolist(), "i32")
    for exponent in range(-32, 2):
        ratio = Fraction(2) ** exponent
        attrs = dict(
            readout_strategy="exact_rational_host",
            scale_numerator=str(ratio.numerator),
            scale_denominator=str(ratio.denominator),
            quant_min=-128,
            quant_max=127,
        )
        expected = np.asarray([min(127, max(-128, round(int(x) * ratio))) for x in values], dtype=np.int8)
        result = apply_declared_acc_scale(tensor, attrs, op="test").to_i8()
        assert np.asarray(result.data, dtype=np.int8).tobytes() == expected.tobytes()
        np.testing.assert_array_equal(
            requantize_int32(values, scale_numerator=ratio.numerator, scale_denominator=ratio.denominator), expected
        )


def test_scalar_column_and_fusion_readouts_share_the_definition():
    from merlin.xdsl_dialects.lowering.output_fusion import _device

    values = np.array([[664, 2**24 - 1, 2**25 + 1], [-664, -(2**24 - 1), -(2**25 + 1)]])
    scales = [form_multiplier(1, 166), 2**-18, 2**-26]
    expected = f32_readout(values, scales, "i8")
    tensor = Tensor(values.shape, values.reshape(-1).tolist(), "i32")
    assert tensor.requant_acc_scale_by_column(scales).to_i8().data == expected.reshape(-1).tolist()
    path = [
        {"kind": "convert", "to": "f32"},
        {"kind": "scale", "dtype": "f32"},
        {"kind": "requant", "to": "i8", "clamp": [-128, 127]},
    ]
    np.testing.assert_array_equal(_device(path, np.asarray(scales), values, relu=False), expected)


def test_fusion_multiplier_formation_rounds_the_declared_ratio_once():
    from types import SimpleNamespace

    from merlin.xdsl_dialects.lowering.output_fusion import _device_scale

    rng = np.random.default_rng(61)
    factors = rng.uniform(0.01, 2, (3, 128)).astype(np.float32)
    divisor = rng.uniform(0.01, 2, 128).astype(np.float32)
    stages = [SimpleNamespace(kind="scale", numbers=f) for f in factors]
    stages.append(SimpleNamespace(kind="quantize", numbers=divisor))
    chain = SimpleNamespace(float_dtype="f32", stages=stages)
    actual, _ = _device_scale(chain, [{"kind": "scale", "dtype": "f32"}], 128)
    exact = [
        Fraction(float(a)) * Fraction(float(b)) * Fraction(float(c)) / Fraction(float(d))
        for a, b, c, d in zip(*factors, divisor, strict=True)
    ]
    expected = form_multiplier(exact)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    stepwise = ((factors[0] * factors[1]) * factors[2]) * (np.float32(1) / divisor)
    assert np.any(stepwise != expected)


def test_prepack_register_values_use_the_shared_multiplier_rule():
    import struct
    from types import SimpleNamespace

    from merlin.xdsl_dialects.lowering.group_prepack import _scalar_row

    group = SimpleNamespace(
        index=0,
        stages=[],
        operand_sum={
            "readout_factor": 1.5,
            "lhs_scale": 1.5,
            "rhs_scale": 0.25,
            "bound_lsb": 0,
        },
    )
    result = _scalar_row(group)["operand_sum"]
    assert result["load_multipliers"] == [1, float(form_multiplier(1, 6))]
    for value, bits in zip(result["load_multipliers"], result["load_multipliers_f32_bits"], strict=True):
        assert struct.pack("<f", value) == struct.pack("<I", bits)


@pytest.mark.parametrize("multiplier", [form_multiplier(1, 166), 2**-18, 2**-26, -0.5])
def test_emitted_scalar_readout_matches_numpy_on_regression_vectors(tmp_path, multiplier):
    from merlin.llvmlower.host_builder import PTR, FnBuilder
    from merlin.llvmlower.host_scalar_builder import FpBuilder
    from merlin.tests.ir.test_host_tensor_codegen import _compiled

    values = np.concatenate(
        (
            pred_pow2_vectors(multiplier, -(2**31), 2**31 - 1, range(-2, 8)),
            [664, -664, 2**24 - 1, 2**25 + 1, -3, -1, 1, 3],
        )
    ).astype(np.int32)
    builder = FnBuilder([PTR, PTR])
    fp = FpBuilder(builder)
    source, destination = builder.entry.args

    def element(index):
        value = builder.load_i64(source, index, "i32")
        result = fp.f32_readout(value, multiplier, "i8")
        builder.store_i64(result, destination, index, "i8")

    builder.for_range(len(values), element)
    kernel = _compiled(builder, tmp_path, f"readout_{np.float32(multiplier).view(np.uint32):x}")
    output = np.empty(values.shape, dtype=np.int8)
    kernel(values.ctypes.data, output.ctypes.data)
    np.testing.assert_array_equal(output, f32_readout(values, multiplier, "i8"))

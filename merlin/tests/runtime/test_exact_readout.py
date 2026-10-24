"""Exact host requantization covers all storage schemes and full int32 tiles."""

from fractions import Fraction

import numpy as np
import pytest

from merlin.runtime.exact_readout import prove_fp32_products, requantize_int32


@pytest.mark.parametrize("lo,hi,zp", [(-128, 127, 0), (-127, 127, 0), (-128, 127, -13), (0, 255, 123), (1, 254, 129)])
@pytest.mark.parametrize("n,d", [(1, 2), (3, 10), (-7, 16), (2**80 + 1, 2**90 - 1), (0, 1)])
def test_all_codes_and_int32_edges_against_independent_fraction(lo, hi, zp, n, d):
    rng = np.random.default_rng(53)
    codes = np.concatenate(
        (np.arange(-128, 256), [-(2**31), 2**31 - 1, -(2**24) - 1, 2**24 + 1], rng.integers(-(2**31), 2**31, 2048))
    )
    actual = requantize_int32(
        codes, scale_numerator=str(n), scale_denominator=str(d), zero_point=zp, quant_min=lo, quant_max=hi
    )
    expected = [min(hi, max(lo, round(Fraction(int(a) * n, d)) + zp)) for a in codes]
    assert actual.tolist() == expected


def test_channel_vectors_negative_ties_and_odd_zero_point():
    actual = requantize_int32(
        [[1, 3, 5], [-1, -3, -5]],
        scale_numerator=["1", "1", "1"],
        scale_denominator=["2", "2", "2"],
        zero_point=[1, 3, 5],
        axis=1,
    )
    assert actual.tolist() == [[1, 5, 7], [1, 1, 3]]
    assert requantize_int32(np.empty((0, 3), np.int32), scale_numerator=[1, 2, 3], scale_denominator=2).shape == (0, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale_denominator=0),
        dict(scale_denominator=-2),
        dict(scale_numerator=1.5),
        dict(scale_numerator=True),
        dict(scale_numerator=[]),
        dict(scale_numerator=[1, 2]),
        dict(rounding="half_up"),
        dict(zero_point=256),
    ],
)
def test_invalid_contract_refuses(kwargs):
    with pytest.raises(ValueError):
        requantize_int32([1, 2, 3], **(dict(scale_numerator=1, scale_denominator=2) | kwargs))


@pytest.mark.parametrize("values", [[2**31], [-(2**31) - 1], [1.5], [True], [float("nan")]])
def test_non_int32_tiles_refuse(values):
    with pytest.raises(ValueError):
        requantize_int32(values, scale_numerator=1, scale_denominator=2)


def test_code_equivalence_is_not_a_product_exactness_proof():
    scale = Fraction(float(np.float32(1 + 2**-23)))
    values = np.arange(-128, 128)
    exact = requantize_int32(values, scale_numerator=scale.numerator, scale_denominator=scale.denominator)
    eager = np.rint(values.astype(np.float32) * np.float32(1 + 2**-23))
    assert np.array_equal(exact, eager)  # This weaker property must NOT license the fp32 product.
    with pytest.raises(ValueError, match="product mantissa"):
        prove_fp32_products(scale, -128, 127)
    proof = prove_fp32_products(Fraction(1, 2), -(2**24), 2**24)
    assert proof["status"] == "proven" and proof["accumulator_max"] == 2**24
    with pytest.raises(ValueError, match="conversion"):
        prove_fp32_products(Fraction(1, 2), -(2**24) - 1, 2**24)


def test_runtime_reference_and_grader_share_exact_host_quantization():
    from merlin.runtime import integer_ops
    from merlin.runtime.tensor import Tensor

    values = [2**24 + 1, -(2**24) - 1]
    # An fp32 conversion erases the odd bit and picks the wrong half-even result; the exact readout does not.
    tensor = Tensor((2,), values, "i32")
    assert tensor.requant_exact(1.5).data == [25165826, -25165826]
    attrs = dict(
        weights=[1],
        constant=0,
        activation="none",
        scale_numerator="1",
        scale_denominator="2",
        zero_point=1,
        quant_min=-(2**31),
        quant_max=2**31 - 1,
        output_dtype="i16",
    )
    # Normal i16 bounds below: saturation occurs only after the exact boundary.
    attrs.update(quant_min=-32768, quant_max=32767)
    assert integer_ops.affine_map(tensor, None, attrs).data == [32767, -32768]
    ties = Tensor((2,), [1, 3], "i32")
    attrs.update(output_dtype="i8", quant_min=-128, quant_max=127)
    assert integer_ops.affine_map(ties, None, attrs).data == [1, 3]


def test_empty_tiles_still_validate_denominators():
    with pytest.raises(ValueError, match="denominators"):
        requantize_int32(np.empty((0, 3), dtype=np.int32), scale_numerator=1, scale_denominator=0)


def test_grader_uses_the_rational_pair_even_without_an_informational_scale():
    from merlin.runtime.tensor import Tensor
    from merlin.targetgen.capsule_golden import _apply_epilogue

    attrs = dict(
        epilogue=["acc_scale"],
        scale_numerator=["1", "3"],
        scale_denominator=["2", "2"],
        zero_point=1,
        quant_min=-128,
        quant_max=127,
    )
    assert _apply_epilogue(Tensor((1, 2), [1, 3], "i32"), attrs, {}).data == [1, 5]


def test_full_int32_readout_cannot_silently_ignore_a_zero_point():
    from merlin.runtime.integer_ops import IntegerOpError, affine_map
    from merlin.runtime.tensor import Tensor

    with pytest.raises(IntegerOpError, match="no readout"):
        affine_map(
            Tensor((1,), [1], "i32"),
            None,
            dict(
                weights=[1],
                constant=0,
                activation="none",
                scale_numerator="1",
                scale_denominator="1",
                zero_point=1,
                output_dtype="i32",
            ),
        )


def test_plain_rational_random_full_int32_and_adversarial_boundaries():
    import random
    from merlin.runtime.exact_readout import round_rational_codes

    rng = random.Random(7234)
    ratios = [Fraction(1, 1 << 149), Fraction(1 << 127), Fraction(1, 3), Fraction(17, 31), Fraction(0xFFFFFF, 1 << 24)]
    ratios += [Fraction(rng.randrange(1, 1 << 24), rng.randrange(1, 1 << 24)) for _ in range(60)]
    for scale in ratios:
        bias = Fraction(rng.randrange(-500, 500), rng.randrange(1, 40))
        zero = rng.randrange(255)
        points = {-(1 << 31), (1 << 31) - 1, 0}
        for code in range(-zero, 256-zero):
            boundary = int((Fraction(2*code+1, 2)-bias) / scale)
            points.update(v for v in (boundary-1, boundary, boundary+1) if -(1 << 31) <= v < 1 << 31)
        points.update(rng.randrange(-(1 << 31), 1 << 31) for _ in range(50))
        values = [x*scale+bias for x in points]
        expected = [min(255,max(0,round(v)+zero)) for v in values]
        assert round_rational_codes(values, zero, 0, 255).tolist() == expected


def test_plain_rational_exhaustive_small_domain():
    from merlin.runtime.exact_readout import round_rational_codes

    for scale in [Fraction(1,2), Fraction(3,8), Fraction(5,7), Fraction(1), Fraction(2), Fraction(17,31)]:
        for bias in [Fraction(0), Fraction(-1,2), Fraction(7,3)]:
            values = [x*scale+bias for x in range(-1024,1025)]
            assert round_rational_codes(values,123,1,255).tolist() == [min(255,max(1,round(v)+123)) for v in values]

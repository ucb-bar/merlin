"""Finite native conversion evidence, using integer-only IEEE round-to-nearest-even references."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify, program, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_typed_stage


def fields(width):
    if width == 32:
        return 23, 8, 127
    if width == 64:
        return 52, 11, 1023
    raise ValueError(f"Unqualified IEEE format width: {width}")


def round_shift(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def finite_bits(sign, significand, power, width):
    """Round (-1)^sign * significand * 2^power once; no host float conversion occurs."""
    fraction, exponent, bias = fields(width)
    prefix = sign << (width - 1)
    if significand == 0:
        return prefix
    top = significand.bit_length() - 1
    unbiased = top + power
    if unbiased < 1 - bias:
        return prefix | round_shift(significand, (1 - bias - fraction) - power)
    rounded = round_shift(significand, top - fraction)
    if rounded == 1 << (fraction + 1):
        rounded >>= 1
        unbiased += 1
    if unbiased >= bias + 1:
        return prefix | (((1 << exponent) - 1) << fraction)
    return prefix | ((unbiased + bias) << fraction) | (rounded - (1 << fraction))


def integer_bits(value, name, source_width, target_width):
    signed = value - (1 << source_width) if name == "sitofp" and value & (1 << (source_width - 1)) else value
    return finite_bits(int(signed < 0), abs(signed), 0, target_width)


def convert_bits(value, source_width, target_width):
    fraction, exponent, bias = fields(source_width)
    target_fraction, target_exponent, _ = fields(target_width)
    sign = value >> (source_width - 1)
    exp = (value >> fraction) & ((1 << exponent) - 1)
    tail = value & ((1 << fraction) - 1)
    if exp == (1 << exponent) - 1:
        # Only NaN class is compared; arithmetic conversion need not preserve its payload/sign.
        return (sign << (target_width - 1)) | (((1 << target_exponent) - 1) << target_fraction) | int(tail != 0)
    significand = tail | ((1 << fraction) if exp else 0)
    return finite_bits(sign, significand, (exp - bias if exp else 1 - bias) - fraction, target_width)


def is_nan(value, width):
    fraction, exponent, _ = fields(width)
    return ((value >> fraction) & ((1 << exponent) - 1)) == (1 << exponent) - 1 and value & ((1 << fraction) - 1) != 0


def integer_inputs(width):
    if width <= 8:
        return list(range(1 << width))
    candidates = [0, 1, (1 << width) - 1, (1 << (width - 1)) - 1, 1 << (width - 1)]
    for precision in [24, 53]:
        candidates += [(1 << precision) + offset for offset in [-3, -1, 0, 1, 2, 3, 5]]
        # Around a binary32 halfway point at an exponent larger than binary64's precision.
        # These distinguish a direct integer conversion from accidental double rounding.
        candidates += [(1 << 62) + (1 << 38) + offset for offset in [-513, -1, 0, 1, 513]]
    rng = random.Random(827163 + width)
    candidates += [rng.getrandbits(width) for _ in range(24)]
    return list(dict.fromkeys(x % (1 << width) for x in candidates))


def floating_inputs(width):
    fraction, exponent, bias = fields(width)
    sign = 1 << (width - 1)
    infinity = ((1 << exponent) - 1) << fraction
    one = bias << fraction
    values = [
        0,
        sign,
        1,
        sign | 1,
        (1 << fraction) - 1,
        1 << fraction,
        one,
        one - 1,
        one + 1,
        infinity,
        sign | infinity,
        infinity - 1,
        sign | (infinity - 1),
        infinity | 1,
        infinity | (1 << (fraction - 1)),
    ]
    if width == 64:
        # Binary32 halfway values near 1, underflow, and overflow encoded exactly in binary64.
        values += [(1023 << 52) | (1 << 28) | delta for delta in [0, 1, 3]]
        values += [(873 << 52), (874 << 52), (1150 << 52), (1151 << 52)]
    rng = random.Random(628173 + width)
    return values + [rng.getrandbits(width) for _ in range(25)]


def certified(tmp_path, tools, text):
    lean_tools = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, lean_tools, text)
    certify(stage, before, lean_tools, tmp_path)
    return stage, before, after


def compare_expected(tmp_path, tools, before, after, samples, expected, width):
    for label, path in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, tools, function="forward", arithmetic_nans=True)
        results = json.loads((work / "native-results.json").read_text())["results"]
        assert len(results) == len(expected)
        for observed, reference in zip(results, expected, strict=True):
            assert observed[-1] == reference[-1]
            assert len(observed) == len(reference)
            for bits, correct in zip(observed[:-1], reference[:-1], strict=True):
                assert is_nan(int(bits), width) if is_nan(int(correct), width) else int(bits) == int(correct)


@pytest.mark.parametrize("name", ["sitofp", "uitofp"])
@pytest.mark.parametrize("source_width", [1, 8, 13, 64])
@pytest.mark.parametrize("target_width", [32, 64])
def test_integer_to_float_has_single_rounding_native_reference(
    tmp_path, fidelity_tools, name, source_width, target_width
):
    _, before, after = certified(tmp_path, fidelity_tools, program(name, f"i{source_width}", f"f{target_width}"))
    values = integer_inputs(source_width)
    samples = [[[value], [keep]] for keep, value in enumerate(values, 827163)]
    expected = [
        [str(integer_bits(value, name, source_width, target_width)), str(keep)]
        for keep, value in enumerate(values, 827163)
    ]
    compare_expected(tmp_path, fidelity_tools, before, after, samples, expected, target_width)


@pytest.mark.parametrize("source_width,target_width", [(32, 64), (64, 32)])
def test_float_format_conversion_matches_exact_binary_reference(tmp_path, fidelity_tools, source_width, target_width):
    name = "extf" if source_width < target_width else "truncf"
    _, before, after = certified(tmp_path, fidelity_tools, program(name, f"f{source_width}", f"f{target_width}"))
    values = floating_inputs(source_width)
    samples = [[[value], [keep]] for keep, value in enumerate(values, 827163)]
    expected = [
        [str(convert_bits(value, source_width, target_width)), str(keep)] for keep, value in enumerate(values, 827163)
    ]
    compare_expected(tmp_path, fidelity_tools, before, after, samples, expected, target_width)


@pytest.mark.parametrize(
    "name,source,target",
    [("extf", "f32", "f64"), ("truncf", "f64", "f32"), ("sitofp", "i13", "f32"), ("uitofp", "i13", "f64")],
)
def test_actual_generic_conversions_have_defined_native_execution(tmp_path, fidelity_tools, name, source, target):
    width, output_width = int(source[1:]), int(target[1:])
    _, before, after = certified(
        tmp_path, fidelity_tools, program(name, source, target, generic=True, accumulate=False)
    )
    values = integer_inputs(width) if source.startswith("i") else floating_inputs(width)
    samples, expected = [], []
    for n, value in enumerate(values):
        data = [value, values[(n + 1) % len(values)], values[(n + 2) % len(values)]]
        convert = (
            (lambda x: integer_bits(x, name, width, output_width))
            if source.startswith("i")
            else (lambda x: convert_bits(x, width, output_width))
        )
        samples.append([data, [0, 0, 0], [827163 + n]])
        expected.append([*map(str, map(convert, data)), str(827163 + n)])
    compare_expected(tmp_path, fidelity_tools, before, after, samples, expected, output_width)


@pytest.mark.parametrize("target_width", [32, 64])
def test_wrong_signedness_remains_defined_but_fails_refinement_and_native_values(
    tmp_path, fidelity_tools, target_width
):
    stage, before, after = certified(tmp_path, fidelity_tools, program("sitofp", "i64", f"f{target_width}"))
    after.write_text(after.read_text().replace("arith.sitofp", "arith.uitofp", 1))
    lean_tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    verify(after, lean_tools)
    result = check_definedness(after, "forward", tmp_path / "wrong-defined", m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    value, keep = (1 << 64) - 1, 827163
    samples = [[[value], [keep]]]
    results = []
    for label, path in [("before", before), ("wrong", after)]:
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, fidelity_tools, function="forward")
        actual = json.loads((work / "native-results.json").read_text())["results"]
        name = "sitofp" if label == "before" else "uitofp"
        assert actual == [[str(integer_bits(value, name, 64, target_width)), str(keep)]]
        results.append(actual)
    assert results[0] != results[1]

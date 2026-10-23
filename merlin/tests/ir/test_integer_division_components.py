"""Full small dtype quotient domains, universal wide bounds and exception refusal."""

import pytest

from merlin.llvmlower.integer_division_components import cpu_floor_wrap, reciprocal


@pytest.mark.parametrize("divisor", [d for d in range(-128, 128) if d])
def test_reciprocal_product_correction_exhaustive_i8(divisor):
    multiplier, shift = reciprocal(8, divisor)
    for value in range(-128, 128):
        n = abs(value)
        q = n * multiplier >> shift
        q -= q * abs(divisor) > n
        assert q == n // abs(divisor)
        trunc = -q if (value < 0) != (divisor < 0) else q
        remainder = value - trunc * divisor
        floor = trunc - bool(remainder and ((value < 0) != (divisor < 0)))
        assert floor == value // divisor


@pytest.mark.parametrize("width", [16, 32, 64])
def test_universal_wide_interval_and_extremes(width):
    for divisor in (1, 2, 3, 7, (1 << (width - 1)) - 1, -(1 << (width - 1))):
        multiplier, shift = reciprocal(width, divisor)
        for value in (0, 1, (1 << (width - 1)) - 1, 1 << (width - 1)):
            q = value * multiplier >> shift
            q -= q * abs(divisor) > value
            assert q == value // abs(divisor)


def test_zero_divisor_cannot_be_proved():
    with pytest.raises(ValueError):
        reciprocal(64, 0)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_floor_overflow_requires_the_exact_cpu_exception_certificate(width):
    expected = b"".join((v % (1 << width)).to_bytes(width // 8, "little") for v in (-(1 << (width - 1)), 0, -1))
    row = dict(
        width=width,
        mode="floor",
        divisor=-1,
        observation=dict(status="value", dtype="torch.int" + str(width), bits=list(expected)),
    )
    proof = {"integer_checks": {"exceptions": [row]}}
    assert cpu_floor_wrap(proof, width)
    row["observation"]["bits"][0] ^= 1
    assert not cpu_floor_wrap(proof, width)
    assert not cpu_floor_wrap(proof, 1)

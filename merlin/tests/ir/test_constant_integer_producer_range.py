"""Constant contraction bounds use every weight and every ordered prefix."""

import ctypes
import itertools
import subprocess

import pytest

from merlin.llvmlower.integer_producer_range import ConstantIntegerSumProductsRange as Domain


@pytest.mark.parametrize("weights", [(2, -3, 1), (-4, 0, 2), (0,), ()])
def test_independent_exhaustive_prefix_oracle(weights):
    domain = Domain(weights, -2, 1, -3, 4)
    finals, prefixes = [], []
    for seed in range(-3, 5):
        for values in itertools.product(range(-2, 2), repeat=len(weights)):
            current = seed
            prefixes.append(current)
            for a, w in zip(values, weights, strict=True):
                current += a * w
                prefixes.append(current)
            finals.append(current)
    assert domain.interval() == (min(finals), max(finals))
    assert domain.prefix_interval() == (min(prefixes), max(prefixes))


def test_prefix_overflow_refuses_even_if_final_cancels():
    # A fixed positive activation cancels eventually, but the second prefix
    # overflows. Looking at just the final weight sum would admit incorrectly.
    domain = Domain((1 << 30, 1 << 30, -(1 << 30), -(1 << 30)), 1, 1)
    with pytest.raises(ValueError, match="prefix overflow"):
        domain.interval()


def test_product_overflow_refuses_even_if_seed_would_cancel():
    with pytest.raises(ValueError, match="product overflow"):
        Domain((1 << 30,), 2, 2, -(1 << 30), -(1 << 30)).interval()


def test_i8_negative_endpoint_and_nonzero_seed():
    assert Domain((-128, 127, 0), -128, 127).interval() == (-32512, 32513)
    domain = Domain((1, -1), 2, 3, -7, -5)
    assert domain.interval() == (-8, -4)
    assert domain.prefix_interval() == (-8, -2)


@pytest.mark.parametrize("domain", [
    Domain([1], -1, 1),
    Domain((True,), -1, 1),
    Domain((1.0,), -1, 1),
    Domain((1 << 31,), -1, 1),
    Domain((1,), True, 1),
    Domain((1,), 2, 1),
    Domain((1,), -1, 1, 2, 1),
])
def test_malformed_or_mutable_constants_refuse(domain):
    with pytest.raises(ValueError):
        domain.interval()


def test_exact_conversion_boundary_and_sufficient_refusal():
    assert Domain((1,), -(1 << 24), 1 << 24).require_exact_binary_conversion(significand_bits=24) == (-(1 << 24), 1 << 24)
    with pytest.raises(ValueError, match="inexact floating"):
        Domain((1,), 0, (1 << 24) + 1).require_exact_binary_conversion(significand_bits=24)
    # This representable singleton is intentionally outside the sufficient
    # contiguous-integer certificate. No necessary-exactness claim is made.
    with pytest.raises(ValueError, match="inexact floating"):
        Domain((1,), (1 << 24) + 2, (1 << 24) + 2).require_exact_binary_conversion(significand_bits=24)


@pytest.mark.parametrize("precision", [True, 1, 65, 24.0])
def test_precision_is_explicit_and_validated(precision):
    with pytest.raises(ValueError, match="significand"):
        Domain((1,), -1, 1).require_exact_binary_conversion(significand_bits=precision)


def test_actual_compiled_cast_of_all_small_certified_outputs_and_boundaries(tmp_path):
    source = tmp_path / "casts.c"
    source.write_text("""#include <stdint.h>
#include <fenv.h>
int same(int32_t x,int mode){int modes[]={FE_TONEAREST,FE_TOWARDZERO,FE_UPWARD,FE_DOWNWARD};
fesetround(modes[mode]);volatile int32_t a=x;volatile float b=(float)a;return (double)b==(double)a;}
""")
    so = tmp_path / "casts.so"
    subprocess.run(["cc", "-O2", "-frounding-math", "-shared", "-fPIC", str(source), "-lm", "-o", str(so)], check=True)
    library = ctypes.CDLL(str(so))
    library.same.argtypes = [ctypes.c_int32, ctypes.c_int]
    try:
        domain = Domain((2, -3, 1), -2, 1, -3, 4)
        low, high = domain.require_exact_binary_conversion(significand_bits=24)
        for mode in range(4):
            for value in [*range(low, high + 1), -(1 << 24), 1 << 24]:
                assert library.same(value, mode)
            assert not library.same((1 << 24) + 1, mode)
    finally:
        library.same(0, 0)

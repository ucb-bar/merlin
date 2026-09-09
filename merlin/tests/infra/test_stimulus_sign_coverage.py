"""Sign coverage in the shared stimulus, and C/Python agreement across the SIGN BOUNDARY.

Why this suite exists, concretely. The default stimulus range is four of the 256 i8 values and
every one of them is non-negative, so sign, saturation and domain-error paths are untested by
construction. That let a real defect ship in 655 compiler packages: the generated host lane's
`powf` was `exp(y * log(x))` -- valid only for x > 0, said so in its own docstring, and enforced
nothing. Its `log` is a generated polynomial that cannot produce NaN, so every negative base
returned a fixed finite 1.6516362661361307e+38 regardless of magnitude. `x ** 2` in RMSNorm
squared each negative activation to +1.65e38, the mean overflowed to +inf, and the quantizer's
clamp -- all of whose comparisons are false against the resulting NaN -- pinned every element to
-127. The whole-model symptom was eight identical argmaxes on a language model. One negative
stimulus value would have caught it.

The C emitters are tested by COMPILING AND RUNNING them against `fill`, not by comparing strings:
`{lo} + merlin_mix(...)%{span}u` is unsigned arithmetic, so a negative `lo` was promoted and the
sum became ~4.29e9. It survived a cast to an integer type by two's-complement truncation and was
silently wrong for `float`, which is exactly the shape of bug a string comparison cannot see.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.stimulus import (
    C_MIX_FN,
    DEFAULT_HI,
    DEFAULT_LO,
    SIGNED_HI,
    SIGNED_LO,
    c_fill_loop,
    c_fill_loop_2d,
    fill,
    fill_signed,
    sign_coverage,
)

SHAPES = [(4, 4), (8, 2048), (3, 5), (1, 64), (2, 3, 7)]


def test_the_default_range_is_non_negative_and_that_is_the_known_limitation() -> None:
    """Pinned as a FACT, not an aspiration: changing it would invalidate committed goldens."""
    assert (DEFAULT_LO, DEFAULT_HI) == (0, 3)
    for shape in SHAPES:
        values = fill("A0", shape)
        assert min(values) >= 0
        assert sign_coverage(values)["negative"] is False


def test_the_signed_range_actually_produces_both_signs() -> None:
    """`fill_signed` draws from a hash, so this must be checked rather than assumed."""
    assert (SIGNED_LO, SIGNED_HI) == (-3, 3)
    for shape in SHAPES:
        coverage = sign_coverage(fill_signed("A0", shape))
        assert coverage["negative"], f"{shape} produced no negative value"
        assert coverage["positive"], f"{shape} produced no positive value"


def test_signed_fill_stays_inside_its_declared_range() -> None:
    for shape in SHAPES:
        values = fill_signed("W", shape)
        assert min(values) >= SIGNED_LO
        assert max(values) <= SIGNED_HI


def test_sign_coverage_reports_each_class_independently() -> None:
    assert sign_coverage([-1, 0, 1]) == {"negative": True, "zero": True, "positive": True}
    assert sign_coverage([1, 2]) == {"negative": False, "zero": False, "positive": True}
    assert sign_coverage([-2, -1]) == {"negative": True, "zero": False, "positive": False}
    assert sign_coverage([]) == {"negative": False, "zero": False, "positive": False}


def test_fill_is_deterministic_in_the_tensor_name() -> None:
    assert fill_signed("A0", (4, 4)) == fill_signed("A0", (4, 4))
    assert fill_signed("A0", (4, 4)) != fill_signed("W", (4, 4))


def _run_c(body: str, count: int, ctype: str, fmt: str, decl: str | None = None) -> list[str]:
    """Compile and run a C program that fills a buffer, and return what it printed."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler available to check the emitted fill")
    declaration = decl or f"static {ctype} dst[{count}];"
    reader = "((const " + ctype + " *)dst)[i]"
    program = f"""
#include <stdio.h>
#include <stdint.h>
typedef signed char elem_t;
{C_MIX_FN}
int main(void) {{
  {declaration}
{body}
  for (int i = 0; i < {count}; ++i) printf("{fmt}\\n", {reader});
  return 0;
}}
"""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "fill.c"
        binary = Path(tmp) / "fill"
        source.write_text(program)
        build = subprocess.run([cc, "-O1", "-std=gnu99", str(source), "-o", str(binary)],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[-2000:]
        out = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return out.stdout.split()


@pytest.mark.parametrize("lo,hi", [(DEFAULT_LO, DEFAULT_HI), (SIGNED_LO, SIGNED_HI)])
@pytest.mark.parametrize("ctype,fmt,cast", [("int32_t", "%d", "int32_t"),
                                            ("float", "%.0f", "float"),
                                            ("elem_t", "%d", "elem_t")])
def test_the_emitted_c_fill_matches_python_for_both_ranges(lo, hi, ctype, fmt, cast) -> None:
    """The reason this module exists is byte-identical data; a negative `lo` must not break it."""
    rows, cols = 4, 6
    from merlin.common.stimulus import det_seed
    body = c_fill_loop("dst", str(rows), str(cols), str(det_seed("A0")),
                       cast=cast, lo=lo, hi=hi)
    printed = [int(float(v)) for v in _run_c(body, rows * cols, ctype, fmt)]
    assert printed == fill("A0", (rows, cols), lo=lo, hi=hi)


@pytest.mark.parametrize("lo,hi", [(DEFAULT_LO, DEFAULT_HI), (SIGNED_LO, SIGNED_HI)])
def test_the_emitted_2d_c_fill_also_matches_python(lo, hi) -> None:
    rows, cols = 3, 5
    from merlin.common.stimulus import det_seed
    body = c_fill_loop_2d("dst", str(rows), str(cols), str(det_seed("W")),
                          cast="float", lo=lo, hi=hi)
    printed = [int(float(v)) for v in _run_c(
        body, rows * cols, "float", "%.0f", decl=f"static float dst[{rows}][{cols}];")]
    assert printed == fill("W", (rows, cols), lo=lo, hi=hi)


def test_a_negative_lo_would_be_wrong_under_unsigned_arithmetic() -> None:
    """Guards the specific defect: the emitted expression must keep the addition SIGNED.

    Written as a property of the emitted text because the failure it prevents is invisible for
    integer casts (two's-complement truncation restores the value) and only shows up for float.
    """
    emitted = c_fill_loop("dst", "R", "C", "S", cast="float", lo=SIGNED_LO, hi=SIGNED_HI)
    assert "int32_t" in emitted, "the modulus must be cast to a signed type before adding lo"
    assert f"{SIGNED_LO} + merlin_mix" not in emitted, "unsigned promotion of a negative lo"

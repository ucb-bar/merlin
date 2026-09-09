"""The gate must catch a non-finite output even though the harness is built with -ffast-math.

`harness_build_recipe("gemmini").cflags` contains `-ffast-math`, which implies
`-ffinite-math-only`: the compiler may assume no NaN exists, so a `got == got` guard folds to
`true` and is DELETED. Measured 2026-09-09 on a small_llama whole-model ELF whose every output
element was 0x7fc00000 -- the gate reported `bad=0 nonfinite=0`. The tolerance criterion fails open
on NaN too (every comparison is false, so `diff` is NaN and `NaN > tol` never counts), so an
all-NaN output PASSED everything except the per-row top-1. These tests compile the RENDERED gate
with the real flags and feed it NaN.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap

import pytest

from merlin.runtime.backends import base as backends

CC = shutil.which("cc") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler")

FAST_MATH_FLAGS = ["-O2", "-ffast-math"]


def test_the_recipe_really_does_use_fast_math() -> None:
    """If this ever stops being true the guard below is belt-and-braces, not load-bearing."""
    cflags = list(backends.harness_build_recipe("gemmini").cflags)
    assert "-ffast-math" in cflags, cflags


def _probe(source_body: str, tmp_path) -> str:
    src = tmp_path / "probe.c"
    src.write_text(textwrap.dedent(f"""
        #include <stdio.h>
        static const float out[4] = {{0.0f/0.0f, 1.0f, 0.0f/0.0f, 2.0f}};
        int main(void) {{
          int nonfinite = 0;
          for (int i = 0; i < 4; ++i) {{
        {source_body}
          }}
          printf("nonfinite=%d\\n", nonfinite);
          return 0;
        }}
        """), encoding="utf-8")
    exe = tmp_path / "probe"
    build = subprocess.run([CC, *FAST_MATH_FLAGS, str(src), "-o", str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return run.stdout.strip()


def test_a_float_comparison_guard_is_deleted_by_fast_math(tmp_path) -> None:
    """The shape the gate used to emit: under -ffast-math it counts NOTHING."""
    got = _probe("""
        const double v = (double)out[i];
        if (!(v == v)) { ++nonfinite; continue; }
    """, tmp_path)

    assert got == "nonfinite=0", (
        "if this now reports 2 the toolchain stopped assuming finite math; the bit test is then "
        "redundant rather than wrong")


def test_the_bit_pattern_guard_survives_fast_math(tmp_path) -> None:
    """What the gate emits now: an integer test no float flag can optimise away."""
    got = _probe("""
        unsigned long long bits = 0;
        for (unsigned b = 0; b < sizeof(out[0]); ++b) {
          bits |= (unsigned long long)((const unsigned char *)&out[i])[b] << (8u * b);
        }
        const unsigned long long expmask =
            (sizeof(out[0]) == 4) ? 0x7f800000ULL : 0x7ff0000000000000ULL;
        if ((bits & expmask) == expmask) { ++nonfinite; continue; }
    """, tmp_path)

    assert got == "nonfinite=2", "the bit test must see both NaNs"


def test_the_rendered_gate_uses_the_bit_test_not_a_float_comparison() -> None:
    """Pin the emitted source, so a future edit cannot quietly reintroduce the deleted guard."""
    import inspect

    from merlin.targetgen import bundle_harness as BH

    source = inspect.getsource(BH)
    assert "merlin_expmask" in source, "the gate no longer tests the bit pattern"
    assert '"  if (!(got == got)) { ++merlin_nonfinite; continue; }"' not in source, (
        "the float-comparison NaN guard is back; -ffast-math deletes it")

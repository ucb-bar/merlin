"""With no declared preference, the fallback format must be the narrowest -- not the alphabetical one.

`workload_spec.precision_preference` is authorial: which formats a target's owner wants ranked, and in
what order, is an intent nobody can derive. Three of six targets declare none, and the fallback was
`sorted(admitted_dtypes)[0]` -- ALPHABETICAL.

Measured consequence on the microscaling target, whose entire purpose is mxfp4/mxfp6/mxfp8: its
admitted set is {bf16, i8, mxfp4, mxfp6, mxfp8} and the alphabetical answer is `bf16`, the WIDEST of
the five. The stated goal is to exercise the best quantization format the hardware supports rather
than a wide one, so the alphabet was silently choosing the opposite of the intent.

"Narrowest first" is a derivation, not an invention: storage width comes from `quant_formats` through
`capsule_dram.dtype_bits`, which already knows the sub-byte packed widths (mxfp4 -> 4, mxfp6 -> 6).
A declared preference still wins outright.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import corpus_synth as CS
from merlin.targetgen.capsule_dram import dtype_bits


def test_the_narrowest_admitted_format_wins_not_the_first_alphabetically():
    admitted = {"bf16", "i8", "mxfp4", "mxfp6", "mxfp8"}
    assert sorted(admitted)[0] == "bf16", "fixture must reproduce the alphabetical trap"
    assert CS.narrowest_admitted(admitted) == "mxfp4"
    assert dtype_bits("mxfp4") < dtype_bits("bf16")


def test_a_single_admitted_format_is_returned_unchanged():
    assert CS.narrowest_admitted({"i8"}) == "i8"


def test_nothing_admitted_yields_no_dtype_rather_than_raising():
    """The caller reports the cell as unexpressable; an exception here would abort the corpus."""
    assert CS.narrowest_admitted(set()) == ""


def test_ties_break_deterministically_so_regeneration_stays_byte_stable():
    """Two formats of equal width must not reorder between runs."""
    same_width = {d for d in ("fp8_e4m3", "fp8_e5m2", "i8") }
    widths = {d: dtype_bits(d) for d in same_width}
    if len(set(widths.values())) != 1:
        pytest.skip(f"fixture is not a tie: {widths}")
    first = CS.narrowest_admitted(same_width)
    assert first == min(same_width)
    assert all(CS.narrowest_admitted(same_width) == first for _ in range(3))


def test_a_token_of_unknown_width_never_outranks_a_known_one():
    """It sorts last rather than raising: unmeasurable is not narrow."""
    got = CS.narrowest_admitted({"not_a_real_dtype", "bf16"})
    assert got == "bf16"
    # ...and on its own it is still returned, so a corpus with only exotic tokens still generates.
    assert CS.narrowest_admitted({"not_a_real_dtype"}) == "not_a_real_dtype"


def test_the_widths_it_ranks_by_are_the_registry_s_own():
    """If these drift the ranking is wrong, so assert the packed sub-byte widths explicitly."""
    assert dtype_bits("mxfp4") == 4
    assert dtype_bits("mxfp6") == 6
    assert dtype_bits("i8") == 8
    assert dtype_bits("bf16") == 16
    assert dtype_bits("f32") == 32

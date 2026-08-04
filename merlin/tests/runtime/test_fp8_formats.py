"""fp8 is not one format: e4m3 and e5m2 decode the same byte to different floats — the module must
DIFFERENTIATE them (a grader that assumes one silently mis-values operands of the other).

Anchors, not a full table: e4m3 stays byte-identical to the proven whole-model decoder (no numeric drift),
e5m2 decodes its own layout incl. IEEE inf/NaN, and the two representable sets differ (distinct ranges).
"""
from __future__ import annotations

import numpy as np

from merlin.runtime.fp8_formats import canonical_fp8, fp8_to_f32, representable_values


def test_e4m3_matches_the_existing_whole_model_decoder():
    from merlin.runtime.dispatch_runtime import f8e4m3fn_to_f32
    allb = np.arange(256, dtype=np.uint8)
    a = np.nan_to_num(fp8_to_f32(allb, "fp8_e4m3"), nan=1e9)
    b = np.nan_to_num(f8e4m3fn_to_f32(allb), nan=1e9)
    assert np.array_equal(a, b)                       # no numeric drift for the e4m3 path


def test_e5m2_decodes_its_own_layout_including_inf_nan():
    dec = lambda u, f: float(fp8_to_f32(np.array([u], np.uint8), f)[0])
    assert dec(0x3C, "fp8_e5m2") == 1.0               # e5m2 1.0 (exp 15, man 0)
    assert dec(0x38, "fp8_e4m3") == 1.0               # e4m3 1.0 (exp 7, man 0) — same byte, DIFFERENT value:
    assert dec(0x3C, "fp8_e4m3") != 1.0               #   0x3C is not 1.0 in e4m3
    assert np.isinf(dec(0x7C, "fp8_e5m2"))            # e5m2 has inf; e4m3fn does not
    assert np.isnan(dec(0xFF, "fp8_e5m2"))


def test_representable_sets_differ_and_are_finite():
    r4, r5 = representable_values("fp8_e4m3"), representable_values("fp8_e5m2")
    assert set(r4) != set(r5)                         # formats are genuinely distinguished
    assert max(r4) == 448.0 and max(r5) == 57344.0    # e5m2 trades mantissa for range
    assert all(np.isfinite(v) for v in r4 + r5)       # finite-only (safe to encode back exactly)


def test_unknown_format_fails_closed():
    import pytest
    with pytest.raises(KeyError):
        canonical_fp8("bf16")                         # not an fp8 format -> raise, never silently assume

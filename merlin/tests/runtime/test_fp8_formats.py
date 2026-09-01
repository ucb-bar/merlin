"""fp8 is not one format: e4m3 and e5m2 decode the same byte to different floats — the module must
DIFFERENTIATE them (a grader that assumes one silently mis-values operands of the other).

Anchors, not a full table: e4m3 stays byte-identical to the proven whole-model decoder (no numeric drift),
e5m2 decodes its own layout incl. IEEE inf/NaN, and the two representable sets differ (distinct ranges).
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime.fp8_formats import (
    canonical_float,
    canonical_fp8,
    e8m0_decode,
    float_format_params,
    fp8_to_f32,
    representable_values,
)


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
    with pytest.raises(KeyError):
        canonical_float("garbage")                    # unknown float format -> raise, never assume a default


# --- MX microscaling formats (fp6 e3m2, fp4 e2m1) + the E8M0 block scale ---------------------------------
def test_mx_format_params_are_derived_bias():
    # bias == (1 << (exp_bits-1)) - 1 for every MX width (the derived IEEE value, no baked table).
    assert float_format_params("fp6_e3m2")[:3] == (3, 2, 3)
    assert float_format_params("fp4_e2m1")[:3] == (2, 1, 1)
    assert float_format_params("fp8_e4m3")[:3] == (4, 3, 7)


def test_mx_manifest_aliases_map_to_layout():
    # the mxfp* manifest tokens resolve to their float layout (same code<->value map)
    assert canonical_float("mxfp8") == "fp8_e4m3"
    assert canonical_float("mxfp6") == "fp6_e3m2"
    assert canonical_float("mxfp4") == "fp4_e2m1"


def test_fp4_e2m1_representable_set():
    r = representable_values("fp4_e2m1")
    assert max(r) == 6.0 and min(r) == -6.0            # e2m1 max normal = (1+1/2)*2^2 = 6
    assert 0.5 in r and 1.5 in r and 3.0 in r          # subnormal 0.5, normals 1.5 / 3.0
    assert all(v == v and abs(v) != float("inf") for v in r)   # MX is finite-only (no inf/NaN)
    assert len(r) == 15                                # 7 magnitudes * 2 signs + zero


def test_fp6_e3m2_representable_set_and_distinct_from_fp4():
    r6 = representable_values("fp6_e3m2")
    assert max(r6) == 28.0                             # e3m2 max normal = (1+3/4)*2^4 = 28
    assert set(r6) != set(representable_values("fp4_e2m1"))   # genuinely different formats
    assert all(v == v and abs(v) != float("inf") for v in r6)


def test_fp4_and_fp6_normals_present():
    r4 = representable_values("fp4_e2m1", finite_only=True)
    assert 1.0 in r4 and 2.0 in r4 and 4.0 in r4       # e2m1 normals at exp 1..3
    r6 = representable_values("fp6_e3m2")
    assert 0.25 in r6 and 1.0 in r6                     # e3m2 (1)*2^-2 subnormal-adjacent + 1.0


def test_e8m0_block_scale_is_power_of_two_with_nan():
    import math
    assert e8m0_decode(127) == 1.0                     # bias 127 -> 2^0
    assert e8m0_decode(128) == 2.0 and e8m0_decode(126) == 0.5
    assert e8m0_decode(120) == 2.0 ** -7
    assert math.isnan(e8m0_decode(0xFF))               # 0xFF is the E8M0 NaN code


# --- encode direction ------------------------------------------------------------------------------
# The decoder was the only direction that existed, so a capsule whose operands were recorded as decoded
# floats (every bf16/fp16/f32 golden) could not be turned back into device bytes and was preloaded with
# NOTHING. These pin the inverse: it must agree with the decoder code-for-code, and must refuse rather
# than invent bytes it cannot represent.

def test_encode_is_the_exact_inverse_of_decode_for_every_code():
    from merlin.runtime.fp8_formats import _decode, float_to_codes, storage_bits
    for fmt in ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp4_e2m1"):
        codes = np.arange(1 << storage_bits(fmt), dtype=np.uint32)
        vals = _decode(codes, fmt)
        finite = np.isfinite(vals)
        back = _decode(float_to_codes(vals[finite], fmt), fmt)
        assert np.array_equal(back, vals[finite]), f"{fmt}: decode(encode(v)) != v"


def test_out_of_range_saturates_instead_of_becoming_inf():
    """An operand that silently turned into inf would poison every comparison it took part in."""
    from merlin.runtime.fp8_formats import _decode, float_to_codes, normal_range
    for fmt in ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2"):
        _, biggest = normal_range(fmt)
        # beyond THIS format's range (which overflows to float32 inf for bf16, whose largest finite
        # value is already near float32's own -- saturation has to cope with that too)
        with np.errstate(over="ignore"):
            over = np.float32(biggest) * np.float32(2.0)
        got = _decode(float_to_codes([over, -over], fmt), fmt)
        assert np.all(np.isfinite(got)), f"{fmt}: saturation produced a non-finite value"
        assert got[0] == biggest and got[1] == -biggest, f"{fmt}: did not saturate to the largest finite"


def test_nan_and_sub_byte_are_refused_not_guessed():
    from merlin.runtime.fp8_formats import encode_bytes, float_to_codes
    with pytest.raises(ValueError):
        float_to_codes([float("nan")], "bf16")
    with pytest.raises(ValueError):          # 4-bit element: packing is the caller's layout decision
        encode_bytes([0.5], "fp4_e2m1")


def test_encoded_bytes_are_little_endian_at_the_declared_width():
    from merlin.runtime.fp8_formats import encode_bytes
    assert encode_bytes([1.0], "bf16") == b"\x80\x3f"          # bf16 1.0 == 0x3F80
    assert len(encode_bytes([0.0] * 8, "bf16")) == 16
    assert len(encode_bytes([0.0] * 8, "fp8_e4m3")) == 8

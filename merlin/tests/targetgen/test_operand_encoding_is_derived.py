"""Device operand bytes are derived from the FORMAT REGISTRY, not from the dtype's spelling.

The mesh path injects real activations/weights onto the device by encoding them to the bytes the
command buffer's declared dtype implies. That encoder used to be four name comparisons (i8/u8/i32/f32)
with a fallback that imported ``merlin.targetgen.rtl.fp8_codec.encode_bytes`` — a module path that does
not exist and a function that was never written. So the fallback raised ``ModuleNotFoundError`` into a
bare ``except`` and returned ``None`` for EVERY float format.

That mattered far beyond the encoder, because of what ``None`` means downstream: the caller reports it
as "no reachable oracle in this env", so an unimplemented encoder presented as a MISSING SIMULATOR. The
float-datapath mesh path looked unavailable rather than broken, on every target that uses it, and the
tests that would have caught it skipped instead of failing.

Deriving from the registry also fixes the safety direction: e4m3 and e5m2 are both 8-bit floats with the
same element width, and only their exponent/mantissa split distinguishes them. A name-matched encoder
that fell through to "the fp8 codec" would encode one as the other and produce plausible wrong numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common import quant_formats as qf
from merlin.targetgen.mesh_program_run import _encode_operand

VALUES = [[1.0, -2.0], [0.5, 3.25]]


def test_every_byte_aligned_registry_format_encodes():
    """The regression: a whole CLASS of formats silently produced no bytes at all."""
    encodable, refused = [], []
    for name, f in sorted(qf.registry().items()):
        bits = int(f.element_bits or 0)
        got = _encode_operand(VALUES, name)
        if bits and bits % 8 == 0 and f.kind in ("int_affine", "float_ieee", "fp_ocp"):
            (encodable if got is not None else refused).append(name)
    assert not refused, f"byte-aligned formats with no encoding: {refused}"
    assert len(encodable) >= 4, encodable


def test_a_sub_byte_format_fails_closed_rather_than_guessing_a_packing():
    """Packing 4- and 6-bit codes into bytes is a layout decision, not this function's to invent."""
    sub = [n for n, f in qf.registry().items() if (f.element_bits or 0) % 8]
    if not sub:
        pytest.skip("no sub-byte format in the registry")
    for name in sub:
        assert _encode_operand(VALUES, name) is None, f"{name} was encoded despite being sub-byte"


def test_an_alias_encodes_identically_to_its_canonical_name():
    """Same format, different spelling — the bytes cannot differ."""
    checked = 0
    for name, f in sorted(qf.registry().items()):
        if (f.element_bits or 0) % 8 or _encode_operand(VALUES, name) is None:
            continue
        for alias in f.aliases:
            assert _encode_operand(VALUES, alias) == _encode_operand(VALUES, name), \
                f"{alias} encodes differently from {name}"
            checked += 1
    assert checked, "no aliases exercised"


def test_two_eight_bit_float_formats_do_not_encode_alike():
    """The safety property. e4m3 and e5m2 share a width; only the exponent split separates them, and
    encoding one as the other is a miscompile that produces plausible numbers."""
    a, b = _encode_operand(VALUES, "fp8_e4m3"), _encode_operand(VALUES, "fp8_e5m2")
    if a is None or b is None:
        pytest.skip("one of the 8-bit float formats is absent from this registry")
    assert len(a) == len(b) == 4
    assert a != b, "e4m3 and e5m2 produced identical bytes — the encoder is name-matching, not deriving"


@pytest.mark.parametrize("dtype,expect", [
    ("f32", "0000803f000000c00000003f00005040"),
    ("bf16", "803f00c0003f5040"),
    ("fp16", "003c00c000388042"),
    ("int8", "01fe0003"),
])
def test_known_encodings_are_bit_exact(dtype, expect):
    """Pinned against hand-computed IEEE/two's-complement bit patterns, so a plausible-looking but wrong
    encoder (byte order, rounding mode, truncation instead of round-to-nearest-even) is caught."""
    got = _encode_operand(VALUES, dtype)
    assert got is not None and got.hex() == expect, (dtype, got.hex() if got else None)


def test_bf16_rounds_to_nearest_even_rather_than_truncating():
    """Truncation is the easy bf16 bug and is invisible on values that fit exactly."""
    # 1.00390625 = 0x3F808000 — exactly halfway between bf16 0x3F80 and 0x3F81, so RNE keeps 0x3F80.
    assert _encode_operand([[1.00390625]], "bf16").hex() == "803f"
    # 1.0078125 = 0x3F810000 is exact; 1.005859375 = 0x3F80C000 rounds UP to 0x3F81 under RNE but
    # DOWN to 0x3F80 under truncation.
    assert _encode_operand([[1.005859375]], "bf16").hex() == "813f"


def test_an_unknown_dtype_is_refused_not_guessed():
    assert _encode_operand(VALUES, "not_a_real_format") is None
    assert _encode_operand(VALUES, "") is None

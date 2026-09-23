"""The program oracle must decode a readback at the DECLARED element width, or say why it cannot.

``_decode_output`` carried its own two-branch dtype table (bf16, i32, else int8) while its sibling
``_out_nbytes`` sized the DRAM read window from a *different* ad-hoc rule (``"16" in dtype`` -> 2 B,
``"32" in dtype`` -> 4 B, else 1 B). The two disagreed for every dtype the decoder's table did not
name: an ``f32`` output was READ as 4 bytes per element and DECODED as 1, so ``arr.reshape(shape)``
raised a bare numpy ``ValueError`` -- ``cannot reshape array of size 1024 into shape (16,16)`` for a
16x16 f32 tensor -- which the ladder classified as ``tool_crash``.

That was measured, not hypothesized. ``AF0_rmsnorm_bf16_pt`` and ``AF8_rope_bf16_pt`` passed L3 on
2026-09-05 while their command buffers still declared a ``bf16`` ``Y0``; commit 34ef3294 correctly
bound the command buffer's dtypes to the interface, whose ``Y0`` is ``tensor<16x16xf32>``, and both
capsules immediately died at that reshape with a message whose integers the verdict redacts to ``#``
("cannot reshape array of size # into shape (#,#)") -- uninterpretable in an artifact.

So two properties are gated here:

1. **One derived width.** Both the read window and the value decode come from the shared format
   registry (:mod:`merlin.common.quant_formats`), so they cannot disagree again -- and every dtype the
   capsule corpus declares as an output round-trips, not just the two the old table named.
2. **Fail closed with a name.** A byte count that cannot be reconciled with the declared shape, or a
   dtype whose width/value cannot be derived at all, raises ``OracleUnavailable`` naming the tensor,
   its declared shape and dtype, the expected byte count and the observed one. A bare numpy traceback
   is not a diagnostic an agent (or a reader of the artifact) can act on.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.targetgen import program_oracle as PO
from merlin.targetgen.fp8_codec import ocp_encode


def test_extracted_decoder_preserves_class_function_and_legacy_patch_identity(monkeypatch):
    from merlin.targetgen import program_values as values

    assert PO.OracleUnavailable is values.OracleUnavailable
    assert PO._decode_output is values._decode_output
    assert PO._resolve_out_specs is values._resolve_out_specs
    assert PO._IEEE_NATIVE is values._IEEE_NATIVE
    monkeypatch.setattr(PO, "_element_bytes", lambda dtype: 2)
    monkeypatch.setattr(PO, "_decode_elements", lambda raw, dtype, width: np.array([width]))
    assert values._decode_output(b"\x00\x00", [1], "fixture", None).tolist() == [2]


def test_f32_output_decodes_at_its_declared_width():
    """The regression itself: a 16x16 f32 readback is 1024 bytes and 256 elements, not 1024."""
    want = (np.arange(256, dtype=np.float32) * 0.5 - 3.0).reshape(16, 16)
    spec = {"shape": [16, 16], "dtype": "f32"}

    assert PO._out_nbytes(spec) == want.nbytes == 1024
    got = PO._decode_output(want.tobytes(), spec["shape"], spec["dtype"], None, name="Y0")
    assert got.shape == (16, 16)
    assert np.array_equal(got, want)


def test_read_window_and_decode_agree_for_every_declared_output_dtype():
    """The window `_out_nbytes` opens is exactly the number of bytes the decoder consumes.

    The disagreement between the two rules WAS the defect, so it is the invariant, checked over the
    whole output-dtype vocabulary the capsule corpus declares (``bf16``/``f32``/``fp8_e4m3``/``i32``/
    ``i8``) plus the ``torch.*`` spellings the program goldens use.
    """
    shape = [4, 4]
    for dtype in (
        "bf16",
        "f32",
        "fp8_e4m3",
        "i32",
        "i8",
        "f16",
        "int8",
        "fp64",
        "torch.bfloat16",
        "torch.int32",
        "torch.float32",
    ):
        nbytes = PO._out_nbytes({"shape": shape, "dtype": dtype})
        decoded = PO._decode_output(b"\x00" * nbytes, shape, dtype, None, name="Y0")
        assert decoded.shape == tuple(shape), dtype
        assert decoded.size == 16, dtype


def test_the_window_is_the_registrys_width_not_a_guess_from_the_spelling():
    """The window must come from the FORMAT, not from digits in its name.

    The retired ``_out_nbytes`` rule was ``2 if "16" in dtype else (4 if "32" in dtype else 1)``. It is
    right for every dtype whose spelling happens to contain its own width and silently wrong for the
    rest -- and "wrong" here is a mis-sized DRAM read, not a mis-formatted one. ``fp64`` is the cheapest
    witness the shipped registry provides: 8 bytes per element, and a spelling that contains neither
    "16" nor "32", so the old rule opened a 1-byte window -- an EIGHTH of the result -- and every byte
    after the first element came from whatever else lives at that address.

    Asserting the width against the registry (rather than against a literal 8) keeps this test derived:
    it states that the two agree, not what either one happens to be.
    """
    from merlin.common import quant_formats as qf

    for dtype in sorted(qf.names()):
        try:
            width = PO._element_bytes(dtype)
        except PO.OracleUnavailable:
            continue  # sub-byte / block-scaled: refused, and gated separately
        assert width == qf.storage_bits(dtype) // 8, dtype
        assert PO._out_nbytes({"shape": [4, 4], "dtype": dtype}) == 16 * width, dtype

    # The witness, spelled out: the pre-fix rule and the derived width DISAGREE here, so a test that
    # passes under both is not gating this.
    assert PO._element_bytes("fp64") == 8
    assert PO._out_nbytes({"shape": [16, 16], "dtype": "fp64"}) == 16 * 16 * 8


def test_bf16_and_i32_decode_is_unchanged():
    """The two dtypes the old table did name must decode bit-for-bit as they did before."""
    bits = np.array([0x3F80, 0x4000, 0xBF80, 0x0000, 0x7F80, 0x4049], dtype="<u2")
    bf16 = PO._decode_output(bits.tobytes(), [3, 2], "bf16", None, name="Y0")
    assert np.array_equal(bf16.view(np.uint32).ravel(), bits.astype(np.uint32) << 16)
    assert bf16[0, 0] == 1.0 and bf16[0, 1] == 2.0 and bf16[1, 0] == -1.0

    words = np.arange(-3, 3, dtype="<i4")
    assert np.array_equal(PO._decode_output(words.tobytes(), [3, 2], "i32", None), words.reshape(3, 2))


def test_fp8_output_decodes_as_its_format_not_as_raw_bytes():
    """An 8-bit float is a FLOAT. Reading it as int8 keeps the reshape legal and the values wrong,
    which is the silent half of the same fallback."""
    values = [0.0, 1.0, -2.5, 9.0]
    raw = bytes(ocp_encode(v, 4, 3) for v in values)
    got = PO._decode_output(raw, [4], "fp8_e4m3", None, name="Y0")
    assert got.tolist() == values


def test_declared_physical_unstacking_still_applies():
    """The physical->logical row un-stacking the emitting backend declares survives the rewrite."""
    stacked = np.arange(8, dtype=np.float32).reshape(4, 2)
    got = PO._decode_output(stacked.tobytes(), [4, 2], "f32", {"unstack_row_halves": 2})
    assert got.tolist() == [[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0]]


def test_byte_count_mismatch_names_the_tensor_shape_and_counts():
    """The fail-closed half. Not a bare ``cannot reshape array of size # into shape (#,#)``."""
    with pytest.raises(PO.OracleUnavailable) as exc:
        PO._decode_output(b"\x00" * 1024, [16, 16], "bf16", None, name="Y0")
    msg = str(exc.value)
    for token in ("'Y0'", "1024", "[16, 16]", "bf16", "256", "512"):
        assert token in msg, f"{token!r} missing from the diagnostic: {msg}"


def test_a_dtype_whose_width_cannot_be_derived_fails_closed():
    """UNKNOWN is recorded and surfaced, never replaced by a one-byte default."""
    with pytest.raises(PO.OracleUnavailable, match="UNKNOWN"):
        PO._decode_output(b"\x00" * 4, [4], "not_a_registered_format", None, name="Y0")
    with pytest.raises(PO.OracleUnavailable, match="UNKNOWN"):
        PO._out_nbytes({"shape": [4], "dtype": "not_a_registered_format"})


def test_a_block_scaled_dtype_fails_closed_rather_than_inventing_values():
    """A block-scaled element's value is not determined by the bytes in the output window."""
    with pytest.raises(PO.OracleUnavailable, match="scale plane"):
        PO._decode_output(b"\x00" * 4, [4], "mxfp8", None, name="Y0")


def test_a_block_scaled_integer_dtype_fails_closed_for_the_same_reason_a_float_one_does():
    """The refusal is about the SCALE PLANE, not about the element being a float.

    ``mxint8``/``gguf_q8_0`` are int8 codes under a per-32-block scale. Reading them as int8 keeps the
    reshape legal and every value wrong by its block's scale -- the identical silent failure the fp8
    case above pins, with the element kind swapped, so it must fail the identical way. A per-channel
    scale is NOT this: it is a property of the whole tensor rather than a plane inside it, and the
    codes are what the golden compares -- so ``i8`` must keep decoding.
    """
    from merlin.common import quant_formats as qf

    block_scaled = [n for n in qf.names() if qf.get(n).granularity == "per_block"]
    assert block_scaled, "registry declares no per-block format; this test would be vacuous"
    named_the_scale = 0
    for name in block_scaled:
        # EVERY per-block format is refused. Some are refused earlier and for a different true reason
        # (a sub-byte element has no whole-byte stride at all), which is equally fail-closed; the ones
        # whose bytes ARE addressable must be refused specifically because the scales are elsewhere.
        with pytest.raises(PO.OracleUnavailable) as exc:
            PO._decode_output(b"\x00" * 4, [4], name, None, name="Y0")
        if qf.storage_bits(name) % 8 == 0:
            assert "scale" in str(exc.value), f"{name}: {exc.value}"
            named_the_scale += 1
    assert named_the_scale, "no whole-byte per-block format exercised the scale-plane refusal"

    # The control: a per-channel int8 output still decodes, because 19 shipped capsules declare one.
    assert PO._decode_output(b"\x01\x02\x03\x04", [4], "i8", None, name="Y0").tolist() == [1, 2, 3, 4]
    assert qf.get("int8").granularity == "per_channel"


def test_the_two_regressed_atlas_capsules_declare_an_f32_output():
    """Ties the unit above to the corpus: the interface these capsules are graded against declares an
    f32 result, so decoding f32 is not a hypothetical -- it is what their L3 needs.

    Read from the capsule corpus via ``repo_root()`` so the test survives a move.
    """
    from merlin.common.paths import repo_root

    root = repo_root() / "merlin" / "contract" / "capsules"
    found = 0
    for name in ("AF0_rmsnorm_bf16_pt", "AF8_rope_bf16_pt"):
        for iface in root.rglob(f"{name}/capsule.interface.mlir"):
            text = iface.read_text(encoding="utf-8")
            assert "xf32>" in text, f"{name}: interface declares no f32 tensor"
            found += 1
    if not found:
        pytest.skip("capsule corpus not present in this checkout")

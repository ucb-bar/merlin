"""Tests for the target-agnostic quantization-format registry (merlin.common.quant_formats)."""
from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.common.yaml import write_yaml


def test_registry_loads_and_every_entry_is_valid():
    reg = qf.registry()
    assert reg, "registry is non-empty"
    # Spot-check the precisions the model-export matrix needs.
    for expected in ("fp32", "fp16", "bf16", "int8", "fp6_e3m2", "fp4_e2m1", "mxfp6", "nvfp4"):
        assert expected in reg, expected


def test_float_formats_obey_sign_exp_mantissa_identity():
    """A self-describing float element is exactly sign + exponent + mantissa.

    The sign bit counts only when the element has one. E8M0 -- the MX block-scale type, whose OCP name
    ends FNU for "finite, no sign, unsigned" -- spends all eight bits on the exponent, and asserting an
    unconditional sign bit made it unregisterable.
    """
    for fmt in qf.registry().values():
        if fmt.is_float:
            sign_bits = 1 if fmt.signed else 0
            assert sign_bits + fmt.exp_bits + fmt.mant_bits == fmt.element_bits, fmt.name


def test_an_unsigned_float_element_spends_every_bit_on_the_exponent():
    """Pins the case the identity above was widened for, so a future tightening cannot silently
    re-exclude it: the registry must be able to express a mantissa-less, sign-less float."""
    e8m0 = qf.get("e8m0")
    assert (e8m0.signed, e8m0.exp_bits, e8m0.mant_bits, e8m0.element_bits) == (False, 8, 0, 8)
    assert qf.get("f8E8M0FNU").name == "e8m0", "the MLIR spelling resolves to the same format"


def test_request_spec_encodings():
    # The exact OCP encodings named in the request.
    assert (qf.get("fp16").exp_bits, qf.get("fp16").mant_bits) == (5, 10)
    assert (qf.get("bf16").exp_bits, qf.get("bf16").mant_bits) == (8, 7)
    assert (qf.get("fp6_e3m2").exp_bits, qf.get("fp6_e3m2").mant_bits) == (3, 2)
    assert (qf.get("fp4_e2m1").exp_bits, qf.get("fp4_e2m1").mant_bits) == (2, 1)


def test_alias_resolution():
    assert qf.get("e2m1").name == "fp4_e2m1"
    assert qf.get("f16").name == "fp16"
    assert qf.has("bfloat16")
    assert not qf.has("does_not_exist")


def test_sub_byte_and_block_scale_flags():
    assert qf.get("fp4_e2m1").is_sub_byte and qf.get("fp4_e2m1").pack_bits == 4
    assert qf.get("mxfp6").is_block_scaled and qf.get("mxfp6").scale.block == 32
    assert not qf.get("bf16").is_block_scaled
    assert qf.get("nvfp4").scale.kind == "nvfp4_block"


def test_source_cross_references():
    assert qf.from_ggml("Q6_K").name == "gguf_q6_k"
    assert qf.from_ggml("Q8_0").name == "gguf_q8_0"
    assert qf.from_torchao("int8_weight_only").name == "int8"
    assert qf.get("mxfp6").quant_ext_type == "mx_tensor"
    assert qf.get("nvfp4").quant_ext_type == "nvfp4_tensor"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        qf.get("totally_unknown_format")


def test_validate_entry_rejects_bad_encoding():
    # sign+exp+mant must equal element_bits for a float kind.
    with pytest.raises(ValueError):
        qf._validate_entry("bad_fp", {"kind": "fp_ocp", "element_bits": 4, "exp_bits": 3, "mant_bits": 2})
    # unknown kind.
    with pytest.raises(ValueError):
        qf._validate_entry("bad_kind", {"kind": "nonsense", "element_bits": 8})
    # block scale without a block size.
    with pytest.raises(ValueError):
        qf._validate_entry(
            "bad_scale", {"kind": "int_affine", "element_bits": 8, "scale": {"kind": "block_affine"}}
        )


def test_overlay_merges_and_overrides(tmp_path, monkeypatch):
    overlay = tmp_path / "extra.yaml"
    write_yaml(
        overlay,
        {
            "version": 1,
            "formats": {
                "fp3_e2m0": {
                    "kind": "fp_ocp",
                    "element_bits": 3,
                    "exp_bits": 2,
                    "mant_bits": 0,
                    "pack": {"bits": 3, "dim": -1},
                    "scale": {"kind": "none"},
                }
            },
        },
    )
    monkeypatch.setenv("MERLIN_QUANT_FORMATS", str(overlay))
    qf.registry.cache_clear()
    qf._alias_index.cache_clear()
    try:
        assert qf.has("fp3_e2m0")
        assert qf.get("fp3_e2m0").element_bits == 3
    finally:
        monkeypatch.delenv("MERLIN_QUANT_FORMATS", raising=False)
        qf.registry.cache_clear()
        qf._alias_index.cache_clear()

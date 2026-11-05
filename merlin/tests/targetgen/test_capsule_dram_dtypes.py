"""The DRAM layout must not RUNNER_CRASH on a dtype spelling synonym.

Regression for the radiance suite: capsules declared ``fp16``/``mxfp8`` where ``_DTYPE_BYTES`` only knew
``f16``/(no mx), so ``dtype_bytes`` raised ``KeyError: unknown dtype 'fp16'`` and the RUNNER crashed on
3/6 capsules — an ungradeable harness failure, not an agent error (and its "add it to _DTYPE_BYTES"
message even lured the agent into probing the grader source). A width has one size regardless of
spelling; only a genuinely unknown dtype should fail closed.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import capsule_dram as CD


def test_fp_synonyms_fold_to_canonical_width():
    assert CD.dtype_bytes("fp16") == CD.dtype_bytes("f16") == 2
    assert CD.dtype_bytes("fp32") == CD.dtype_bytes("f32") == 4
    assert CD.dtype_bytes("fp64") == CD.dtype_bytes("f64") == 8


def test_mx_block_float_is_whole_byte():
    assert CD.dtype_bytes("mxfp8") == 1
    assert CD.dtype_bytes("mxint8") == 1


def test_explicit_fp8_and_bf16_unchanged():
    # the fold must NOT clobber explicit tokens that merely start with 'f'
    assert CD.dtype_bytes("fp8_e4m3") == 1
    assert CD.dtype_bytes("bf16") == 2
    assert CD.dtype_bytes("f8E4M3FN") == 1


def test_tensor_nbytes_uses_folded_width():
    # 32x32 fp16 tile = 1024 elems * 2 bytes (would have crashed before the fix)
    assert CD.tensor_nbytes([32, 32], "fp16") == 1024 * 2
    assert CD.tensor_nbytes([32, 32], "mxfp8") == 1024 * 1


def test_genuinely_unknown_dtype_still_fails_closed():
    with pytest.raises(KeyError):
        CD.dtype_bytes("float8_e3m4_weird")


def _pointer_contract():
    return {
        "harness_abi": {
            "version": 2,
            "kind": "logical_pointer",
            "entry_symbol": "kernel_entry",
            "fence_symbol": "kernel_complete",
            "tensor_alignment": 64,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": "out_b64_v1",
        }
    }


def test_selected_pointer_harness_receives_original_storage_without_fixed_bases():
    from merlin.runtime.harness_render import logical_output_names

    cb = {
        "tensors": {
            "X": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": "i8", "role": "output"},
        }
    }
    # Capsule-address derivation must not run for a pointer ABI, even when the
    # scalar output shape cannot be inferred by the fixed-address mechanism.
    assert CD.inject_bases(cb, {}, contract=_pointer_contract()) is cb
    assert all("base" not in spec for spec in cb["tensors"].values())
    assert logical_output_names(cb) == ("Y",)
    cb["tensors"]["X"]["base"] = 4096
    CD.inject_bases(cb, {}, contract=_pointer_contract())
    with pytest.raises(ValueError, match="ambiguous tensor storage"):
        logical_output_names(cb)


def test_incomplete_selected_pointer_abi_cannot_bypass_address_validation():
    contract = _pointer_contract()
    del contract["harness_abi"]["version"]
    with pytest.raises(ValueError, match="complete logical-pointer"):
        CD.inject_bases({"tensors": {}}, {}, contract=contract)


@pytest.mark.parametrize("contract", [None, {}, {"harness_abi": {"entry_symbol": "legacy_entry"}}])
def test_fixed_address_storage_still_uses_capsule_layout(contract):
    capsule = {
        "inputs": [{"name": "X", "shape": [2, 3], "dtype": "i8", "role": "input"}],
        "numeric_policy": {"dtype": "i8"},
    }
    cb = {"tensors": {"X": {"shape": [2, 3], "dtype": "i8", "role": "input"}}}
    CD.inject_bases(cb, capsule, base=8192, contract=contract)
    assert cb["tensors"]["X"]["base"] == 8192

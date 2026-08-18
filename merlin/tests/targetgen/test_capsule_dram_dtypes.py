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

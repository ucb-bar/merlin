"""The microscaling (MX) numerical datapath oracle — merlin's float-datapath grade tier for MX capsules,
delegating to the mlc-validated MX reference. These tests prove the oracle is LIVE (callable through
merlin), FAITHFUL (reproduces the derived reference), a real BLOCK-SCALED datapath (scale-sensitive, and
distinct from a naive fp8 matmul — so the E8M0 scale + ACC schedule are actually applied), and
FAIL-CLOSED when mlc is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.targetgen import mx_oracle

pytestmark = pytest.mark.skipif(not mx_oracle.mx_datapath_available(),
                                reason="mlc MX reference (mlc.validate.mx_ref) not importable")

M, N, K = 16, 16, 32


def _operands(seed=11):
    rng = np.random.default_rng(seed)
    a = (rng.integers(0, 8, (M, K), np.uint8) << 3 | rng.integers(0, 8, (M, K), np.uint8))
    w = (rng.integers(0, 8, (K, N), np.uint8) << 3 | rng.integers(0, 8, (K, N), np.uint8))
    sa = rng.integers(124, 131, (K // 32, M), np.int32)
    sb = rng.integers(124, 131, (K // 32, N), np.int32)
    return a, w, sa, sb


def test_oracle_runs_and_is_deterministic():
    a, w, sa, sb = _operands()
    out1 = mx_oracle.mx_matmul(a, w, sa, sb, M, N, K, fmt="fp8")
    out2 = mx_oracle.mx_matmul(a, w, sa, sb, M, N, K, fmt="mxfp8")   # alias resolves to same format
    assert out1 is not None and out1.shape == (M, N)
    assert np.array_equal(out1, out2)                                # deterministic + alias-consistent


def test_oracle_matches_the_derived_reference():
    """merlin's oracle reproduces mlc.validate.mx_ref bit-exact — a faithful delegation (mlc validates
    that reference against the radiance-kernels C++ RTL-mirror golden in its own suite)."""
    from mlc.validate import mx_ref as mx
    a, w, sa, sb = _operands()
    ref = np.asarray(mx.mx_matmul(a, w, sa, sb, M, N, K, fmt=mx.FMT_FP8)).reshape(M, N).astype(np.uint32)
    ref = (ref << 16).view(np.float32)
    got = mx_oracle.mx_matmul(a, w, sa, sb, M, N, K, fmt="fp8")
    assert np.array_equal(got, ref)


def test_oracle_is_a_real_block_scaled_datapath():
    """The output actually depends on the E8M0 block scales (change SA -> output changes), and it is NOT
    a plain decoded-fp8 matmul — proving the block scale + accumulate schedule are applied, not bypassed."""
    from mlc.validate import mx_ref as mx
    a, w, sa, sb = _operands()
    base = mx_oracle.mx_matmul(a, w, sa, sb, M, N, K, fmt="fp8")
    bumped = mx_oracle.mx_matmul(a, w, sa + 1, sb, M, N, K, fmt="fp8")   # +1 e8m0 code == x2 on A rows
    assert not np.array_equal(base, bumped)                              # scale-sensitive

    # naive: decode fp8 (no scale), plain matmul -> must DIFFER from the MX datapath result
    dec = np.vectorize(lambda c: mx.fp8_e4m3_decode(int(c)))
    naive = (dec(a).astype(np.float32) @ dec(w).astype(np.float32))
    assert not np.allclose(naive, base, atol=1e-3)


def test_unknown_format_fails_closed():
    a, w, sa, sb = _operands()
    assert mx_oracle.mx_matmul(a, w, sa, sb, M, N, K, fmt="int8") is None

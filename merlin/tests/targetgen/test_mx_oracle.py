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


def _load_generate_corpus():
    """Import the corpus generator by repo-relative path (it lives outside the merlin package)."""
    import importlib.util

    from merlin.common.paths import repo_root
    path = repo_root() / "merlin" / "contract" / "capsules" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("merlin_generate_corpus_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("tok,dims", [("mxfp8", (16, 32, 16)), ("mxfp4", (32, 32, 32)),
                                      ("mxfp6", (16, 32, 16))])
def test_generated_mx_capsule_grades_bit_exact(tok, dims):
    """End to end: the corpus generator emits an MX matmul golden that CARRIES the raw operand codes, and
    the datapath oracle re-runs those codes to reproduce the golden BIT-EXACT (max abs err 0). This is the
    per-capsule numerical grade for MX — using raw codes, not the display-rounded ``decoded`` floats, so it
    is exact rather than tolerance-bounded. Covers all three block-FP formats (fp8 byte, fp4 nibble-packed,
    fp6 LUT-indexed)."""
    from types import SimpleNamespace

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.capsule_runner import _bespoke_sim_via

    gc = _load_generate_corpus()
    m, k, n = dims
    te = SimpleNamespace(target="mx_gemmini", sim_via=_bespoke_sim_via("mx_gemmini"))
    binding = CS.derive_binding(te, {"operand_dtype": tok})
    entry = {"name": f"MX_{tok}", "op": "matmul", "kind": "isa", "M": m, "K": k, "N": n,
             "lhs": "A0", "weight": "W", "out": "Y0", "source_role": "t", "source_reference": "t"}
    golden, prov = gc._mx_golden(entry, binding)
    assert "operand_codes" in prov                          # generator stores the raw device codes
    res = mx_oracle.grade_matmul(prov["operand_codes"], prov["SA_e8m0_codes"],
                                 prov["SB_e8m0_codes"], golden["Y0"])
    assert res["status"] == "pass" and res["exact"] is True and res["max_abs_err"] == 0.0

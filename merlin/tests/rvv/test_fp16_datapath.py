"""fp16 datapath: the f32-accumulate rewrite, its numerics, and the gate that catches a bad one.

These tests are host-only (no board, no cross-toolchain): they pin the IR rewrite and the
NUMERICAL CONTRACT. The emitted-instruction evidence (effective vtype e16, vfwmul.vf) is
recorded in the commit message / report, since it needs the RISC-V toolchain to reproduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import repo_root  # noqa: F401  (layout convention: no parents[N])


def _bundle(tmp_path, M=64, N=64, K=64):
    from merlin.rvvgen.workloads import gen_matmul_f16
    return gen_matmul_f16(tmp_path, M=M, N=N, K=K)


def test_gen_matmul_f16_emits_an_f16_matmul_bundle(tmp_path):
    b = _bundle(tmp_path)
    mlir = (b / "model.mlir").read_text()
    assert "linalg.matmul" in mlir
    assert "xf16>" in mlir and "xf32>" not in mlir      # authored in f16 end-to-end
    d = np.load(b / "inputs.npz")
    assert d["in0"].dtype == np.float16 and d["in1"].dtype == np.float16


def test_f32acc_rewrite_fires_on_fp16_and_accumulates_in_f32(tmp_path):
    """The bf16 pass also covers fp16 (Float16Type) -- extf to f32, accumulate, truncf back."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc
    from merlin.xdsl_dialects._common import text as to_text

    b = _bundle(tmp_path)
    module = parse_mlir_file(b / "model.mlir")
    assert lower_bf16_matmul_f32acc(module) == 1
    txt = to_text(module)
    assert "arith.extf" in txt and "arith.truncf" in txt
    # the reduction body must accumulate in f32, not f16
    assert "arith.addf" in txt and ": f32" in txt


def test_accumulate_ops_carry_the_contract_fastmath_flag(tmp_path):
    """Without `contract` LLVM may not fuse mul+add into an FMA at all.

    This is the license for a fused (widening) MAC; it is a single-rounding ACCURACY win, and
    no other fast-math flag is set (reduction order and NaN/Inf behavior are unchanged).
    """
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc
    from merlin.xdsl_dialects._common import text as to_text

    b = _bundle(tmp_path)
    module = parse_mlir_file(b / "model.mlir")
    lower_bf16_matmul_f32acc(module)
    txt = to_text(module)
    assert "arith.mulf" in txt and "fastmath<contract>" in txt
    for flag in ("reassoc", "nnan", "ninf", "nsz"):
        assert flag not in txt, f"unexpected fast-math relaxation {flag!r}"


def test_scalarize_gate_is_element_type_parameterized_not_f32_only():
    """The v3 micro-kernel A-scalarization (which produces the `.vf` MAC form) must admit the
    16-bit-float element types, or fp16 silently falls off the shared micro-kernel path."""
    from merlin.llvmlower.accum_microkernel import rewrite_source
    src = rewrite_source()
    assert 'for cand in ("f32", "f16", "bf16")' in src
    assert 'str(owner.results[0].type) != elem' in src


# ---------------------------------------------------------------------------------------
# The correctness gate. fp16 cannot be bit-exact, so the gate is statistical -- but the
# OBVIOUS statistical gate is not sufficient, which is the point of these two tests.
# ---------------------------------------------------------------------------------------

def _variants(M=128, N=128, K=128, seed=0):
    r = np.random.default_rng(seed)
    a = r.standard_normal((M, K)).astype(np.float16)
    b = r.standard_normal((K, N)).astype(np.float16)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    good = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16).astype(np.float64)
    bad = np.zeros((M, N), np.float16)          # a REAL f16-accumulating kernel
    for k in range(K):
        bad += (a[:, k:k + 1] * b[k:k + 1, :]).astype(np.float16)
    return exact, good, bad.astype(np.float64)


def _metrics(x, ref):
    cos = float((x * ref).sum() / (np.linalg.norm(x) * np.linalg.norm(ref) + 1e-12))
    rel_l2 = float(np.linalg.norm(x - ref) / (np.linalg.norm(ref) + 1e-12))
    max_rel = float(np.max(np.abs(x - ref) / np.maximum(np.abs(ref), 1e-3)))
    return cos, rel_l2, max_rel


def test_aggregate_only_gate_would_ACCEPT_an_f16_accumulating_kernel():
    """REGRESSION GUARD on the gate itself, not on the kernel.

    The int8 driver's gate (cos > 0.99 AND relative-L2 < 5e-2) is an AGGREGATE gate, and both
    terms are dominated by the bulk of well-conditioned outputs. An f16-ACCUMULATING matmul --
    a genuinely broken datapath, off by >1000% on individual elements from catastrophic
    cancellation in the 10-bit mantissa -- sails through it. So fp16 must NOT reuse that gate
    as-is; it needs the per-element term asserted in the next test.
    """
    exact, _good, bad = _variants()
    cos, rel_l2, max_rel = _metrics(bad, exact)
    assert cos > 0.99 and rel_l2 < 5e-2          # the int8-tier gate is satisfied ...
    assert max_rel > 1.0                          # ... yet an element is off by >100%


def test_fp16_gate_cos_relL2_and_MAXREL_separates_good_from_bad():
    """The gate fp16 actually uses: cos > 0.9999 AND rel-L2 < 1e-2 AND max-rel < 0.05.

    Threshold justification -- f16 has 10 explicit mantissa bits, so eps ~= 9.8e-4. A correctly
    f32-accumulated product rounded ONCE to f16 carries <= ~1 ulp of relative error (measured:
    max-rel 9e-4 at 128^3). 0.05 is ~50x that headroom, so it cannot flake on rounding; the
    f16-accumulating kernel misses by 12.09, ~240x over the line. Wide separation both ways.
    """
    exact, good, bad = _variants()

    def gate(x):
        cos, rel_l2, max_rel = _metrics(x, exact)
        return cos > 0.9999 and rel_l2 < 1e-2 and max_rel < 0.05

    assert gate(good), "the f32-accumulate datapath must PASS"
    assert not gate(bad), "an f16-accumulate datapath must FAIL"


@pytest.mark.parametrize("K", [64, 128, 256])
def test_f32acc_stays_within_the_gate_as_the_reduction_lengthens(K):
    """f16 accumulation degrades with K; f32 accumulation must not."""
    exact, good, _bad = _variants(M=64, N=64, K=K, seed=K)
    _cos, _rel, max_rel = _metrics(good, exact)
    assert max_rel < 0.05

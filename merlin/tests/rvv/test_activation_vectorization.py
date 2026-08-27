"""Vectorized-transcendental-activation feature: build THROUGH the real RVV compiler and assert it
(a) vectorizes the activation to a vfmacc chain (the baseline emits a scalar libm-call loop, zero
vector ops), and (b) stays accurate within the stated approximation tolerance vs the libm golden on
spike — across GELU (erf), sigmoid (exp) AND SiLU (exp), so the feature is GENERAL, not one
memorized activation/size.

The vectorize assertions are build-only (decode model.o); the accuracy assertions boot spike and
gate on cosine / relative-error (these are APPROXIMATIONS — the gate is NOT bit-exact). All
auto-skip without the riscv toolchain / spike. The full instret gap-closure measurement (GELU ~6x,
sigmoid/SiLU ~3.2x vs the scalar baseline at 1K/16K/256K) is recorded in the task report.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()
FEATURE = "vectorized_transcendental_activation"


def _toolchain() -> bool:
    # These tests compile a whole model through zephyr_model.build_app (the Zephyr/spike path), which
    # needs the Zephyr SW toolchain (ZEPHYR_BASE / MERLIN_ZEPHYR_SW / SDK), not just the asm toolchain.
    # Gate on BOTH so the test SKIPS cleanly when Zephyr is absent instead of hard-failing inside
    # build_app with ZephyrModelError (honest fail-closed).
    try:
        from merlin.kernels import build_asm
        from merlin.runtime.backends import zephyr_model
        return build_asm.asm_toolchain_available() and zephyr_model.available()
    except Exception:
        return False


def _pkg(features):
    from merlin.mining.registry import load_rvv_package
    pkg = load_rvv_package(REPO / "out/artifacts/targets" / "rvv" / "hand_v0")
    return replace(pkg, run_id="test_act", compiler_features=list(features))


def _build(features, bundle):
    """apply_rvv_package -> (decoded InsnStream, build dict, work dir)."""
    from merlin.kernels.decode import rvv
    from merlin.mining.apply import apply_rvv_package
    work = Path(tempfile.mkdtemp(prefix="test_act_"))
    build = apply_rvv_package(_pkg(features), bundle, work,
                              board="spike_riscv64", harts=1, arena_mb=64)
    return rvv.decode(str(work / "model.o")), build, work


def _gens():
    from merlin.mining import workloads
    return {"gelu": workloads.gen_gelu_f32, "sigmoid": workloads.gen_sigmoid_f32,
            "silu": workloads.gen_silu_f32}


# A REALISTIC activation-bearing module: a rank-4 attention softmax-exp generic (bitvla's actual
# activation: tensor<1x8x32x32xf32>), a rank-0 scalar generic (iterator_types=[] — bitvla has 56 of
# these), AND an index-bearing generic (linalg.index — bitvla has 20). The OLD schedule's hard-coded
# rank-1 `tile_sizes [16]` raised "too many tiles provided, expected at most 0 found 1" on the rank-0
# generic and fell the whole model back to scalar (k1_e2e_activation.md). This module reproduces that
# exact structure on HOST so the regression can't recur without the board.
_REALISTIC_ACTIVATION_MLIR = """\
builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%x: tensor<1x8x32x32xf32>, %s: tensor<f32>) -> (tensor<1x8x32x32xf32>, tensor<f32>, tensor<64xf32>) {
    %c1 = arith.constant 1.000000e+00 : f32
    %ao = tensor.empty() : tensor<1x8x32x32xf32>
    %act = linalg.generic {indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(a,b,c,d)>], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x : tensor<1x8x32x32xf32>) outs(%ao : tensor<1x8x32x32xf32>) {
    ^bb0(%in: f32, %o: f32):
      %e = math.exp %in : f32
      linalg.yield %e : f32
    } -> tensor<1x8x32x32xf32>
    %so = tensor.empty() : tensor<f32>
    %sc = linalg.generic {indexing_maps = [affine_map<()->()>, affine_map<()->()>], iterator_types=[]} ins(%s : tensor<f32>) outs(%so : tensor<f32>) {
    ^bb1(%in: f32, %o: f32):
      %y = arith.addf %in, %c1 : f32
      linalg.yield %y : f32
    } -> tensor<f32>
    %io = tensor.empty() : tensor<64xf32>
    %iota = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>], iterator_types=["parallel"]} outs(%io : tensor<64xf32>) {
    ^bb2(%o: f32):
      %i = linalg.index 0 : index
      %fi = arith.index_cast %i : index to i32
      %ff = arith.sitofp %fi : i32 to f32
      linalg.yield %ff : f32
    } -> tensor<64xf32>
    return %act, %sc, %iota : tensor<1x8x32x32xf32>, tensor<f32>, tensor<64xf32>
  }
}
"""


# A mixed module with a SOFTMAX (prov.family="normalization", an exp that must NOT be approximated)
# AND a GELU (prov.op="gelu", an erf that MUST be approximated+vectorized). The blanket rewriter
# replaced BOTH exps with the minimax poly; the softmax exp's ~1e-7 error then amplified through the
# row-sum normalization -> openvla whole-model cos 0.541. This module reproduces that exact structure
# (softmax exp + gelu erf) on HOST so the precise-targeting fix (provenance: softmax exp stays libm,
# gelu erf -> vectorized poly) is locked in.
_SOFTMAX_PLUS_GELU_MLIR = """\
builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%x: tensor<1x4x16x16xf32>, %g: tensor<64xf32>) -> (tensor<1x4x16x16xf32>, tensor<64xf32>) {
    %ninf = arith.constant -3.40282347E+38 : f32
    %z = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32
    %ch = arith.constant 5.000000e-01 : f32
    %cs = arith.constant 0.7071067811865476 : f32
    %me = tensor.empty() : tensor<1x4x16xf32>
    %mf = linalg.fill ins(%ninf : f32) outs(%me : tensor<1x4x16xf32>) -> tensor<1x4x16xf32>
    %mx = linalg.generic {indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(a,b,c)>], iterator_types=["parallel","parallel","parallel","reduction"]} ins(%x : tensor<1x4x16x16xf32>) outs(%mf : tensor<1x4x16xf32>) attrs={prov.op="softmax", prov.family="normalization"} {
    ^bbm(%i: f32, %o: f32):
      %m = arith.maximumf %i, %o : f32
      linalg.yield %m : f32
    } -> tensor<1x4x16xf32>
    %eo = tensor.empty() : tensor<1x4x16x16xf32>
    %ex = linalg.generic {indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(a,b,c)>, affine_map<(a,b,c,d)->(a,b,c,d)>], iterator_types=["parallel","parallel","parallel","parallel"]} ins(%x, %mx : tensor<1x4x16x16xf32>, tensor<1x4x16xf32>) outs(%eo : tensor<1x4x16x16xf32>) attrs={prov.op="softmax", prov.family="normalization"} {
    ^bbe(%i: f32, %m: f32, %o: f32):
      %s = arith.subf %i, %m : f32
      %e = math.exp %s : f32
      linalg.yield %e : f32
    } -> tensor<1x4x16x16xf32>
    %go = tensor.empty() : tensor<64xf32>
    %gel = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types=["parallel"]} ins(%g : tensor<64xf32>) outs(%go : tensor<64xf32>) attrs={prov.op="gelu", prov.family="elementwise"} {
    ^bbg(%in: f32, %o: f32):
      %t = arith.mulf %in, %cs : f32
      %er = math.erf %t : f32
      %p = arith.addf %er, %c1 : f32
      %h = arith.mulf %p, %ch : f32
      %y = arith.mulf %h, %in : f32
      linalg.yield %y : f32
    } -> tensor<64xf32>
    return %ex, %gel : tensor<1x4x16x16xf32>, tensor<64xf32>
  }
}
"""


def test_rewriter_targets_activation_provenance_not_softmax():
    # PRECISE TARGETING (the openvla cos-0.541 fix): the poly rewriter must rewrite a transcendental
    # ONLY inside a provenance-marked elementwise ACTIVATION generic and LEAVE a softmax/normalization
    # exp alone. The classifier runs inside the m2m-venv rewriter source, so assert its contract on
    # that source string (the importable surface — same approach as the v3 matcher-guard test).
    from merlin.llvmlower.act_poly import rewrite_source
    src = rewrite_source()
    # the activation-op set carries gelu/silu/sigmoid/tanh and does NOT carry softmax.
    assert "_AP_ACTIVATION_OPS" in src
    assert '"gelu"' in src and '"sigmoid"' in src and '"silu"' in src and '"tanh"' in src
    # it gates on the enclosing generic's provenance (op identity), not op-class alone...
    assert "prov.op" in src and "prov.family" in src
    # ...explicitly EXCLUDES softmax/normalization (the exp that must stay on the exact libm path)...
    assert '"softmax"' in src and "normalization" in src
    # ...and TAGS each targeted generic so the schedule vectorizes exactly those.
    assert "merlin.act_vectorize" in src
    # the body is PURE ARITH (no math.fma/absf/roundeven OP CONSTRUCTORS) so the softmax exp can take
    # the libm path without a convert-math-to-llvm pass (which would crash on llvm.intr.exp). The
    # comments may mention these op names; assert no CONSTRUCTOR CALL emits them.
    assert "_math.FmaOp(" not in src and "_math.RoundEvenOp(" not in src and "_math.AbsFOp(" not in src


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_softmax_exp_stays_libm_gelu_vectorizes():
    # On a module that MIXES a softmax exp (normalization) and a gelu erf (activation), lowering with
    # the feature ON must: (a) succeed, (b) leave the softmax exp on the exact exp path (NOT the
    # minimax poly — the math.exp survives as an exp intrinsic/call, not a Horner chain), and (c)
    # vectorize the gelu poly to a vector fma chain. Reproduces openvla's structure on HOST.
    import re
    from merlin.llvmlower import pipeline as P
    ll = P.lower_to_llvm_ir(_SOFTMAX_PLUS_GELU_MLIR, vectorize=True,
                            features=frozenset([FEATURE]), timeout=900)
    # the gelu erf was approximated + VECTORIZED (a >1-lane vector fp arith from the poly chain)...
    assert re.search(r"f(mul|add) <[0-9]+ x float>", ll), "gelu poly did not vectorize"
    assert "math.erf" not in ll                 # the gelu erf was rewritten to the arith poly
    # ...and the softmax exp was NOT poly-rewritten: it took the EXACT libm path (scalar `expf`), NOT
    # `llvm.intr.exp` (`llvm.exp.f32`) — that intrinsic is what crashed openvla whole-model on spike.
    assert "@expf" in ll, "softmax exp must lower to the scalar libm expf (the crash-free path)"
    assert "llvm.exp" not in ll, "softmax exp must NOT become llvm.intr.exp (the spike crash source)"


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_realistic_activation_module_lowers_without_pipeline_error():
    # REGRESSION (k1_e2e_activation.md): with the feature ON, a realistic activation module that mixes
    # a rank-4 softmax-exp generic, a rank-0 scalar generic, and an index-bearing generic must lower
    # through the real RVV pipeline WITHOUT the "too many tiles provided, expected at most 0 found 1"
    # PipelineError (which previously forced whole-model scalar fallback / a 0.76x regression). The
    # synthetic rank-1 gelu/sigmoid workloads never exercised the rank-0 / rank-N / index generics, so
    # this is the test that locks the fix in. Lowering succeeding (no raise) is the assertion.
    from merlin.llvmlower import pipeline as P
    ll = P.lower_to_llvm_ir(_REALISTIC_ACTIVATION_MLIR, vectorize=True,
                            features=frozenset([FEATURE]), timeout=900)
    assert "math.exp" not in ll                # the transcendental was rewritten to the arith poly
    # the rank-4 exp activation vectorized: a >1-lane vector fmul from the polynomial Horner chain
    assert "x float>" in ll and " fmul <" in ll


def _n_vector_fp(stream):
    """Number of vector floating-point arith ops (the vectorized polynomial chain). The activation
    poly is PURE ARITH (mul+add — see act_poly._ap_fma) so it vectorizes to vfmul.vv/vfadd.vv (the
    RISC-V backend may further contract adjacent pairs into vfmacc, but that is not guaranteed and is
    NOT the success criterion — VECTORIZATION is). We avoid math.fma on purpose: it would force a
    convert-math-to-llvm pass that ALSO converts the un-rewritten softmax exp to llvm.intr.exp, which
    crashes the freestanding spike/RVV runtime (the openvla 'bad syscall')."""
    return sum(stream.count(mn) for mn in
               ("vfmul.vv", "vfadd.vv", "vfsub.vv", "vfmacc.vv", "vfmacc.vf", "vfmadd.vv"))


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.parametrize("op", ["gelu", "sigmoid", "silu"])
def test_activation_vectorizes_when_on(op):
    # GENERAL: each activation (gelu=erf, sigmoid/silu=exp) must VECTORIZE to a vector-fp polynomial
    # chain with the feature on, and emit ZERO vector ops with the feature off (scalar libm-call
    # loop). One feature, three activations — proves it is not gelu/sigmoid-overfit. The criterion is
    # vectorization (vector vfmul/vfadd...), not fused vfmacc specifically (see _n_vector_fp).
    bundle = _gens()[op](tempfile.mkdtemp(), N=1024)
    base, _, _ = _build([], bundle)
    assert _n_vector_fp(base) == 0                     # baseline: scalar libm loop, no vectorization
    on, _, _ = _build([FEATURE], bundle)
    assert _n_vector_fp(on) > 0                        # feature: vectorized polynomial chain


def test_activation_polynomials_accurate_on_realistic_ranges():
    # ACCURACY on REALISTIC ranges (the corollary: verify on realistic inputs, not just synthetic
    # small). Mirror act_poly's EXACT exp + erf coefficients/structure in numpy and check cos/max-abs
    # vs libm over N(0,3) (typical activations), N(0,10) (wide logits) and N(0,30) (extreme tails) —
    # the ranges a real model's gelu/sigmoid actually carry. No toolchain/spike: this isolates the
    # polynomial math from the lowering, so a coefficient regression is caught fast and deterministically.
    def poly_exp(x):
        x = x.astype(np.float32)
        x = np.maximum(x, np.float32(-87.0)); x = np.minimum(x, np.float32(88.0))
        nf = np.rint(x * np.float32(1.4426950408889634)).astype(np.float32)
        r = nf * np.float32(-0.6931471824645996) + x
        r = nf * np.float32(1.904654323148236e-09) + r
        p = np.float32(0.0013888889)
        for c in (0.008333334, 0.041666668, 0.16666667, 0.5, 1.0, 1.0):
            p = p * r + np.float32(c)
        scale = ((nf.astype(np.int32) + 127) << 23).view(np.float32)
        return p * scale

    def poly_erf(x):
        x = x.astype(np.float32)
        ax = np.abs(x); s = np.where(x >= 0, np.float32(1.0), np.float32(-1.0))
        tt = np.float32(1.0) / (np.float32(0.3275911) * ax + np.float32(1.0))
        poly = np.float32(1.061405429)
        for c in (-1.453152027, 1.421413741, -0.284496736, 0.254829592):
            poly = poly * tt + np.float32(c)
        poly = poly * tt
        return s * (np.float32(1.0) - poly * poly_exp(-(ax * ax)))

    from math import erf as _erf
    rng = np.random.default_rng(0)

    def cos(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    for scale in (3.0, 10.0, 30.0):
        x = (rng.standard_normal(50000).astype(np.float32)) * scale
        ref_erf = np.vectorize(_erf)(x.astype(np.float64) / np.sqrt(2.0))    # gelu's erf(x/sqrt2)
        my_erf = poly_erf(x / np.float32(np.sqrt(2.0)))
        ref_sig = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))                # sigmoid 1/(1+exp(-x))
        my_sig = np.float32(1.0) / (np.float32(1.0) + poly_exp(-x))
        assert cos(my_erf, ref_erf) > 0.999, f"erf cos low at scale {scale}"
        assert np.max(np.abs(my_erf - ref_erf)) < 5e-6, f"erf max-abs high at scale {scale}"
        assert cos(my_sig, ref_sig) > 0.999, f"sigmoid cos low at scale {scale}"
        assert np.max(np.abs(my_sig - ref_sig)) < 5e-6, f"sigmoid max-abs high at scale {scale}"


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.parametrize("op", ["gelu", "sigmoid", "silu"])
def test_activation_accurate_on_spike(op):
    # The activation is an APPROXIMATION (f32 minimax polynomial), so the gate is cos / relative
    # error vs the libm golden, NOT bit-exact. Tolerance is generous vs the measured ~1e-7 so the
    # test is not brittle, but tight enough to catch a wrong polynomial. Runs on spike (correctness
    # authority). General: asserted for gelu, sigmoid AND silu.
    from merlin.runtime.backends import zephyr_model as zm
    bundle = _gens()[op](tempfile.mkdtemp(), N=1024)
    _, build, _ = _build([FEATURE], bundle)
    refs = {"fp32": np.load(bundle / "golden.npy")}
    run = zm.run_on_spike(build["elf"], harts=1,
                          mem_bytes=build.get("ram_bytes", 1 << 31), timeout=900)
    gate = zm._gate(run["prefix"], refs)
    assert gate.get("ok") is True
    assert gate.get("fp32_cos") is not None and gate["fp32_cos"] > 0.9999
    assert gate.get("fp32_rel") is not None and gate["fp32_rel"] < 1e-4

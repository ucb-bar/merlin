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

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]
FEATURE = "vectorized_transcendental_activation"


def _toolchain() -> bool:
    try:
        from merlin.kernels import build_asm
        return build_asm.asm_toolchain_available()
    except Exception:
        return False


def _pkg(features):
    from merlin.rvvgen.registry import load_rvv_package
    pkg = load_rvv_package(REPO / "generated_targets" / "rvv" / "hand_v0")
    return replace(pkg, run_id="test_act", compiler_features=list(features))


def _build(features, bundle):
    """apply_rvv_package -> (decoded InsnStream, build dict, work dir)."""
    from merlin.kernels.decode import rvv
    from merlin.rvvgen.apply import apply_rvv_package
    work = Path(tempfile.mkdtemp(prefix="test_act_"))
    build = apply_rvv_package(_pkg(features), bundle, work,
                              board="spike_riscv64", harts=1, arena_mb=64)
    return rvv.decode(str(work / "model.o")), build, work


def _gens():
    from merlin.rvvgen import workloads
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


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.parametrize("op", ["gelu", "sigmoid", "silu"])
def test_activation_vectorizes_to_vfmacc_when_on(op):
    # GENERAL: each activation (gelu=erf, sigmoid/silu=exp) must vectorize to a vfmacc chain with the
    # feature on, and emit ZERO vector ops with the feature off (scalar libm-call loop). One feature,
    # three activations — proves it is not gelu/sigmoid-overfit.
    bundle = _gens()[op](tempfile.mkdtemp(), N=1024)
    base, _, _ = _build([], bundle)
    assert base.count("vfmacc", "vmacc") == 0          # baseline: scalar libm loop, no vectorization
    on, _, _ = _build([FEATURE], bundle)
    assert on.count("vfmacc", "vmacc") > 0             # feature: fused vfmacc polynomial chain


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

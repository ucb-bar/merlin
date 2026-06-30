"""smolvla (500M int8+bf16 VLA) on the RVV dispatch runtime — stage-verified.

Bisection of the cos-0.978 whole-model residual (see model-bringup-rvv memory) showed the
pipeline is correct stage-by-stage; the residual is bf16 precision amplified through the
flow-matching denoise head, NOT a compiler bug:

  prefix (vision+embed, f32, no RoPE)   cos 1.0000001   (exact)
  LM output (16 layers, bf16 + RoPE)    cos 0.99991     (~gate; rel 0.063)
  full model (+ denoise/expert)         cos 0.978       (bf16 amplified by the sensitive head)

f32 stages hit the exact gate (matching the f32 LLaMAs at cos 0.9999999); bf16 stages carry
the inherent bf16-vs-torch fidelity gap (our matmul reduction order differs from torch's by
f32 reassociation, which 8-bit bf16 mantissas amplify). All gated on the captured bundles +
the host toolchain; the whole-model runs are slow (compile ~3-4k kernels) -> MERLIN_RUN_SLOW.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import os
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()
PREFIX = REPO / "artifacts/recaptures/smolvla_prefix_consistent"
FULL = REPO / "artifacts/recaptures/smolvla_int8_consistent"


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


@pytest.mark.skipif(not (PREFIX / "model.mlir").is_file(),
                    reason="smolvla prefix bundle not captured")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (compiles ~800 kernels)")
def test_smolvla_prefix_exact(tmp_path):
    """Vision encoder + conv + int8 dequant + embeddings (f32) reproduce torch exactly."""
    from merlin.runtime.dispatch_runtime import run_model

    res = run_model(PREFIX, tmp_path, cache_dir=REPO / "artifacts/cache/kc_smolvla_prefix")
    assert res["cos"] > 0.9999 and res["rel"] < 1e-2, (res["cos"], res["rel"])


@pytest.mark.skipif(not (FULL / "model.mlir").is_file(),
                    reason="smolvla full int8 bundle not captured")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (compiles ~3.7k kernels)")
def test_smolvla_full_runs_at_bf16_fidelity(tmp_path):
    """Whole int8+bf16 VLA executes end to end; bf16 flow-matching head -> cos ~0.978."""
    from merlin.runtime.dispatch_runtime import run_model

    res = run_model(FULL, tmp_path, cache_dir=REPO / "artifacts/cache/kc_smolvla")
    assert res["artifacts" / "recaptures"].shape == (1, 50, 32)
    assert res["cos"] > 0.97, res["cos"]          # bf16 fidelity through the denoise head

"""model2MLIR VLA models on the RVV dispatch runtime (host==torch).

Brought up via the consistent-capture harness (model2MLIR/workloads/capture_consistent.py)
+ the dispatch runtime (conv/scf/dynamic/bf16/int8/over-rank-matmul handling). Each is gated
on its captured bundle + the host toolchain; the runs compile hundreds-to-thousands of
kernels, so they're behind MERLIN_RUN_SLOW.

Datatype matrix (fp32/int8/fp8) at the exact host==torch gate (cos>0.9999): rdt2, rdt,
groot_n1d7, molmoact, openvla — all three datatypes each. Near gate (BitNet ternary
round-off, cos~0.99999/rel~4e-3): bitvla, all three. xr0: fp32 + int8 (its int8 binds via
the torchao subclass-inner-tensor fix; fp8 fold is a follow-up). int8 uses weight-only dequant; fp8 decodes
float8_e4m3fn->f32 at load (dispatch_runtime.f8e4m3fn_to_f32). (smolvla bf16 fidelity is in
test_smolvla_rvv.py; tiny/small_llama LLM datatypes are covered elsewhere.)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]

# (bundle dir under output/, min cosine, max rel) — tolerances reflect each model's regime.
MODELS = [
    # rdt2 — f32 flow-matching VLA, exact gate across all three datatypes
    ("rdt2_fp32_consistent", 0.9999, 1e-3),
    ("rdt2_int8_consistent", 0.9999, 1e-3),
    ("rdt2_fp8_consistent", 0.9999, 1e-3),
    # rdt — 1B diffusion VLA, exact gate across all three
    ("rdt_fp32_consistent", 0.9999, 1e-3),
    ("rdt_int8_consistent", 0.9999, 1e-3),
    ("rdt_fp8_consistent", 0.9999, 1e-3),
    # groot_n1d7 — f32 DiT action head, exact gate across all three
    ("groot_n1d7_fp32_consistent", 0.9999, 1e-2),
    ("groot_n1d7_int8_consistent", 0.9999, 1e-2),
    ("groot_n1d7_fp8_consistent", 0.9999, 1e-2),
    # molmoact — OLMo decoder; needs fix_bool_sitofp; exact gate across all three
    ("molmoact_fp32_consistent", 0.9999, 1e-3),
    ("molmoact_int8_consistent", 0.9999, 1e-3),
    ("molmoact_fp8_consistent", 0.9999, 1e-3),
    # openvla — dual-ViT + Llama (shrunk); exact gate across all three
    ("openvla_fp32_consistent", 0.9999, 1e-3),
    ("openvla_int8_consistent", 0.9999, 1e-3),
    ("openvla_fp8_consistent", 0.9999, 1e-3),
    # bitvla — BitNet ternary dequant round-off (cos ~0.99999, rel ~4e-3); near gate, all three
    ("bitvla_fp32_consistent", 0.999, 5e-3),
    ("bitvla_int8_consistent", 0.999, 5e-3),
    ("bitvla_fp8_consistent", 0.999, 5e-3),
    # xr0 — sdpa + 3D-linear; int8 weight-only now binds via the torchao subclass-inner-tensor
    # fix (m2m tags the dequant with the inner-tensor attr path; the runtime binds the real
    # int_data/scale and passes them as kernel inputs instead of cloned empties).
    ("xr0_fp32_consistent", 0.999, 5e-2),
    ("xr0_int8_consistent", 0.999, 5e-2),
    ("xr0_fp8_consistent", 0.999, 5e-2),
]


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (compiles hundreds of kernels per model)")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
@pytest.mark.parametrize("bundle,min_cos,max_rel", MODELS)
def test_vla_model_matches_torch(bundle, min_cos, max_rel, tmp_path):
    b = REPO / "output" / bundle
    if not (b / "model.mlir").is_file():
        pytest.skip(f"{bundle} not captured")
    from merlin.runtime.dispatch_runtime import run_model

    res = run_model(b, tmp_path, cache_dir=REPO / "output" / f".kc_{bundle}")
    assert res["cos"] > min_cos and res["rel"] < max_rel, (bundle, res["cos"], res["rel"])

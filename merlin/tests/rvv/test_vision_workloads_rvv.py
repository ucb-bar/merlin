"""The vision / audio / control workloads on the RVV dispatch runtime.

These four models are the first conv-heavy, spectral and recurrent captures in the corpus, and
each one reaches real linalg only because of frontend work in model2MLIR (padded / rank-3 /
grouped / transposed convolution as im2col + matmul, ``torch.fft`` as real DFT contractions,
bilinear resize as a resize contraction, inference batch norm, integer ``abs``). Structural
checks live in model2MLIR's own suite; what can only be checked HERE is whether the emitted MLIR
computes the right numbers, because that needs a runtime to execute it.

So the gate is: compile every outlined kernel, run the whole model on the host, and compare
against the torch golden captured in the same process. A wrong index map or a wrong
normalization shows up as a mismatch; valid-looking IR does not hide it.

Whole-model runs compile hundreds to thousands of kernels, so they are behind
``MERLIN_RUN_SLOW`` like the other whole-model tests here.
"""
from __future__ import annotations

import os

import pytest

from merlin.common.artifacts import recaptures_dir
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

RECAP = recaptures_dir()

#: (workload, why it is interesting, the gate it must clear).
#: The tolerances are the DEFAULT bundle gate (cos > 0.9999, rel < 1e-3); none of these needed a
#: looser floor, and adding a redundant per-model entry would only weaken the shared default.
WORKLOADS = [
    ("spectformer", "spectral gating: rfft2 -> complex gate -> irfft2 as DFT contractions"),
    ("whisper_tiny", "Conv1d encoder stem + cross-attention decoder step"),
    ("lstmnetvit", "padded/depthwise conv, PixelShuffle, bilinear resize, unrolled LSTM"),
    ("deepjscc", "ConvTranspose, reflection padding (integer abs), inference batch norm"),
]


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


def _bundle(workload: str, dtype: str):
    from merlin.baselines.bundle import resolve

    return resolve(workload, dtype).root


@pytest.mark.parametrize("workload,reason", WORKLOADS, ids=[w for w, _ in WORKLOADS])
def test_bundle_is_fully_lowered(workload, reason):
    """No opaque ``aten_*`` call may survive in the captured module.

    An opaque call carries an external declaration, so it is an undefined symbol at link time
    rather than a slow path -- a capture that still has one cannot be built for any target. Cheap
    enough to run without MERLIN_RUN_SLOW: it only reads the text.
    """
    d = _bundle(workload, "fp32")
    if not (d / "model.mlir").is_file():
        pytest.skip(f"{workload} fp32 bundle not captured")
    text = (d / "model.mlir").read_text()
    opaque = sorted({line.split("@")[1].split("(")[0]
                     for line in text.splitlines()
                     if line.lstrip().startswith("func.func private @aten_")})
    assert not opaque, f"{workload} ({reason}) still has opaque calls: {opaque}"


@pytest.mark.parametrize("workload,reason", WORKLOADS, ids=[w for w, _ in WORKLOADS])
def test_spectral_and_conv_land_on_contractions(workload, reason):
    """Each capture must carry real ``linalg.matmul``.

    A vector schedule matches contractions BY OP NAME, and the parallel-loop split only covers
    matmul/batch_matmul, so an op that lowers to a fused ``linalg.generic`` instead gets neither
    vectorization nor multicore. This asserts the property the frontend work exists to produce;
    it does not assert HOW MANY, which would break on every harmless canonicalization.
    """
    d = _bundle(workload, "fp32")
    if not (d / "model.mlir").is_file():
        pytest.skip(f"{workload} fp32 bundle not captured")
    text = (d / "model.mlir").read_text()
    assert "linalg.matmul" in text, f"{workload} has no named matmul; it would run scalar"


@pytest.mark.parametrize("dtype", ["fp32", "int8"])
@pytest.mark.parametrize("workload,reason", WORKLOADS, ids=[w for w, _ in WORKLOADS])
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (compiles hundreds of kernels per model)")
def test_host_matches_torch_golden(workload, reason, dtype, tmp_path):
    """Whole model on the host == the torch golden captured with it.

    This is the stage that proves the new decompositions are numerically right rather than merely
    well-typed. For int8 the reference is the weight-only golden the capture recorded, so the
    comparison is of the same computation -- W8A8 activation quantization is a separate tier and
    is gated by the Zephyr/spike path, not here.
    """
    from merlin.baselines.bundle import tolerance
    from merlin.runtime.dispatch_runtime import run_model

    d = _bundle(workload, dtype)
    if not (d / "model.mlir").is_file():
        pytest.skip(f"{workload} {dtype} bundle not captured")
    cache = RECAP.parent / "cache" / f"kc_{workload}_{dtype}"
    res = run_model(d, tmp_path, cache_dir=cache)
    min_cos, max_rel = tolerance(workload, dtype)
    assert res["cos"] > min_cos, f"{workload} {dtype} ({reason}): cos={res['cos']}"
    assert res["rel"] < max_rel, f"{workload} {dtype} ({reason}): rel={res['rel']}"

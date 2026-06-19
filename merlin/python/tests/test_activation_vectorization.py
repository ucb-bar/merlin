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

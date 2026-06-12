"""Whole-model execution on spike == host == torch.

Builds a small but complete LLaMA (RMSNorm/RoPE/attention/softmax/SwiGLU/lm_head) end to
end — MLIR → LLVM IR → rv64gcv → the Merlin C runtime (generic descriptor builder + arg
table + weights blob + bump allocator) → spike — and gates the output against the torch
golden. The artifacts must already be captured under ``output/small_consistent`` (a
seeded, consistent inputs+golden+MLIR+weights bundle). Skips when the chipyard toolchain
or the captured model is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "output/small_consistent"


def _toolchain():
    from merlin.runtime.backends import spike

    return spike.available()


pytestmark = pytest.mark.skipif(
    not (MODEL / "model.mlir").is_file(), reason="small_llama capture not present")


def test_c_runtime_generation_is_data_driven(tmp_path):
    """The arg table + trampoline + weights blob generate from the MLIR signature."""
    from merlin.llvmlower import c_runtime

    info = c_runtime.generate(MODEL, tmp_path, MODEL / "inputs.npz")
    assert info["out_shape"] == [1, 8, 256]
    gen = (tmp_path / "model_gen.h").read_text()
    assert "MERLIN_N_ARGS 23" in gen          # 22 forward args + output
    assert "MERLIN_ARGS" in gen
    call = (tmp_path / "model_call.c").read_text()
    assert call.count("d[") == 23             # unrolled ciface arity
    assert (tmp_path / "weights.bin").stat().st_size == info["weights_bytes"]


@pytest.mark.skipif(not _toolchain(), reason="chipyard toolchain/spike not available")
def test_small_llama_whole_model_on_spike(tmp_path):
    import numpy as np

    from merlin.runtime.backends import spike_model

    gold = np.load(MODEL / "golden.npy")
    res = spike_model.build_and_run(MODEL, tmp_path, arena_mb=64, reference=gold)
    # spike output matches torch to f32 round-off (RVV vs x86 FP reassociation ~1e-7).
    assert res["ok"], f"cos={res['cos']} rel={res['rel']}"
    assert res["cos"] > 0.9999
    assert res["rel"] < 1e-4
    assert res["metrics"]["cycles"] > 0

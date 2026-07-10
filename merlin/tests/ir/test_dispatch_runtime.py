"""Whole-model execution through the Merlin dispatch table (host reference runtime).

Outlines a captured model into per-dispatch kernels, compiles each kernel in isolation,
walks the driver evaluating view ops in numpy + invoking the compiled kernels, and gates
the whole-model output against the torch golden. Auto-skips without the host toolchain or
the captured model. (~40 s: it compiles ~160 kernels.)
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import os
from pathlib import Path

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()
MODEL = REPO / "out/artifacts/recaptures/small_consistent"
TINY = REPO / "out/artifacts/recaptures/tiny_consistent"


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


def test_resolve_forward_args_binds_inputs_and_weights():
    """Every forward arg gets a numpy array; weights come from the safetensors blob."""
    if not (MODEL / "model.mlir").is_file():
        pytest.skip("small_llama capture not present")
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    args = resolve_forward_args(MODEL)
    assert len(args) == 22                       # 1 input + 21 weights
    assert args[0].shape == (256, 128)           # emb.weight, read from the blob
    assert all(isinstance(a, np.ndarray) for a in args)


# A reduction generic with a by-value scalar accumulator-init arg (the cumsum/position-id
# pattern in tiny_llama). Guards the scalar-arg-by-value path: emit_c_interface passes the
# `%init : i64` by value, not as a memref descriptor.
CUMSUM = """
builtin.module {
  func.func @forward(%mask: tensor<1x4xi1>, %acc: tensor<1x4xi64>, %init: i64)
      -> tensor<1x4xi64> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%mask : tensor<1x4xi1>) outs(%acc : tensor<1x4xi64>) {
      ^bb0(%m: i1, %a: i64):
        %i1 = linalg.index 1 : index
        %i2 = linalg.index 2 : index
        %c = arith.cmpi ule, %i2, %i1 : index
        %e = arith.extui %m : i1 to i64
        %s = arith.select %c, %e, %init : i64
        %add = arith.addi %a, %s : i64
        linalg.yield %add : i64
    } -> tensor<1x4xi64>
    func.return %r : tensor<1x4xi64>
  }
}
"""


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_scalar_arg_kernel_is_passed_by_value(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.dispatch_runtime import execute
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    outlined = outline_dispatches(parse_mlir_text(CUMSUM))
    mask = np.ones((1, 4), np.int8)          # i1 all-ones
    acc = np.zeros((1, 4), np.int64)
    init = np.int64(0)
    (out,) = execute(outlined, [mask, acc, init], tmp_path)
    # cumsum of ones along the causal triangle -> [1, 2, 3, 4]
    assert out.ravel().tolist() == [1, 2, 3, 4]


@pytest.mark.skipif(not (MODEL / "model.mlir").is_file(),
                    reason="small_llama capture not present")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_whole_small_llama_via_dispatch_table_matches_torch(tmp_path):
    from merlin.runtime.dispatch_runtime import run_model

    res = run_model(MODEL, tmp_path)
    assert res["n_kernels"] == 183
    assert res["output"].shape == (1, 8, 256)
    # Same fidelity as the monolithic compile, but through the per-kernel dispatch table.
    assert res["ok"], f"cos={res['cos']} rel={res['rel']}"
    assert res["cos"] > 0.9999
    assert res["rel"] < 1e-3


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="set MERLIN_RUN_SLOW=1 (compiles ~1000 kernels)")
@pytest.mark.skipif(not (TINY / "model.mlir").is_file(),
                    reason="tiny_llama capture not present")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_whole_tiny_llama_via_dispatch_table_matches_torch(tmp_path):
    """TinyLlama-1.1B end to end via the dispatch table; argmax == torch on all tokens."""
    from merlin.runtime.dispatch_runtime import run_model

    cache = REPO / "out/artifacts/cache/kernel_cache_tiny"
    res = run_model(TINY, tmp_path, cache_dir=cache)
    assert res["n_kernels"] == 1402
    out = np.asarray(res["output"], np.float32).reshape(8, -1)
    gold = np.load(TINY / "golden.npy").astype(np.float32).reshape(8, -1)
    assert (out.argmax(1) == gold.argmax(1)).all()       # next-token prediction matches
    assert res["cos"] > 0.999

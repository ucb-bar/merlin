"""Per-kernel backend + checker (``llvmlower.kernel_backend``).

Each outlined kernel is compiled in isolation and gated against the analytic numpy
reference. Auto-skips without the host toolchain (m2m venv / clang-23).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]

CHAIN = """
builtin.module {
  func.func @forward(%w: tensor<8x6xf32>, %x: tensor<4x8xf32>, %z: tensor<6x5xf32>)
      -> tensor<4x5xf32> {
    %e0 = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y0 = linalg.matmul ins(%x, %w : tensor<4x8xf32>, tensor<8x6xf32>)
          outs(%f0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %e1 = tensor.empty() : tensor<4x5xf32>
    %c1 = arith.constant 0.0 : f32
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %y1 = linalg.matmul ins(%y0, %z : tensor<4x6xf32>, tensor<6x5xf32>)
          outs(%f1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    func.return %y1 : tensor<4x5xf32>
  }
}
"""


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


def test_extract_kernel_is_standalone_module():
    """Extraction yields a one-func module, renamed to the entry symbol, no extras."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.kernel_backend import extract_kernel, signature_of
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_text(CHAIN))
    km = extract_kernel(res.module, res.dispatches[0].symbol)
    funcs = [op for op in km.walk() if op.name == "func.func"]
    assert len(funcs) == 1
    assert funcs[0].sym_name.data == "forward"
    sig = signature_of(funcs[0])
    assert len(sig.in_shapes) == 2 and len(sig.out_shapes) == 1


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_each_kernel_compiles_and_matches_numpy(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.kernel_backend import check_matmul_kernels
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_text(CHAIN))
    checks = check_matmul_kernels(res, tmp_path)
    assert len(checks) == 2
    assert all(c.ok for c in checks), [(c.symbol, c.max_abs) for c in checks]


@pytest.mark.skipif(not (REPO / "artifacts/recaptures/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_real_small_llama_matmul_kernels_all_pass(tmp_path):
    """Every contraction dispatch of the real model compiles and is numerically correct."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.kernel_backend import check_matmul_kernels
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_file(REPO / "artifacts/recaptures/small_consistent/model.mlir"))
    checks = check_matmul_kernels(res, tmp_path)
    assert len(checks) == 15
    failures = [(c.symbol, c.shapes, c.max_abs) for c in checks if not c.ok]
    assert not failures, failures

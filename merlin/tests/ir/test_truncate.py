"""The subgraph-truncation bisection tool (seed of the per-dispatch outliner)."""
from __future__ import annotations

from merlin.llvmlower.truncate import multi_return, tensor_defs, truncate_to

# Single-line ops, matching model2MLIR's printer (tensor_defs is line-based).
MOD = """
builtin.module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %e = tensor.empty() : tensor<4x4xf32>
    %c0 = arith.constant 0.0 : f32
    %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %y = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
    %t = tensor.empty() : tensor<4x4xf32>
    %z = linalg.transpose ins(%y : tensor<4x4xf32>) outs(%t : tensor<4x4xf32>) permutation = [1, 0]
    func.return %z : tensor<4x4xf32>
  }
}
"""


def test_tensor_defs_covers_all_op_forms():
    defs = tensor_defs(MOD)
    results = {d.result for d in defs}
    # matmul, fill, empties, AND the transpose (which doesn't end in `-> tensor`)
    assert "%y" in results
    assert "%z" in results          # transpose captured by the broadened matcher
    by = {d.result: d for d in defs}
    assert by["%y"].type == "tensor<4x4xf32>"
    assert by["%z"].type == "tensor<4x4xf32>"


def test_truncate_to_rewrites_return():
    defs = {d.result: d for d in tensor_defs(MOD)}
    out = truncate_to(MOD, defs["%y"])
    assert "func.return %y : tensor<4x4xf32>" in out
    assert "linalg.transpose" not in out      # dropped everything after the target
    assert "-> tensor<4x4xf32>" in out         # function result type patched


def test_multi_return_returns_tuple():
    defs = {d.result: d for d in tensor_defs(MOD)}
    out = multi_return(MOD, [defs["%y"], defs["%z"]])
    assert "func.return %y, %z :" in out
    assert "-> (tensor<4x4xf32>, tensor<4x4xf32>)" in out

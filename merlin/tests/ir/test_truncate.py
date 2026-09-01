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


# A tensor type whose element type is itself parameterized (`!quant.uniform<…>`) or that carries
# an affine-map encoding. The old `tensor<[^<>]*(<[^<>]*>[^<>]*)*>` pattern mis-read both: it cut
# the type at an inner `>` and left a stray `>` behind in the rewritten signature, i.e. a WRONG
# bisection result that still looked plausible. The balanced-bracket scan reads them whole.
NESTED = """
builtin.module {
  func.func @forward(%a: tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 0.1>> {
    %y = tensor.empty() : tensor<4x!quant.uniform<i8:f32, 0.1>>
    func.return %y : tensor<4x!quant.uniform<i8:f32, 0.1>>
  }
}
"""

ENCODED = """
builtin.module {
  func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32, affine_map<(d0) -> (d0)>> {
    %y = tensor.empty() : tensor<4xf32, affine_map<(d0) -> (d0)>>
    func.return %y : tensor<4xf32, affine_map<(d0) -> (d0)>>
  }
}
"""


def test_parameterized_element_type_is_read_whole():
    (d,) = tensor_defs(NESTED)
    assert d.type == "tensor<4x!quant.uniform<i8:f32, 0.1>>"
    out = truncate_to(NESTED, d)
    assert "-> tensor<4x!quant.uniform<i8:f32, 0.1>> {" in out
    assert ">>>" not in out                       # no stray bracket left behind


def test_affine_map_encoding_is_not_cut_at_the_arrows_bracket():
    """`affine_map<(d0) -> (d0)>` contains a `>` that closes an ARROW, not a bracket."""
    (d,) = tensor_defs(ENCODED)
    assert d.type == "tensor<4xf32, affine_map<(d0) -> (d0)>>"
    assert "-> tensor<4xf32, affine_map<(d0) -> (d0)>> {" in truncate_to(ENCODED, d)
    assert "-> (tensor<4xf32, affine_map<(d0) -> (d0)>>) {" in multi_return(ENCODED, [d])


def test_signature_without_a_result_arrow_is_left_alone():
    mod = """
builtin.module {
  func.func @forward(%a: tensor<4xf32>) {
    %y = tensor.empty() : tensor<4xf32>
    func.return
  }
}
"""
    (d,) = tensor_defs(mod)
    assert "func.func @forward(%a: tensor<4xf32>) {" in truncate_to(mod, d)


def test_already_tupled_signature_is_retupled_not_nested():
    mod = """
builtin.module {
  func.func @forward(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {x} {
    %y = tensor.empty() : tensor<4xf32>
    %z = tensor.empty() : tensor<4xf32>
    func.return %y, %z : tensor<4xf32>, tensor<4xf32>
  }
}
"""
    defs = tensor_defs(mod)
    out = multi_return(mod, defs)
    assert "-> (tensor<4xf32>, tensor<4xf32>) attributes {x} {" in out
    assert "-> ((" not in out

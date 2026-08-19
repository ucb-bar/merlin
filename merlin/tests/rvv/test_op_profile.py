"""Unit tests for the whole-model per-op profiler instrumentation (merlin.llvmlower.op_profile).

Board-free: exercises the pure-text IR instrumentation (op-boundary detection, marker splicing,
multi-line-op handling, the prov.* join key, and console parsing) against small MLIR fixtures.
The end-to-end board run is driven by build_tools/scripts/k1_op_profile.py.
"""
from __future__ import annotations

from merlin.llvmlower import op_profile as opf

# A @forward with: an elementwise generic (region opens on the SAME line, closes with `->`),
# a linalg.reduce (reduction region opens on the NEXT line — the case that broke naive
# depth-tracking), and a plain matmul. Mirrors the shape of the real captures.
_MLIR = """\
module {
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<4x4xf32>) attrs = {prov.op = "add", prov.family = "elementwise", prov.region_id = "add_0"} {
    ^bb0(%a: f32):
      linalg.yield %a : f32
    } -> tensor<4x4xf32>
    %2 = tensor.empty() : tensor<4xf32>
    %3 = linalg.reduce ins(%1 : tensor<4x4xf32>) outs(%2 : tensor<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %r = arith.addf %in, %init : f32
      linalg.yield %r : f32
    }
    %4 = linalg.matmul {prov.op = "mm", prov.family = "contraction", prov.region_id = "matmul_0", prov.fqn = "layers.0.mlp"} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %4 : tensor<4x4xf32>
  }
}
"""


def test_find_forward_ops_counts_top_level_only():
    _, _, ops = opf.find_forward_ops(_MLIR)
    # 5 top-level SSA ops: empty, generic, empty, reduce, matmul. Region bodies / bb args / the
    # reduce's `(%in,%init){` continuation must NOT be counted.
    names = [o["mlir_op"] for o in ops]
    assert names == ["tensor.empty", "linalg.generic", "tensor.empty",
                     "linalg.reduce", "linalg.matmul"], names


def test_reduce_continuation_not_marked():
    # The reduce's reduction region `(%in: f32, %init: f32) {` opens on its own depth-0 line;
    # it must not be mistaken for a new op (that would splice a marker mid-op and break the IR).
    _, _, ops = opf.find_forward_ops(_MLIR)
    assert all(not o["mlir_op"].startswith("(") for o in ops)


def test_prov_and_join_key():
    _, _, ops = opf.find_forward_ops(_MLIR)
    mm = ops[-1]
    assert mm["family"] == "contraction"
    assert mm["fqn"] == "layers.0.mlp"
    # fqn wins as the join key when present; region_id is the fallback.
    assert opf.join_key(mm) == "layers.0.mlp"
    gen = ops[1]
    assert gen["fqn"] is None
    assert opf.join_key(gen) == "add_0"


def test_instrument_marker_count_and_declaration():
    text, table = opf.instrument(_MLIR)
    # one mark per op + one sentinel before func.return.
    assert text.count(f"call @{opf.MARK_SYM}") == len(table) + 1
    # the hook is declared exactly once, as a private func.
    assert text.count(f"func.func private @{opf.MARK_SYM}(i32) -> ()") == 1
    # table has no source-line bookkeeping leaking out, and ids are contiguous.
    assert [r["id"] for r in table] == list(range(len(table)))
    assert all("line" not in r for r in table)


def test_instrument_preserves_reduce_body():
    text, _ = opf.instrument(_MLIR)
    # the reduce's region must be intact (no marker spliced between its head and its region).
    assert "dimensions = [1]\n" in text
    idx = text.index("dimensions = [1]")
    tail = text[idx:idx + 120]
    assert "(%in: f32, %init: f32) {" in tail  # region immediately follows, uninterrupted


def test_result_type_and_elem_count():
    _, _, ops = opf.find_forward_ops(_MLIR)
    mm = ops[-1]
    assert mm["result_type"] == "tensor<4x4xf32>"
    assert mm["elems"] == 16


def test_parse_prof_lines():
    console = "noise\nPROF 0 1234 1\nPROF 3 99 2\nPROF -1 5 1\nMETRIC prof_marks 6\n"
    got = opf.parse_prof_lines(console)
    assert got[0] == (1234, 1)
    assert got[3] == (99, 2)
    assert got[-1] == (5, 1)


def test_no_forward_raises():
    import pytest
    with pytest.raises(opf.OpProfileError):
        opf.find_forward_ops("module { func.func @other() { func.return } }")


# ---------------------------------------------------------------------------------------------------
# Which call is it?
#
# The table used to record every top-level call as a bare `mlir_op: "call"`, so a routed matrix-unit
# entry point was indistinguishable from any other call. Attributing the routed region then meant
# counting call rows and trusting that the count matched what the router reported -- inference that
# breaks silently as soon as @forward contains a second kind of call, and that cannot separate two
# routed signatures from each other at all.
# ---------------------------------------------------------------------------------------------------


def test_a_call_row_records_the_symbol_it_invokes():
    mlir = """
func.func @forward(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = call @merlin_opu_gemm_i8_1(%arg0) : (tensor<4x4xi8>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
"""
    _, _, ops = opf.find_forward_ops(mlir)
    assert len(ops) == 1
    assert ops[0]["callee"] == "@merlin_opu_gemm_i8_1"


def test_two_routed_entry_points_are_told_apart():
    """Counting rows cannot do this; the symbol can."""
    mlir = """
func.func @forward(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = call @merlin_opu_gemm_i8_1(%arg0) : (tensor<4x4xi8>) -> tensor<4x4xi32>
  %1 = call @merlin_opu_gemm_i8_6(%arg0) : (tensor<4x4xi8>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}
"""
    _, _, ops = opf.find_forward_ops(mlir)
    assert [o["callee"] for o in ops] == ["@merlin_opu_gemm_i8_1", "@merlin_opu_gemm_i8_6"]


def test_a_non_call_op_has_no_callee():
    mlir = """
func.func @forward(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.fill ins(%arg0 : tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
"""
    _, _, ops = opf.find_forward_ops(mlir)
    assert ops[0]["callee"] is None


def test_the_qualified_spelling_is_accepted_too():
    """`call` and `func.call` both occur depending on who last round-tripped the module."""
    mlir = """
func.func @forward(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = func.call @merlin_opu_gemm_i8_2(%arg0) : (tensor<4x4xi8>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
"""
    _, _, ops = opf.find_forward_ops(mlir)
    assert ops[0]["callee"] == "@merlin_opu_gemm_i8_2"

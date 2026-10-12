"""Selected production graph rewrites preserve logical tensors and explicit contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir.affine import AffineExpr, AffineMap

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.graph_optimizations import GraphOptimizationContract, optimize_host_graph


def _source(rows=2, cols=3):
    return f"""builtin.module {{
 func.func @kernel(%x:tensor<{rows}x{cols}xi32>, %bias:tensor<{cols}xi32>) -> tensor<{rows}x{cols}xi32> {{
  %e = tensor.empty() : tensor<{cols}x{rows}xi32>
  %t = linalg.transpose ins(%x:tensor<{rows}x{cols}xi32>) outs(%e:tensor<{cols}x{rows}xi32>) permutation = [1,0]
  %o = tensor.empty() : tensor<{cols}x{rows}xi32>
  %a = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
      affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}}
      ins(%t,%bias:tensor<{cols}x{rows}xi32>,tensor<{cols}xi32>) outs(%o:tensor<{cols}x{rows}xi32>) {{
  ^bb0(%v:i32,%b:i32,%init:i32):
    %sum = arith.addi %v,%b : i32
    linalg.yield %sum : i32
  }} -> tensor<{cols}x{rows}xi32>
  %e2 = tensor.empty() : tensor<{rows}x{cols}xi32>
  %r = linalg.transpose ins(%a:tensor<{cols}x{rows}xi32>) outs(%e2:tensor<{rows}x{cols}xi32>) permutation = [1,0]
  func.return %r : tensor<{rows}x{cols}xi32>
 }}
}}"""


def _contract(source, **values):
    return GraphOptimizationContract(hashlib.sha256(source.encode()).hexdigest(), "kernel", **values)


def _layout(source, **values):
    return _contract(source, propagate_layout=True, independent_element_order_unobserved=True, **values)


def _execute(source, matrix, bias):
    interpreter = Interpreter(parse_mlir_text(source))
    for implementation in (ArithFunctions(), FuncFunctions(), LinalgFunctions(), TensorFunctions()):
        interpreter.register_implementations(implementation)
    arguments = (
        ShapedArray(TypedPtr.new(matrix.reshape(-1).tolist(), xtype=i32), list(matrix.shape)),
        ShapedArray(TypedPtr.new(bias.tolist(), xtype=i32), list(bias.shape)),
    )
    return tuple(
        np.asarray(value.data, dtype=np.int32).reshape(value.shape)
        for value in interpreter.call_op("kernel", arguments)
    )


@pytest.mark.parametrize("shape", [(2, 3), (3, 5), (7, 2)])
def test_layout_real_execution_preserves_broadcast_and_wrapping(shape):
    source = _source(*shape)
    optimized, report = optimize_host_graph(source, _layout(source))
    matrix = np.resize(np.array([-(2**31), 2**31 - 1, -1, 0, 1], dtype=np.int32), shape)
    bias = np.resize(np.array([1, -1, 2**31 - 1], dtype=np.int32), shape[1])
    expected = matrix + bias
    np.testing.assert_array_equal(_execute(source, matrix, bias)[0], expected)
    np.testing.assert_array_equal(_execute(optimized, matrix, bias)[0], expected)
    assert report["layout"]["before_transpose_bytes"] > report["layout"]["after_transpose_bytes"]
    assert report["semantics"] == "conditional_structural_checks"
    assert report["timing"] == "UNKNOWN" and report["hardware_conformance"] == "not_established"
    assert report["transformed_sha256"] == hashlib.sha256(optimized.encode()).hexdigest()


def test_shared_producer_and_multiple_ordered_outputs_are_checked_and_executed():
    source = _source().replace("-> tensor<2x3xi32> {", "-> (tensor<2x3xi32>,tensor<3x2xi32>) {")
    source = source.replace("func.return %r : tensor<2x3xi32>", "func.return %r,%a : tensor<2x3xi32>,tensor<3x2xi32>")
    optimized, _ = optimize_host_graph(source, _layout(source))
    matrix, bias = np.arange(6, dtype=np.int32).reshape(2, 3), np.array([5, 7, 11], dtype=np.int32)
    for actual, expected in zip(_execute(optimized, matrix, bias), _execute(source, matrix, bias), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_large_shapes_do_not_enumerate_elements():
    source = _source(1000003, 1000033)
    optimized, report = optimize_host_graph(source, _layout(source))
    assert len(optimized) < 10000
    assert report["layout"]["before_transpose_bytes"] == 2 * 1000003 * 1000033 * 4


def test_fusion_is_actual_logical_host_ir_and_executes_exactly():
    source = _source()
    optimized, report = optimize_host_graph(source, _contract(source, fusion_groups=(tuple(range(6)),)))
    matrix, bias = np.arange(6, dtype=np.int32).reshape(2, 3), np.array([5, 7, 11], dtype=np.int32)
    np.testing.assert_array_equal(_execute(optimized, matrix, bias)[0], matrix + bias)
    assert report["fusion"]["before_dispatches"] == 3
    assert report["fusion"]["after_dispatches"] == 1
    assert report["fusion"]["proof"]["computation"] == "expanded_driver_structurally_equivalent"
    assert "merlin.placement" in optimized and "kernel$kernel_fused_0" in optimized


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("source_sha256", "0" * 64, "actual input"),
        ("entry", "missing", "entry"),
        ("independent_element_order_unobserved", False, "order permission"),
        ("minimum_transpose_bytes_saved", 10000, "cost contract"),
        ("maximum_added_operations", False, "positive integers"),
    ],
)
def test_invalid_source_permissions_and_costs_refuse(field, value, message):
    source = _source()
    with pytest.raises(ValueError, match=message):
        optimize_host_graph(source, replace(_layout(source), **{field: value}))


def test_floating_layout_requires_nontrapping_environment_declaration():
    source = _source().replace("i32", "f32").replace("arith.addi", "arith.addf")
    with pytest.raises(ValueError, match="nontrapping/unobserved"):
        optimize_host_graph(source, _layout(source))
    optimized, _ = optimize_host_graph(source, _layout(source, floating_environment_unobserved_nontrapping=True))
    assert "arith.addf" in optimized


@pytest.mark.parametrize(
    "groups,message",
    [
        (((0, 2),), "consecutive"),
        (((0, 1), (1, 2)), "overlap"),
        (((0, 1),), "dispatch-count"),
    ],
)
def test_fusion_selection_or_insufficient_savings_refuses(groups, message):
    source = _source()
    with pytest.raises(ValueError, match=message):
        optimize_host_graph(source, _contract(source, fusion_groups=groups))


def test_fusion_explicit_code_growth_limit_refuses():
    source = _source()
    with pytest.raises(ValueError, match="compilation-space"):
        optimize_host_graph(source, _contract(source, fusion_groups=(tuple(range(6)),), maximum_added_operations=1))


def test_code_growth_limit_is_checked_before_fused_body_cloning(monkeypatch):
    from merlin.xdsl_dialects.lowering import outlined_plan_emission

    def unexpected(*args):
        pytest.fail("oversized fusion reached the cloning emitter")

    monkeypatch.setattr(outlined_plan_emission.OutlinedGlobalPlanEmitter, "emit_global_plan", unexpected)
    source = _source()
    with pytest.raises(ValueError, match="compilation-space"):
        optimize_host_graph(source, _contract(source, fusion_groups=(tuple(range(6)),), maximum_added_operations=1))


def test_corrupt_outlining_is_rejected_against_actual_source(monkeypatch):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    from merlin.xdsl_dialects.lowering import outline

    real = outline.outline_dispatches

    def corrupt(module, **kwargs):
        outlined = real(module, **kwargs)
        operation = next(op for op in outlined.module.walk() if type(op) is arith.AddiOp)
        Rewriter.replace_op(operation, arith.SubiOp(*operation.operands))
        return outlined

    monkeypatch.setattr(outline, "outline_dispatches", corrupt)
    source = _source()
    with pytest.raises(ValueError, match="unaccounted scalar arithmetic"):
        optimize_host_graph(source, _contract(source, fusion_groups=(tuple(range(6)),)))


def test_corrupt_layout_index_map_is_rejected_independently(monkeypatch):
    from merlin.llvmlower import layout_propagation

    real = layout_propagation.rewrite_module

    def corrupt(module, **kwargs):
        report = real(module, **kwargs)
        result = next(op for op in module.walk() if op.name == "func.return").operands[0].owner
        maps = list(result.indexing_maps)
        maps[1] = AffineMapAttr(AffineMap(2, 0, (AffineExpr.dimension(0),)))
        result.properties["indexing_maps"] = ArrayAttr(maps)
        return report

    source = _source(3, 3)  # Both broadcast choices verify structurally, but compute different coordinates.
    monkeypatch.setattr(layout_propagation, "rewrite_module", corrupt)
    with pytest.raises(ValueError, match="logical output coordinates"):
        optimize_host_graph(source, _layout(source))


def test_opaque_observer_of_intermediate_is_preserved():
    source = _source().replace(
        "  %e2 = tensor.empty()", '  "opaque.observe"(%a) : (tensor<3x2xi32>) -> ()\n  %e2 = tensor.empty()'
    )
    optimized, _ = optimize_host_graph(source, _layout(source))
    assert "opaque.observe" in optimized


def test_unused_opaque_scalar_effect_refuses():
    source = _source().replace("    %sum = arith.addi", '    "opaque.observe"(%v) : (i32) -> ()\n    %sum = arith.addi')
    with pytest.raises(ValueError, match="scalar effects"):
        optimize_host_graph(source, _layout(source))


def test_uninitialized_tensor_value_is_not_assumed_zero():
    source = _source().replace("%v,%b : i32", "%init,%b : i32")
    with pytest.raises(ValueError, match="no semantics for this tensor producer"):
        optimize_host_graph(source, _layout(source))


def test_actual_lowering_route_invokes_selected_graph_hook_and_preserves_default(tmp_path, monkeypatch):
    from merlin.llvmlower import lower

    seen = []

    def llvm_boundary(upstream_text, **kwargs):
        seen.append(upstream_text)
        return "define void @kernel() { ret void }\n"

    monkeypatch.setattr(lower, "lower_to_llvm_ir", llvm_boundary)
    monkeypatch.setenv("MERLIN_STATIC_ARENA", "")
    source = _source()
    baseline = lower.lower_model(source, tmp_path / "baseline", targets=())
    selected = lower.lower_model(source, tmp_path / "selected", targets=(), graph_optimizations=_layout(source))
    assert "graph_optimizations" not in baseline.stats
    assert selected.stats["graph_optimizations"]["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert seen[0] != seen[1] and selected.ll_path.read_text() == baseline.ll_path.read_text()
    # The file API must forward the exact selection rather than bypassing the production hook.
    path = tmp_path / "logical.mlir"
    path.write_text(source)
    selected_file = lower.lower_model_file(
        path, tmp_path / "selected-file", targets=(), graph_optimizations=_layout(source)
    )
    assert selected_file.stats["graph_optimizations"] == selected.stats["graph_optimizations"]

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


def _execute(source, matrix, bias, *additional):
    interpreter = Interpreter(parse_mlir_text(source))
    for implementation in (ArithFunctions(), FuncFunctions(), LinalgFunctions(), TensorFunctions()):
        interpreter.register_implementations(implementation)
    arguments = (
        ShapedArray(TypedPtr.new(value.reshape(-1).tolist(), xtype=i32), list(value.shape))
        for value in (matrix, bias, *additional)
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
    # Read the shared producer in its existing layout while directly writing
    # the next pointwise result in the return layout. Do not recompute %a.
    source = source.replace("  %e2 = tensor.empty()", """  %next_empty = tensor.empty() : tensor<3x2xi32>
  %next = linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
   affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
   ins(%a:tensor<3x2xi32>) outs(%next_empty:tensor<3x2xi32>) {
   ^bb0(%value:i32,%init:i32):
    %two = arith.constant 2 : i32
    %sum = arith.addi %value,%two : i32
    linalg.yield %sum : i32
  } -> tensor<3x2xi32>
  %e2 = tensor.empty()""")
    source = source.replace("ins(%a:tensor<3x2xi32>) outs(%e2", "ins(%next:tensor<3x2xi32>) outs(%e2")
    optimized, report = optimize_host_graph(source, _layout(source))
    assert report["layout"]["pointwise_work"]["before_scalar_evaluations"] == 18
    assert report["layout"]["pointwise_work"]["after_scalar_evaluations"] == 18
    matrix, bias = np.arange(6, dtype=np.int32).reshape(2, 3), np.array([5, 7, 11], dtype=np.int32)
    expected_outputs = ((matrix + bias) + np.int32(2), (matrix + bias).T)
    for actual, expected in zip(_execute(optimized, matrix, bias), expected_outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(_execute(source, matrix, bias), expected_outputs, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_shared_producer_noninvolutive_permutation_preserves_broadcast_and_work():
    source = """builtin.module {
 func.func @kernel(%x:tensor<2x3x4xi32>, %column:tensor<4xi32>, %row:tensor<1x3xi32>)
     -> (tensor<3x4x2xi32>,tensor<2x3x4xi32>) {
  %e = tensor.empty() : tensor<2x3x4xi32>
  %shared = linalg.generic {indexing_maps=[affine_map<(d0,d1,d2)->(d0,d1,d2)>,
   affine_map<(d0,d1,d2)->(d2)>,affine_map<(d0,d1,d2)->(d0,d1,d2)>],
   iterator_types=["parallel","parallel","parallel"]}
   ins(%x,%column:tensor<2x3x4xi32>,tensor<4xi32>) outs(%e:tensor<2x3x4xi32>) {
   ^bb0(%value:i32,%bias:i32,%init:i32):
    %sum = arith.addi %value,%bias : i32
    linalg.yield %sum : i32
  } -> tensor<2x3x4xi32>
  %o = tensor.empty() : tensor<2x3x4xi32>
  %next = linalg.generic {indexing_maps=[affine_map<(d0,d1,d2)->(d0,d1,d2)>,
   affine_map<(d0,d1,d2)->(0,d1)>,affine_map<(d0,d1,d2)->(d0,d1,d2)>],
   iterator_types=["parallel","parallel","parallel"]}
   ins(%shared,%row:tensor<2x3x4xi32>,tensor<1x3xi32>) outs(%o:tensor<2x3x4xi32>) {
   ^bb0(%value:i32,%bias:i32,%init:i32):
    %sum = arith.addi %value,%bias : i32
    linalg.yield %sum : i32
  } -> tensor<2x3x4xi32>
  %t = tensor.empty() : tensor<3x4x2xi32>
  %result = linalg.transpose ins(%next:tensor<2x3x4xi32>) outs(%t:tensor<3x4x2xi32>) permutation=[1,2,0]
  func.return %result,%shared : tensor<3x4x2xi32>,tensor<2x3x4xi32>
 }
}"""
    optimized, report = optimize_host_graph(source, _layout(source))
    assert report["layout"]["before_transpose_bytes"] == 96
    assert report["layout"]["after_transpose_bytes"] == 0
    assert report["layout"]["pointwise_work"]["before_scalar_evaluations"] == 48
    assert report["layout"]["pointwise_work"]["after_scalar_evaluations"] == 48
    generics = [op for op in parse_mlir_text(optimized).walk() if op.name == "linalg.generic"]
    assert len(generics) == 2
    assert tuple(generics[0].results[0].type.get_shape()) == (2, 3, 4)
    assert generics[1].indexing_maps.data[0].data == AffineMap(
        3, 0, (AffineExpr.dimension(2), AffineExpr.dimension(0), AffineExpr.dimension(1))
    )
    matrix = np.resize(np.array([-(2**31), 2**31 - 1, -1, 0, 1], dtype=np.int32), (2, 3, 4))
    column = np.array([1, -1, 2**31 - 1, 3], dtype=np.int32)
    row = np.array([[2, -(2**31), -3]], dtype=np.int32)
    shared = matrix + column
    expected = ((shared + row.reshape(1, 3, 1)).transpose(1, 2, 0), shared)
    for program in (source, optimized):
        for actual, reference in zip(_execute(program, matrix, column, row), expected, strict=True):
            np.testing.assert_array_equal(actual, reference)


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
    # Unknown observers may mutate aliases: moving a fresh input read past them
    # is not justified. The explicit selection diagnoses no legal savings.
    with pytest.raises(ValueError, match="materialization-byte cost contract"):
        optimize_host_graph(source, _layout(source))
    baseline, _ = optimize_host_graph(source, _contract(source))
    assert "opaque.observe" in baseline


def test_unused_opaque_scalar_effect_refuses():
    source = _source().replace("    %sum = arith.addi", '    "opaque.observe"(%v) : (i32) -> ()\n    %sum = arith.addi')
    with pytest.raises(ValueError, match="materialization-byte cost contract"):
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
    # This hook-only test stubs the backend that emits copy-removal reports.
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_COPIES", "0")
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


def _mixed_source():
    source = _source(3, 3).replace("i32", "f32").replace("arith.addi", "arith.addf")
    start, end = source.index("  %e ="), source.index("  func.return")
    body = source[start:end].replace("ins(%x:", "ins(%produced:")
    body = body.replace("linalg.yield %sum", "%shifted = arith.addf %sum,%zero : f32\n    linalg.yield %shifted")
    return source[:start] + """
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %loop = scf.for %i = %c0 to %c3 step %c1 iter_args(%current=%x) -> (tensor<3x3xf32>) {
   %produced = func.call @producer(%current) : (tensor<3x3xf32>) -> tensor<3x3xf32>
""" + body + """
   scf.yield %r : tensor<3x3xf32>
  }
  func.return %loop : tensor<3x3xf32>
 }
 func.func private @producer(tensor<3x3xf32>) -> tensor<3x3xf32>
}"""


def test_mixed_nested_calls_captures_and_declarations_are_checked():
    source = _mixed_source()
    optimized, report = optimize_host_graph(source, _layout(
        source, floating_environment_unobserved_nontrapping=True))
    assert "scf.for" in optimized and "producer" in optimized
    assert report["layout"]["after_transpose_bytes"] == 0
    passes = report["layout"]["passes"]
    assert [row["name"] for row in passes] == ["layout-source", "layout-propagation", "layout-conformance"]
    assert passes[1]["invalidated"] == ("coordinates",)
    assert "layout_conformance" in report["layout"]["library_sources_sha256"]


@pytest.mark.parametrize("mutation", ["capture", "opaque_operand", "loop_yield", "opaque_metadata"])
def test_mixed_layout_corruption_is_rejected_independently(monkeypatch, mutation):
    from xdsl.dialects import arith, builtin

    from merlin.llvmlower import layout_propagation

    real = layout_propagation.rewrite_module

    def corrupt(module, **kwargs):
        report = real(module, **kwargs)
        if mutation == "capture":
            constant = next(op for op in module.walk()
                            if type(op) is arith.ConstantOp and op.results[0].type == builtin.f32)
            constant.properties["value"] = builtin.FloatAttr(-0.0, builtin.f32)
        elif mutation == "opaque_operand":
            call = next(op for op in module.walk() if op.name == "func.call")
            call.operands = [next(op for op in module.walk() if op.name == "func.func").body.block.args[0]]
        elif mutation == "loop_yield":
            yield_op = next(op for op in module.walk() if op.name == "scf.yield")
            yield_op.operands = [yield_op.parent.args[1]]
        else:
            call = next(op for op in module.walk() if op.name == "func.call")
            call.attributes["unknown.effect"] = builtin.UnitAttr()
        return report

    monkeypatch.setattr(layout_propagation, "rewrite_module", corrupt)
    source = _mixed_source()
    with pytest.raises(ValueError, match="unaccounted scalar|logical output coordinates"):
        optimize_host_graph(source, _layout(source, floating_environment_unobserved_nontrapping=True))


@pytest.mark.parametrize("operation", ["linalg.transpose", "linalg.generic"])
def test_unknown_executable_metadata_is_a_barrier(operation):
    from xdsl.dialects import builtin

    from merlin.transforms.certificates import ir_text

    module = parse_mlir_text(_source())
    next(op for op in module.walk() if op.name == operation).attributes["unknown.execution"] = builtin.UnitAttr()
    source = ir_text(module)
    with pytest.raises(ValueError, match="materialization-byte cost contract"):
        optimize_host_graph(source, _layout(source))


def test_layout_growth_is_bounded_before_scalar_region_expansion():
    source = _mixed_source()
    with pytest.raises(ValueError, match="compilation-space"):
        optimize_host_graph(source, _layout(source, maximum_added_operations=1,
                                           floating_environment_unobserved_nontrapping=True))


def test_independent_checker_detects_read_moved_after_opaque_mutation():
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    # A materialized copy is a snapshot; its observation follows the unknown
    # mutator. Replacing it with a later read of %x must fail even if coordinates match.
    source = _source()
    source = source.replace("  func.return %r", '  "opaque.mutate"(%x) : (tensor<2x3xi32>) -> ()\n  func.return %r')
    original = parse_mlir_text(source)
    modified = original.clone()
    generic = next(op for op in modified.walk() if op.name == "linalg.generic")
    # Deliberately move the computation and its input transpose past the effect.
    from merlin.common.mlir_query import op_name

    mutator = next(op for op in modified.walk() if op_name(op) == "opaque.mutate")
    before = next(op for op in modified.walk() if op.name == "linalg.transpose")
    block = generic.parent
    for op in (before, generic):
        block.detach_op(op)
        block.insert_op_after(op, mutator)
        mutator = op
    # Move the final transpose as well, preserving SSA dominance and output type.
    final = next(op for op in modified.walk() if op.name == "linalg.transpose" and op is not before)
    block.detach_op(final)
    block.insert_op_after(final, generic)
    modified.verify()
    with pytest.raises(ValueError, match="logical output coordinates"):
        _check_layout_output(original, modified, _layout_bodies(original))


@pytest.mark.parametrize("mutation", ["input", "selected", "implementation", "contract", "missing"])
def test_saved_graph_artifact_custody_rejects_changed_bytes(tmp_path, mutation):
    import json

    from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts

    source = _source()
    contract = _layout(source)
    selected, report = optimize_host_graph(source, contract)
    input_path, selected_path, report_path = (
        tmp_path / name for name in (
            "host-graph-input.mlir", "host-graph-optimized.mlir", "host-graph-optimization.json")
    )
    input_path.write_text(source)
    selected_path.write_text(selected)
    report_path.write_text(json.dumps(report))
    assert set(recheck_host_graph_artifacts(tmp_path, contract)) == {"input", "selected", "report"}
    if mutation == "input":
        input_path.write_text(source + "\n")
    elif mutation == "selected":
        selected_path.write_text(selected + "\n")
    elif mutation == "implementation":
        report["implementation_sources_sha256"]["layout_conformance"] = "0" * 64
        report_path.write_text(json.dumps(report))
    elif mutation == "contract":
        contract = replace(contract, independent_element_order_unobserved=False)
    else:
        report_path.unlink()
    with pytest.raises(ValueError, match="input source|changed|absent"):
        recheck_host_graph_artifacts(tmp_path, contract)


def test_same_typed_opaque_results_have_distinct_checked_identities():
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    source = _mixed_source().replace(
        "  %e = tensor.empty()", "  %other = func.call @producer(%current) : (tensor<3x3xf32>) -> tensor<3x3xf32>\n"
        "  %e = tensor.empty()", 1)
    original = parse_mlir_text(source)
    modified = original.clone()
    calls = [op for op in modified.walk() if op.name == "func.call"]
    transpose = next(op for op in modified.walk() if op.name == "linalg.transpose")
    transpose.operands = [calls[1].results[0], transpose.operands[1]]
    modified.verify()
    with pytest.raises(ValueError, match="logical output coordinates"):
        _check_layout_output(original, modified, _layout_bodies(original))


def test_alias_cancellation_retains_snapshot_across_later_effect():
    from merlin.llvmlower.layout_propagation import rewrite_module

    source = """module {
 func.func @kernel(%x:tensor<2x3xi32>)->tensor<2x3xi32> {
  %e=tensor.empty():tensor<3x2xi32>
  %t=linalg.transpose ins(%x:tensor<2x3xi32>) outs(%e:tensor<3x2xi32>) permutation=[1,0]
  %o=tensor.empty():tensor<2x3xi32>
  %r=linalg.transpose ins(%t:tensor<3x2xi32>) outs(%o:tensor<2x3xi32>) permutation=[1,0]
  "opaque.mutate"(%x):(tensor<2x3xi32>)->()
  func.return %r:tensor<2x3xi32>
 }
}"""
    module = parse_mlir_text(source)
    rewrite_module(module, pointwise_only=True)
    result = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert result.owner.name == "linalg.transpose"
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    original = parse_mlir_text(source)
    _check_layout_output(original, module, _layout_bodies(original))


@pytest.mark.parametrize("dtype,before,after", [
    ("f32", "7f800001", "7fc00001"), ("f64", "7ff0000000000001", "7ff8000000000001"),
])
@pytest.mark.parametrize("location", ["constant", "nested_metadata"])
def test_exact_dense_nan_bits_cannot_be_changed_by_default_attribute_printing(dtype, before, after, location):
    from xdsl.dialects import builtin

    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    source = f"""module {{ func.func @kernel()->tensor<1x{dtype}> {{
     %c=arith.constant dense<0x{before}>:tensor<1x{dtype}>
     func.return %c:tensor<1x{dtype}>
    }} }}"""
    original = parse_mlir_text(source)
    replacement = parse_mlir_text(source.replace(before, after))
    value = next(op for op in replacement.walk() if op.name == "arith.constant").value
    if location == "nested_metadata":
        function = next(op for op in original.walk() if op.name == "func.func")
        constant = next(op for op in original.walk() if op.name == "arith.constant")
        function.attributes["prov.words"] = builtin.ArrayAttr([builtin.DictionaryAttr({"payload": constant.value})])
    modified = original.clone()
    if location == "constant":
        next(op for op in modified.walk() if op.name == "arith.constant").properties["value"] = value
    else:
        function = next(op for op in modified.walk() if op.name == "func.func")
        function.attributes["prov.words"] = builtin.ArrayAttr([builtin.DictionaryAttr({"payload": value})])
    modified.verify()
    with pytest.raises(ValueError, match="function declarations|logical output coordinates"):
        _check_layout_output(original, modified, _layout_bodies(original))


def _annotated_source(attribute="prov.source"):
    from xdsl.dialects import builtin

    from merlin.transforms.certificates import ir_text

    module = parse_mlir_text(_mixed_source())
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    generic.body.block.last_op.attributes[attribute] = builtin.StringAttr("logical-origin")
    return ir_text(module)


def test_provenance_on_scalar_yield_preserves_checked_layout_optimization():
    from xdsl.dialects import builtin

    source = _annotated_source()
    selected, report = optimize_host_graph(source, _layout(
        source, floating_environment_unobserved_nontrapping=True))
    assert report["layout"]["after_transpose_bytes"] == 0
    yields = [op for op in parse_mlir_text(selected).walk() if op.name == "linalg.yield"]
    assert sum(op.attributes.get("prov.source") == builtin.StringAttr("logical-origin") for op in yields) == 1


def test_executable_metadata_on_scalar_yield_remains_a_barrier():
    source = _annotated_source("unknown.execution")
    with pytest.raises(ValueError, match="materialization-byte cost contract"):
        optimize_host_graph(source, _layout(source, floating_environment_unobserved_nontrapping=True))


def test_scalar_yield_provenance_corruption_is_rejected(monkeypatch):
    from xdsl.dialects import builtin

    from merlin.llvmlower import layout_propagation

    real = layout_propagation.rewrite_module

    def corrupt(module, **kwargs):
        report = real(module, **kwargs)
        for op in module.walk():
            if op.name == "linalg.yield" and "prov.source" in op.attributes:
                op.attributes["prov.source"] = builtin.StringAttr("different-origin")
        return report

    monkeypatch.setattr(layout_propagation, "rewrite_module", corrupt)
    source = _annotated_source()
    with pytest.raises(ValueError, match="unaccounted scalar arithmetic"):
        optimize_host_graph(source, _layout(source, floating_environment_unobserved_nontrapping=True))


def test_yield_provenance_cannot_be_swapped_with_another_admitted_body():
    from xdsl.dialects import builtin
    from xdsl.ir import Operation

    from merlin.frontends.linalg_mlir import make_context
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    original = parse_mlir_text(_source(3, 3))
    generic = next(op for op in original.walk() if op.name == "linalg.generic")
    generic.body.block.last_op.attributes["prov.source"] = builtin.StringAttr("first-origin")
    other = generic.clone()
    other.body.block.last_op.attributes["prov.source"] = builtin.StringAttr("second-origin")
    generic.parent.insert_op_after(other, generic)
    observer_class = make_context().get_optional_op("opaque.observe")
    assert observer_class is not None
    observer: Operation = observer_class.create(operands=[other.results[0]])
    generic.parent.insert_op_after(observer, other)
    original.verify()
    modified = original.clone()
    first = next(op for op in modified.walk() if op.name == "linalg.generic")
    first.body.block.last_op.attributes["prov.source"] = builtin.StringAttr("second-origin")
    modified.verify()
    with pytest.raises(ValueError, match="logical output coordinates"):
        _check_layout_output(original, modified, _layout_bodies(original))


def test_independent_work_check_rejects_numerically_identical_shared_recomputation():
    from merlin.llvmlower.graph_optimizations import _check_layout_output, _layout_bodies

    source = _source(3, 3).replace("-> tensor<3x3xi32> {", "-> (tensor<3x3xi32>,tensor<3x3xi32>) {")
    source = source.replace("func.return %r : tensor<3x3xi32>",
                            "func.return %r,%a : tensor<3x3xi32>,tensor<3x3xi32>")
    original = parse_mlir_text(source)
    modified = original.clone()
    generic = next(op for op in modified.walk() if op.name == "linalg.generic")
    duplicate = generic.clone()
    generic.parent.insert_op_after(duplicate, generic)
    final = [op for op in modified.walk() if op.name == "linalg.transpose"][-1]
    final.operands = [duplicate.results[0], final.operands[1]]
    modified.verify()
    with pytest.raises(ValueError, match="increases pointwise arithmetic"):
        _check_layout_output(original, modified, _layout_bodies(original))

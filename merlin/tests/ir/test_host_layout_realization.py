"""Consume selected host layouts; independently reject wrong realized work."""

from dataclasses import replace

import numpy as np
import pytest
from test_host_graph_optimizations import _execute, _layout, _source
from test_host_graph_policy import select
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, TensorType, i64
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir.affine import AffineExpr, AffineMap

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import host_layout_realization
from merlin.llvmlower.graph_optimizations import optimize_host_graph


@pytest.mark.parametrize("shape", [(2, 3), (3, 5), (7, 2)])
def test_ordinary_policy_consumes_graph_layouts_without_cloning_arithmetic(tmp_path, shape):
    source = _source(*shape)
    selected, report = optimize_host_graph(source, select(tmp_path))
    assert report["contract"]["physical_layout"] is True
    assert [row["name"] for row in report["layout"]["passes"]] == [
        "layout-source",
        "layout-planning",
        "layout-propagation",
        "layout-conformance",
    ]
    realization = report["layout"]["rewrite"]
    assert any(row["status"] == "applied" for row in realization["islands"])
    assert realization["rtl_scopes_consumed"] == [] and realization["capacity"].startswith("UNKNOWN")
    assert report["layout"]["after_transpose_bytes"] == 0
    work = report["layout"]["pointwise_work"]
    assert work["before_scalar_evaluations"] == work["after_scalar_evaluations"]
    assert work["physical_layout"]["new_conversions"] == 0
    matrix = np.resize(np.array([-(2**31), 2**31 - 1, -1, 0, 1], dtype=np.int32), shape)
    bias = np.resize(np.array([1, -1, 2**31 - 1], dtype=np.int32), shape[1])
    np.testing.assert_array_equal(_execute(source, matrix, bias)[0], matrix + bias)
    np.testing.assert_array_equal(_execute(selected, matrix, bias)[0], matrix + bias)


def test_shared_producer_publishes_both_canonical_outputs_once(tmp_path):
    source = _source().replace("-> tensor<2x3xi32> {", "-> (tensor<2x3xi32>,tensor<3x2xi32>) {")
    source = source.replace("func.return %r : tensor<2x3xi32>", "func.return %r,%a : tensor<2x3xi32>,tensor<3x2xi32>")
    selected, report = optimize_host_graph(source, select(tmp_path))
    work = report["layout"]["pointwise_work"]
    assert work["before_scalar_evaluations"] == work["after_scalar_evaluations"] == 6
    assert work["physical_layout"]["new_conversions"] == 1
    assert work["physical_layout"]["requested_read_write_bytes"] == 48
    matrix, bias = np.arange(6, dtype=np.int32).reshape(2, 3), np.array([5, 7, 11], dtype=np.int32)
    actual = _execute(selected, matrix, bias)
    np.testing.assert_array_equal(actual[0], matrix + bias)
    np.testing.assert_array_equal(actual[1], (matrix + bias).T)


@pytest.mark.parametrize("choice", [{"maximum_layout_assignments": 1}, {"maximum_layout_permutations": 1}])
def test_exhausted_search_preserves_the_original_ordinary_route(tmp_path, choice):
    source = _source()
    policy = replace(select(tmp_path), **choice)
    selected, report = optimize_host_graph(source, policy)
    assert selected == source and report["automatic_selection"]["status"] == "unchanged"
    assert "layout" not in report


def test_initialized_pointwise_output_keeps_original_source(tmp_path):
    source = _source().replace("%sum = arith.addi %v,%b", "%sum = arith.addi %v,%init")
    # The checker must not treat the destination as fresh uninitialized storage.
    source = source.replace(
        "%o = tensor.empty() : tensor<3x2xi32>",
        "%zero = arith.constant 0 : i32\n  %o = tensor.splat %zero : tensor<3x2xi32>",
    )
    selected, report = optimize_host_graph(source, select(tmp_path))
    assert selected == source and report["automatic_selection"]["status"] == "unchanged"


@pytest.mark.parametrize("mutation", ["wrong_map", "unreported_conversion"])
def test_realized_mutations_cannot_inherit_a_selected_layout(tmp_path, monkeypatch, mutation):
    original = host_layout_realization.rewrite_module

    def corrupt(module, **kwargs):
        result = original(module, **kwargs)
        operation = next(op for op in module.walk() if op.name == "linalg.generic")
        if mutation == "wrong_map":
            properties = list(operation.indexing_maps)
            properties[0] = AffineMapAttr(AffineMap(2, 0, (AffineExpr.dimension(1), AffineExpr.dimension(0))))
            operation.properties["indexing_maps"] = ArrayAttr(properties)
        else:
            value = operation.results[0]
            typ = TensorType(value.type.get_element_type(), value.type.get_shape())
            empty = EmptyOp([], typ)
            extra = TransposeOp(value, empty.results[0], DenseArrayBase.from_list(i64, (1, 0)), typ)
            operation.parent.insert_op_after(empty, operation)
            operation.parent.insert_op_after(extra, empty)
        module.verify()
        return result

    monkeypatch.setattr(host_layout_realization, "rewrite_module", corrupt)
    with pytest.raises(ValueError, match="coordinates|expressions|unreported work"):
        optimize_host_graph(_source(2, 2), select(tmp_path))


def test_selected_algorithm_can_be_disabled_without_new_semantic_permissions(tmp_path):
    source = _source()
    policy = replace(select(tmp_path), physical_layout=False)
    _, report = optimize_host_graph(source, policy)
    assert report["contract"]["physical_layout"] is False
    assert [row["name"] for row in report["layout"]["passes"]] == [
        "layout-source",
        "layout-propagation",
        "layout-conformance",
    ]


def captured_tensor_source():
    return (
        _source()
        .replace(
            "  %o = tensor.empty()",
            "  %stored_bias = arith.constant dense<[[1,2],[3,4],[5,6]]> : tensor<3x2xi32>\n  %o = tensor.empty()",
        )
        .replace("affine_map<(d0,d1)->(d0)>", "affine_map<(d0,d1)->(d0,d1)>")
        .replace("ins(%t,%bias:tensor<3x2xi32>,tensor<3xi32>)", "ins(%t,%stored_bias:tensor<3x2xi32>,tensor<3x2xi32>)")
    )


def test_mid_island_tensor_constant_conversion_follows_its_definition(tmp_path):
    selected, report = optimize_host_graph(captured_tensor_source(), select(tmp_path))
    assert report["layout"]["pointwise_work"]["physical_layout"]["new_conversions"] == 1
    module = parse_mlir_text(selected)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    positions = {op: index for index, op in enumerate(function.body.block.ops)}
    conversion = next(op for op in module.walk() if type(op) is TransposeOp)
    assert positions[conversion.operands[0].owner] < positions[conversion]


def test_conversion_moved_above_definition_is_rejected_by_actual_conformance(tmp_path, monkeypatch):
    original = host_layout_realization.rewrite_module

    def corrupt(module, **kwargs):
        result = original(module, **kwargs)
        conversion = next(op for op in module.walk() if type(op) is TransposeOp)
        constant = conversion.operands[0].owner
        block = constant.parent
        block.detach_op(constant)
        block.insert_op_after(constant, conversion)
        module.verify()  # Actual dominance adds to this structural check.
        return result

    monkeypatch.setattr(host_layout_realization, "rewrite_module", corrupt)
    with pytest.raises(ValueError, match="dominate"):
        optimize_host_graph(captured_tensor_source(), select(tmp_path))


def test_snapshot_before_unknown_effect_keeps_original_legal_route(tmp_path):
    source = """module {func.func @kernel(%x:tensor<2x3x4xi32>)->tensor<2x3x4xi32>{
%e=tensor.empty():tensor<3x4x2xi32>
%a=linalg.transpose ins(%x:tensor<2x3x4xi32>) outs(%e:tensor<3x4x2xi32>) permutation=[1,2,0]
%o=tensor.empty():tensor<2x3x4xi32>
%r=linalg.transpose ins(%a:tensor<3x4x2xi32>) outs(%o:tensor<2x3x4xi32>) permutation=[2,0,1]
"opaque.mutate"(%x):(tensor<2x3x4xi32>)->()
func.return %r:tensor<2x3x4xi32>
}}"""
    selected, report = optimize_host_graph(source, select(tmp_path))
    assert selected == source and report["automatic_selection"]["status"] == "unchanged"

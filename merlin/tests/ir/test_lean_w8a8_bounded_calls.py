"""Infer named integer bounds from actual helper bodies and check every call operand."""

from __future__ import annotations

import pytest
from test_lean_w8a8_bounded import contract, embedding_source
from test_lean_w8a8_bounded_chain import check_chain, lowered, preprocessed
from test_lean_w8a8_calls import with_bridges
from test_lean_w8a8_integer_coordinates import Coordinates
from test_lean_w8a8_native_recipe import dynamic_matmul_reference
from test_lean_w8a8_views import compare_outputs, words

from merlin.frontends.linalg_mlir import parse_mlir_text

pytestmark = pytest.mark.usefixtures("lean_oracle")


def named_contract(name, vocabulary=7):
    return {"function": name, "intervals": contract(vocabulary)[0]["intervals"]}


def called_embedding(depth):
    from xdsl.dialects.builtin import StringAttr

    before = embedding_source()
    kernel = next(op for op in before.walk() if op.name == "func.func")
    kernel.properties["sym_name"] = StringAttr("kernel")
    wrapper = parse_mlir_text("""func.func @forward(%tokens: tensor<2xi64>, %table: tensor<7x3xf32>,
        %weights: tensor<3x4xf32>) -> tensor<2x4xf32> {
      %r = func.call @kernel(%tokens, %table, %weights)
        : (tensor<2xi64>, tensor<7x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      return %r : tensor<2x4xf32>
    }""")
    before.body.block.add_ops(op.clone() for op in wrapper.body.block.ops)
    with_bridges(before, ["tensor<2xi64>", "tensor<7x3xf32>", "tensor<3x4xf32>"], "tensor<2x4xf32>", depth)
    bounds = [named_contract("kernel"), named_contract("forward")]
    bounds += [named_contract(f"bridge_{index}") for index in range(1, depth)]
    return before, bounds


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_bounded_input_contracts_close_over_actual_call_depth(tmp_path, lean_tools, depth, index_bits):
    before, bounds = called_embedding(depth)
    after = lowered(before)
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert len(report["functions"]) == depth + 1
    assert all(fn["source_initialized"] and fn["reference_initialized"] for fn in report["functions"])


@pytest.mark.parametrize("change", ["caller_wider", "callee_narrower", "unknown", "empty"])
def test_callee_name_cannot_supply_unproved_caller_bounds(tmp_path, lean_tools, change):
    before, bounds = called_embedding(3)
    target = next(
        entry for entry in bounds if entry["function"] == ("forward" if change == "caller_wider" else "kernel")
    )
    interval = target["intervals"][0]
    if change == "caller_wider":
        interval["upper"] = 8
    elif change == "callee_narrower":
        interval["upper"] = 3
    elif change == "unknown":
        target["intervals"][0] = None
    else:
        interval["lower"] = 7
    after = lowered(before)
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]
    assert not all(fn["source_initialized"] for fn in report["functions"])


def body_call_source():
    from xdsl.dialects.builtin import i64
    from xdsl.dialects.func import CallOp

    before = embedding_source()
    cast = next(op for op in before.walk() if op.name == "arith.index_cast")
    call = CallOp("coordinate", [cast.operands[0]], [i64])
    cast.parent.insert_op_before(call, cast)
    cast.operands = [call.results[0]]
    helper = parse_mlir_text("""func.func @coordinate(%x: i64) -> i64 { return %x : i64 }""")
    before.body.block.add_ops(op.clone() for op in helper.body.block.ops)
    bounds = contract(7) + [{"function": "coordinate", "intervals": [{"lower": 0, "upper": 7}]}]
    return before, bounds


@pytest.mark.parametrize("index_bits", [32, 64])
def test_calls_inside_checked_loop_bodies_preserve_uniform_bounds(tmp_path, lean_tools, index_bits):
    before, bounds = body_call_source()
    after = lowered(before)
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("kind", ["cycle", "external", "unknown_callee", "unproved_result"])
def test_names_and_signatures_cannot_replace_a_verified_callee_body(tmp_path, lean_tools, kind):
    from xdsl.dialects.builtin import SymbolRefAttr

    before, bounds = body_call_source()
    helper = next(op for op in before.walk() if op.name == "func.func" and op.sym_name.data == "coordinate")
    before.body.block.erase_op(helper)
    bodies = {
        "cycle": """func.func @coordinate(%x: i64) -> i64 {
          %r = func.call @coordinate(%x) : (i64) -> i64
          return %r : i64
        }""",
        "external": "func.func private @coordinate(i64) -> i64",
        "unknown_callee": "func.func private @other(i64) -> i64",
        "unproved_result": """func.func @coordinate(%x: i64) -> i64 {
          %r = arith.constant 127 : i64
          return %r : i64
        }""",
    }
    extra = parse_mlir_text(bodies[kind])
    before.body.block.add_ops(op.clone() for op in extra.body.block.ops)
    if kind == "unknown_callee":
        call = next(op for op in before.walk() if op.name == "func.call")
        call.properties["callee"] = SymbolRefAttr("other")
        bounds = contract(7)
    elif kind == "external":
        bounds = contract(7)
    after = lowered(before)
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]


def coordinate_helper_source():
    """An output interval must be learned from a helper's body, not its result type."""
    from xdsl.dialects.builtin import TensorType, i64
    from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    before = Coordinates().source()
    forward = next(op for op in before.walk() if op.name == "func.func")
    range_op = next(op for op in forward.walk() if op.name == "linalg.generic")
    destination = range_op.operands[0].owner
    forward.body.block.detach_op(destination)
    forward.body.block.detach_op(range_op)
    returned = ReturnOp(range_op.results[0])
    block = Block([destination, range_op, returned])
    fn = FuncOp.from_region("coordinates", [], [range_op.results[0].type], Region(block))
    call = CallOp("coordinates", [], [TensorType(i64, [4])])
    first = next(iter(forward.body.block.ops))
    forward.body.block.insert_op_before(call, first)
    range_op.results[0].replace_uses_with_if(call.results[0], lambda use: use.operation is not returned)
    before.body.block.add_op(fn)
    return before, [{"function": "forward", "intervals": [None, None]}, {"function": "coordinates", "intervals": []}]


def test_callee_output_interval_is_inferred_from_its_producer(tmp_path, lean_tools):
    before, bounds = coordinate_helper_source()
    after = lowered(before)
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report


def test_called_embedding_and_preprocessing_match_native_lean_and_recipe(
    tmp_path, lean_tools, fidelity_tools, monkeypatch
):
    import numpy as np

    before, bounds = called_embedding(3)
    after = lowered(before)
    rest = preprocessed(after, tmp_path, monkeypatch)
    report, documents, _ = check_chain([before, after, *rest], bounds, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261025)
    samples, expected = [], []
    for indices in ([0, 0], [6, 6], [2, 5]):
        table = rng.uniform(-5, 5, (7, 3)).astype(np.float32)
        weights = rng.uniform(-3, 3, (3, 4)).astype(np.float32)
        samples.append([[str(i) for i in indices], words(table), words(weights)])
        expected.append(words(dynamic_matmul_reference(table[indices], weights)))
    compare_outputs(documents[-1], samples, expected, documents[-1].parent, fidelity_tools)

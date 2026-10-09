"""Ordinary source/partition intent stays joined; no device execution is proved."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.compute_groups import Group
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineError, OutlineResult, outline_dispatches
from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
    OutlinedGlobalPlanEmitter,
    _same_dispatch_wiring,
    plan_dispatch_fusion,
)

SOURCE = """builtin.module {
  func.func @forward(%x: tensor<3x5xi8>) -> (tensor<3x5xi8>, tensor<3x5xi8>) {
    %e = tensor.empty() : tensor<3x5xi8>
    %a = linalg.copy ins(%x : tensor<3x5xi8>) outs(%e : tensor<3x5xi8>) -> tensor<3x5xi8>
    %f = tensor.empty() : tensor<3x5xi8>
    %b = linalg.copy ins(%a : tensor<3x5xi8>) outs(%f : tensor<3x5xi8>) -> tensor<3x5xi8>
    func.return %a, %b : tensor<3x5xi8>, tensor<3x5xi8>
  }
}"""


def _outlined(*, grouped=True):
    module = parse_mlir_text(SOURCE)
    roots = [op for op in module.walk() if op.name == "linalg.copy"]
    groups = [Group(i, "selected_unit", op, [op], ["movement"]) for i, op in enumerate(roots)]
    return outline_dispatches(module, groups=groups if grouped else None)


def _function(outlined, index=0):
    symbol = outlined.dispatches[index].symbol
    return next(op for op in outlined.module.body.block.ops if op.name == "func.func" and op.sym_name.data == symbol)


def _calls(module):
    driver = next(op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == "forward")
    return [op for op in driver.body.block.ops if op.name == "func.call"]


def _plan(graph, *, fuse, placement="selected_unit"):
    return plan_dispatch_fusion(
        graph,
        [tuple(range(len(graph.nodes)))] if fuse else [],
        placement=placement,
        representation=lambda name: ValueRepresentation("tensor_ssa", "logical", graph.buffers[name].dtype),
    )


def test_normal_partition_keeps_complete_ordered_source_results_and_intent():
    outlined = _outlined()
    graph = build_dispatch_program(outlined)
    assert len(graph.results) == 2 and graph.results[0] != graph.results[1]
    assert [node.op for node in graph.nodes if node.kind == "dispatch"] == [row.symbol for row in outlined.dispatches]
    assert [row.placement for row in outlined.dispatches] == ["selected_unit", "selected_unit"]
    assert len([op for op in outlined.module.walk() if op.name == "linalg.copy"]) == 2


@pytest.mark.parametrize("defect", ["host", "missing", "wrong_type", "ambiguous"])
def test_changed_actual_function_cannot_keep_device_table_intent(defect):
    outlined = _outlined()
    function = _function(outlined)
    if defect == "host":
        function.attributes["merlin.placement"] = StringAttr("host")
    elif defect == "missing":
        del function.attributes["merlin.placement"]
    elif defect == "wrong_type":
        function.attributes["merlin.placement"] = IntegerAttr(0, IntegerType(64))
    else:
        function.properties["merlin.placement"] = StringAttr("host")
    # The typed computation and full original result slots did not change.
    assert [op.name for op in function.body.block.ops].count("linalg.copy") == 1
    with pytest.raises(OutlineError, match="merlin.placement.*actual function"):
        build_dispatch_program(outlined)


@pytest.mark.parametrize("defect", ["host_table", "no_table_intent", "group", "result_type", "operands"])
def test_table_cannot_replace_original_function_or_ordered_abi(defect):
    outlined = _outlined()
    row = outlined.dispatches[0]
    changes = {
        "host_table": {"placement": "host"},
        "no_table_intent": {"placement": None},
        "group": {"group": 19},
        "result_type": {"result_types": ["tensor<3x5xi32>"]},
        "operands": {"n_operands": row.n_operands + 1},
    }
    outlined.dispatches[0] = replace(row, **changes[defect])
    with pytest.raises(OutlineError, match="actual function|ordered call/function ABI"):
        build_dispatch_program(outlined)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "external"])
def test_named_call_needs_one_actual_owned_definition(defect):
    outlined = _outlined()
    function = _function(outlined)
    if defect == "missing":
        outlined.module.body.block.erase_op(function)
    elif defect == "duplicate":
        outlined.module.body.block.add_op(function.clone())
    else:
        function.body.erase_block(function.body.block)
    with pytest.raises(OutlineError, match="defined function|external|duplicate function"):
        build_dispatch_program(outlined)


@pytest.mark.parametrize("owner", ["attributes", "properties"])
def test_call_override_cannot_change_function_placement(owner):
    outlined = _outlined()
    getattr(_calls(outlined.module)[0], owner)["merlin.placement"] = StringAttr("host")
    with pytest.raises(OutlineError, match="actual call"):
        build_dispatch_program(outlined)


@pytest.mark.parametrize("index", [False, -1, 1])
def test_dispatch_table_indices_cannot_omit_or_reorder_source_calls(index):
    outlined = _outlined()
    outlined.dispatches[0] = replace(outlined.dispatches[0], index=index)
    with pytest.raises(OutlineError, match="actual call order"):
        build_dispatch_program(outlined)


def test_duplicate_dispatch_index_does_not_discharge_two_original_calls():
    outlined = _outlined()
    outlined.dispatches[1] = replace(outlined.dispatches[1], index=0)
    with pytest.raises(OutlineError, match="actual call order"):
        build_dispatch_program(outlined)


def test_absent_source_placement_stays_unspecified_until_explicit_fusion_choice():
    outlined = _outlined(grouped=False)
    graph = build_dispatch_program(outlined)
    assert all(row.placement is None for row in outlined.dispatches)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emit_global_plan(graph, _plan(graph, fuse=False), emitter)
    assert all(row.placement == "selected_unit" for row in emitter.emitted_outline.dispatches)
    assert emitter.proof["execution_placement"] == "UNKNOWN"


@pytest.mark.parametrize("fuse", [False, True])
def test_selected_fusion_intent_reaches_actual_calls_definitions_and_table(fuse):
    outlined = _outlined()
    before = str(outlined.module)
    graph = build_dispatch_program(outlined)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emission = emit_global_plan(graph, _plan(graph, fuse=fuse), emitter)
    assert str(outlined.module) == before
    assert len(emission.dispatch.results) == 2
    emitter.module.verify()
    replayed = build_dispatch_program(emitter.emitted_outline)
    assert len(replayed.results) == len(emission.dispatch.results) == 2
    assert [replayed.buffers[name].dtype for name in replayed.results] == ["i8", "i8"]
    assert [node.op for node in replayed.nodes] == [node.op for node in emission.dispatch.nodes]
    calls = _calls(emitter.module)
    assert len(calls) == (1 if fuse else 2)
    for index, call in enumerate(calls):
        assert call.attributes["merlin.placement"] == StringAttr("selected_unit")
        assert _function(emitter.emitted_outline, index).attributes["merlin.placement"] == StringAttr("selected_unit")
        assert emitter.emitted_outline.dispatches[index].placement == "selected_unit"
    assert emitter.proof["computation"] == "expanded_driver_structurally_equivalent"
    for role in ("execution_placement", "physical_movement", "resource_legality", "synchronization", "timing"):
        assert emitter.proof[role] == "UNKNOWN"


@pytest.mark.parametrize("fuse", [False, True])
def test_value_preserving_host_route_cannot_erase_original_selected_endpoint(fuse):
    outlined = _outlined()
    graph = build_dispatch_program(outlined)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    with pytest.raises(ValueError, match="differs from its original function"):
        emit_global_plan(graph, _plan(graph, fuse=fuse, placement="host"), emitter)
    assert emitter.module is emitter.proof is emitter.emitted_outline is None
    assert [_function(outlined, index).attributes["merlin.placement"].data for index in range(2)] == [
        "selected_unit",
        "selected_unit",
    ]


def test_fused_call_endpoint_mutation_is_detected_from_actual_ir():
    outlined = _outlined()
    graph = build_dispatch_program(outlined)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emit_global_plan(graph, _plan(graph, fuse=True), emitter)
    _calls(emitter.module)[0].attributes["merlin.placement"] = StringAttr("host")
    with pytest.raises(OutlineError, match="actual call"):
        build_dispatch_program(emitter.emitted_outline)


def test_actual_return_slots_cannot_swap_during_buffer_id_reassignment():
    outlined = _outlined()
    graph = build_dispatch_program(outlined)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emission = emit_global_plan(graph, _plan(graph, fuse=True), emitter)
    original_replay = build_dispatch_program(emitter.emitted_outline)
    assert _same_dispatch_wiring(emission.dispatch, original_replay)
    assert emission.dispatch.results != original_replay.results  # Fresh SSA ids are legitimate.
    driver = next(
        op for op in emitter.module.body.block.ops if op.name == "func.func" and op.sym_name.data == "forward"
    )
    returned = driver.body.block.last_op
    returned.operands = tuple(reversed(returned.operands))
    emitter.module.verify()  # Same typed values, wrong original ordered outputs.
    assert not _same_dispatch_wiring(emission.dispatch, build_dispatch_program(emitter.emitted_outline))


@pytest.mark.parametrize("conflicting", [False, True])
def test_repeated_actual_endpoint_keeps_each_call_and_refuses_conflicting_choices(conflicting):
    module = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<3x5xi8>) -> (tensor<3x5xi8>, tensor<3x5xi8>) {
        %a = func.call @forward$kernel_0(%x) : (tensor<3x5xi8>) -> tensor<3x5xi8>
        %b = func.call @forward$kernel_0(%a) : (tensor<3x5xi8>) -> tensor<3x5xi8>
        func.return %a, %b : tensor<3x5xi8>, tensor<3x5xi8>
      }
      func.func private @forward$kernel_0(%x: tensor<3x5xi8>) -> tensor<3x5xi8> {
        %e = tensor.empty() : tensor<3x5xi8>
        %a = linalg.copy ins(%x : tensor<3x5xi8>) outs(%e : tensor<3x5xi8>) -> tensor<3x5xi8>
        func.return %a : tensor<3x5xi8>
      }
    }""")
    module.verify()
    outlined = OutlineResult(
        module, [DispatchInfo(i, "forward$kernel_0", "linalg.copy", 1, ["tensor<3x5xi8>"]) for i in range(2)]
    )
    graph = build_dispatch_program(outlined)
    plan = _plan(graph, fuse=False)
    emitter = OutlinedGlobalPlanEmitter(outlined)
    if conflicting:
        plan = replace(plan, selected=(plan.selected[0], replace(plan.selected[1], placement="another_unit")))
        with pytest.raises(ValueError, match="conflicting placements"):
            emit_global_plan(graph, plan, emitter)
    else:
        emit_global_plan(graph, plan, emitter)
        assert len(_calls(emitter.module)) == 2
        assert [row.index for row in emitter.emitted_outline.dispatches] == [0, 1]
        assert [row.symbol for row in emitter.emitted_outline.dispatches] == ["forward$kernel_0"] * 2
        assert len(build_dispatch_program(emitter.emitted_outline).results) == 2

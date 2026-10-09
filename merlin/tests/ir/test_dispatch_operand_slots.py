"""Original ordered ABI slots survive DAG serialization and runtime consumption."""

from __future__ import annotations

import json

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.runtime.interpret import run_dispatch_program
from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, prune_dead_nodes, verify_program
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineResult, outline_dispatches
from merlin.xdsl_dialects.lowering.outlined_plan_emission import OutlinedGlobalPlanEmitter, plan_dispatch_fusion
from merlin.xdsl_dialects.lowering.schedule_dispatch import dependencies, emit_schedule_c, schedule


def _program(source, *, external=()):
    module = parse_mlir_text(source)
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    calls = [
        op
        for op in functions["forward"].body.block.ops
        if op.name == "func.call" and op.callee.string_value() not in external
    ]
    dispatches = [
        DispatchInfo(
            index,
            call.callee.string_value(),
            "func.func",
            len(call.operands),
            [str(value.type) for value in call.results],
        )
        for index, call in enumerate(calls)
    ]
    return build_dispatch_program(OutlineResult(module, dispatches, tuple(external)))


SCALAR = """builtin.module {
  func.func @forward(%x: i32) -> i32 {
    %y = func.call @forward$kernel_0(%x, %x) : (i32, i32) -> i32
    func.return %y : i32
  }
  func.func private @forward$kernel_0(%a: i32, %b: i32) -> i32 {
    %y = arith.addi %a, %b : i32
    func.return %y : i32
  }
}"""


def test_repeated_call_slots_reach_serialized_command_buffer_and_actual_consumer():
    program = _program(SCALAR)
    assert verify_program(program) == []
    assert program.nodes[0].inputs == ["b0", "b0"]
    command_buffer = json.loads(build_program(program, capability="scalar").to_json())
    assert command_buffer["dispatch"]["nodes"][0]["inputs"] == ["b0", "b0"]
    calls = []

    def invoke(symbol, arguments):
        calls.append((symbol, arguments))
        left, right = arguments
        return [left + right]

    assert run_dispatch_program(program, {"b0": 7}, invoke_kernel=invoke) == {"b1": 14}
    assert calls == [("forward$kernel_0", [7, 7])]
    scheduled = emit_schedule_c(program, schedule(program, n_harts=1))
    assert "{0, 0, -1, -1}, 2, 1" in scheduled


@pytest.mark.parametrize("element_type", ["i32", "tensor<3x5xi8>"])
def test_repeated_arguments_keep_slot_multiplicity_for_scalars_and_tensors(element_type):
    source = f"""builtin.module {{
      func.func @forward(%x: {element_type}) -> {element_type} {{
        %y = func.call @forward$kernel_0(%x, %x, %x)
          : ({element_type}, {element_type}, {element_type}) -> {element_type}
        func.return %y : {element_type}
      }}
      func.func private @forward$kernel_0(%a: {element_type}, %b: {element_type}, %c: {element_type})
          -> {element_type} {{
        func.return %a : {element_type}
      }}
    }}"""
    program = _program(source)
    assert program.nodes[0].inputs == ["b0", "b0", "b0"]
    assert prune_dead_nodes(program).nodes[0].inputs == ["b0", "b0", "b0"]
    value = object()

    def invoke(_symbol, arguments):
        assert arguments == [value, value, value]
        return [arguments[0]]

    assert run_dispatch_program(program, {"b0": value}, invoke_kernel=invoke) == {"b1": value}


def test_repeated_multi_result_slot_is_distinct_from_other_results_and_dependencies():
    source = """builtin.module {
      func.func @forward(%x: i32, %y: i32) -> (i32, i32, i32) {
        %a:2 = func.call @forward$kernel_0(%x, %y) : (i32, i32) -> (i32, i32)
        %b = func.call @forward$kernel_1(%a#1, %a#0, %a#1) : (i32, i32, i32) -> i32
        func.return %a#1, %b, %a#1 : i32, i32, i32
      }
      func.func private @forward$kernel_0(%a: i32, %b: i32) -> (i32, i32) {
        func.return %a, %b : i32, i32
      }
      func.func private @forward$kernel_1(%a: i32, %b: i32, %c: i32) -> i32 {
        %d = arith.addi %a, %b : i32
        %e = arith.addi %d, %c : i32
        func.return %e : i32
      }
    }"""
    program = _program(source)
    assert program.nodes[1].inputs == ["b3", "b2", "b3"]
    assert program.results == ["b3", "b4", "b3"]
    assert dependencies(program) == [set(), {0}]
    assert json.loads(build_program(program, capability="scalar").to_json())["dispatch"]["results"] == [
        "b3",
        "b4",
        "b3",
    ]

    def invoke(symbol, arguments):
        return list(arguments) if symbol.endswith("_0") else [sum(arguments)]

    assert run_dispatch_program(program, {"b0": 2, "b1": 5}, invoke_kernel=invoke) == {"b3": 5, "b4": 12}


def test_scalar_view_retains_two_original_add_operands():
    program = _program("""builtin.module {
      func.func @forward(%x: i32) -> i32 {
        %y = arith.addi %x, %x : i32
        func.return %y : i32
      }
    }""")
    assert program.nodes[0].inputs == ["b0", "b0"]

    def view(_op, arguments, _node):
        left, right = arguments
        return [left + right]

    assert run_dispatch_program(program, {"b0": 9}, invoke_kernel=None, eval_view=view) == {"b1": 18}


def test_external_original_abi_also_preserves_repeated_call_arguments():
    program = _program(
        SCALAR.replace("forward$kernel_0", "opaque").replace(
            "(%a: i32, %b: i32) -> i32 {\n    %y = arith.addi %a, %b : i32\n    func.return %y : i32\n  }",
            "(i32, i32) -> i32",
        ),
        external=("opaque",),
    )
    assert program.nodes[0].inputs == ["b0", "b0"]
    assert program.nodes[0].op == "opaque"


def test_region_capture_set_does_not_deduplicate_original_loop_operand_slots():
    source = """builtin.module {
      func.func @forward(%x: i32, %captured: i32) -> (i32, i32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %r:2 = scf.for %i = %c0 to %c1 step %c1 iter_args(%a = %x, %b = %x) -> (i32, i32) {
          %d = arith.addi %captured, %captured : i32
          %m = arith.index_cast %c1 : index to i32
          %n = arith.addi %m, %d : i32
          scf.yield %n, %b : i32, i32
        }
        func.return %r#0, %r#1 : i32, i32
      }
    }"""
    program = _program(source)
    loop = program.nodes[2]
    assert loop.inputs == ["b2", "b3", "b3", "b0", "b0", "b1"]
    assert loop.captures == ["b1", "b3"]
    assert dependencies(program) == [set(), set(), {0, 1}]
    assert verify_program(program) == []


def test_singleton_global_emission_reopens_the_actual_repeated_call_slots():
    outlined = OutlineResult(parse_mlir_text(SCALAR), [DispatchInfo(0, "forward$kernel_0", "arith.addi", 2, ["i32"])])
    program = build_dispatch_program(outlined)
    plan = plan_dispatch_fusion(
        program,
        [],
        placement="selected_endpoint",
        representation=lambda name: ValueRepresentation("tensor_ssa", "logical", program.buffers[name].dtype),
    )
    emitter = OutlinedGlobalPlanEmitter(outlined)
    emission = emit_global_plan(program, plan, emitter)
    assert emission.dispatch.nodes[0].inputs == ["b0", "b0"]
    assert build_dispatch_program(emitter.emitted_outline).nodes[0].inputs == ["b0", "b0"]
    assert emitter.proof["execution_placement"] == "UNKNOWN"


def test_outliner_unique_free_values_still_form_the_actual_kernel_parameter_list():
    source = """builtin.module {
      func.func @forward(%x: tensor<3x5xi8>) -> tensor<3x5xi8> {
        %e = tensor.empty() : tensor<3x5xi8>
        %y = linalg.add ins(%x, %x : tensor<3x5xi8>, tensor<3x5xi8>)
             outs(%e : tensor<3x5xi8>) -> tensor<3x5xi8>
        func.return %y : tensor<3x5xi8>
      }
    }"""
    outlined = outline_dispatches(parse_mlir_text(source))
    assert outlined.dispatches[0].n_operands == 1
    program = prune_dead_nodes(build_dispatch_program(outlined))
    assert program.nodes[0].inputs == ["b0"]
    function = next(
        op for op in outlined.module.body.block.ops if op.name == "func.func" and "$kernel_" in op.sym_name.data
    )
    add = next(op for op in function.walk() if op.name == "linalg.add")
    assert add.inputs[0] is add.inputs[1]

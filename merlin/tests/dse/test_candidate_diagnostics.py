"""Own-artifact diagnostics retain unknowns and never grant timing/numerics."""

import hashlib
import json
from copy import deepcopy

from merlin.perf.candidate_diagnostics import observe_candidate_artifacts
from merlin.targetgen.component_program import SCHEMA_V2, render

HOST = '''module { llvm.func @kernel(%a: !llvm.ptr) {
%v = llvm.load %a : !llvm.ptr -> i32
%w = llvm.add %v, %v : i32
llvm.store %w, %a : i32, !llvm.ptr
llvm.return
} }'''


def test_integer_reduction_source_domain_host_payload_and_refusal_are_separate():
    program = {"schema": SCHEMA_V2,
               "inputs": [{"name": "a", "role": "input", "shape": [2, 3], "dtype": "i8"}],
               "nodes": [{"name": "r", "op": "reduce_sum", "inputs": ["a"], "axis": 1, "dtype": "i32"}],
               "outputs": [{"name": "result", "value": "r"}]}
    _, source = render(program, operand_dtype="i8", accumulator_dtype="i32")
    report = observe_candidate_artifacts(
        source_mlir=source, emitted_mlir=HOST,
        command_buffer={"declined": {"reason": "unsupported reduction"}, "tensors": {}, "commands": []},
        artifacts={"source": source.encode(), "emitted": HOST.encode()}, entry_symbol="kernel",
        compiler_options={"packing": True}, numeric_context={"accumulator_dtype": "i32"},
    )
    assert report["source"]["status"] == "observed", report["source"]
    assert report["source"]["reduction_domain_points"] == 6
    features = report["candidate_features"]["features"]
    assert features["host_integer_arithmetic"] == 1
    assert features["host_load_payload_bytes"] == features["host_store_payload_bytes"] == 4
    assert report["placement"]["refusal"]["reason"] == "unsupported reduction"
    assert report["placement"]["source_operation_to_execution_lane"] == "UNKNOWN"
    assert report["array_utilization"] is report["physical_memory_traffic"] is report["timing_cycles"] is None
    assert report["numerical_context"]["numerical_verdict"] == "UNKNOWN"
    assert report["compiler_options"]["applied_to_codegen"] == "UNKNOWN"


def test_observation_identity_changes_with_artifacts_and_options_without_exposing_paths():
    report = observe_candidate_artifacts(emitted_mlir=HOST, artifacts={"object": b"one"},
                                        compiler_options={"argv": ["-O2", "/private/toolchain"], "packing": True})
    assert "/private" not in json.dumps(report)
    assert report["artifacts"]["object"]["sha256"] == hashlib.sha256(b"one").hexdigest()
    changed = observe_candidate_artifacts(emitted_mlir=HOST, artifacts={"object": b"two"},
                                         compiler_options={"packing": False})
    assert report["observation_sha256"] != changed["observation_sha256"]


def test_observer_scope_and_placement_prose_stay_readable_while_private_paths_are_hashed():
    report = observe_candidate_artifacts(command_buffer={"declined": {"reason": "/private/candidate"}})
    assert report["dependency_stages"]["scope"] == (
        "original complete source and CFG ownership and declared tensor dependencies"
    )
    assert report["placement"]["basis"] == (
        "candidate placement and refusal declarations and exact named interface tensor types"
    )
    assert report["placement"]["limitation"].startswith("matching names and types")
    assert report["placement"]["refusal"]["reason"] == {
        "text_sha256": hashlib.sha256(b"/private/candidate").hexdigest(), "text_withheld": True,
    }
    assert "/private" not in json.dumps(report)


def test_unsupported_source_host_and_frame_remain_unknown_not_zero():
    report = observe_candidate_artifacts(source_mlir='module { invalid "/private/source" }',
                                        emitted_mlir="not LLVM", stack_usage_text="ambiguous frame")
    assert report["source"]["status"] == report["host"]["status"] == report["stack_frames"]["status"] == "UNKNOWN"
    assert report["candidate_features"]["features"]["source_macs"] is None
    assert report["candidate_features"]["features"]["host_load_payload_bytes"] is None
    assert "/private" not in json.dumps(report)


def test_unregistered_source_work_is_visible_but_cannot_be_priced_as_zero():
    source = 'module { func.func @forward() { "unknown.work"() : () -> ()\nfunc.return } }'
    report = observe_candidate_artifacts(source_mlir=source)
    assert report["source"]["status"] == "observed"
    assert report["source"]["operation_histogram"]["unknown.work"] == 1
    assert report["source"]["macs"] is report["source"]["reduction_domain_points"] is None


def test_original_global_plan_source_join_and_drift_control():
    source = '''module { func.func @forward() -> tensor<2xi32> {
%v = arith.constant dense<[3, 7]> : tensor<2xi32>
func.return %v : tensor<2xi32> } }'''
    emitted = '''module { llvm.func @kernel(%p: !llvm.ptr) {
%i = llvm.mlir.constant(0 : i64) : i64
%q = llvm.getelementptr %p[%i] {merlin.global_task = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
llvm.return } }'''
    cb = {"tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
          "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "result"}], "outputs": ["result"]},
          "params": {"global_program_plan": {
              "schema": "mixed_program_plan_v1", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "source_op_count": 1, "tasks": [{"task_index": 0, "kind": "host", "source_op_indices": [0],
              "instruction_start": 1, "instruction_end": 2, "reads": [], "writes": ["result"]}],
              "schedule_instruction_count": 3, "prologue_instruction_range": [0, 1],
              "epilogue_instruction_range": [2, 3],
              "source_values": [{"op_index": 0, "result_index": 0, "tensor": "result"}],
              "entry_bindings": [], "output_bindings": ["result"]}}}
    def observe(buffer):
        return observe_candidate_artifacts(source_mlir=source, emitted_mlir=emitted,
                                          command_buffer=buffer, candidate_sha256="a" * 64)["source_emission_join"]
    joined = observe(cb)
    assert joined["status"] == "verified", joined
    assert joined["timing"] == "UNKNOWN"
    assert "UNVERIFIED" in joined["execution_placement"]
    changed = deepcopy(cb)
    changed["params"]["global_program_plan"]["source_sha256"] = "b" * 64
    assert observe(changed)["status"] == "refused"


def test_interface_source_work_resolves_resident_rhs_from_original_semantics():
    from merlin.perf.candidate_source import source_observation
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    cb = {"target": "test_unit", "tensors": {
        "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
        "W": {"shape": [3, 4], "dtype": "i8", "role": "weight"},
        "Y": {"shape": [2, 4], "dtype": "i32", "role": "output"}}, "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "private_weight"}, "attributes": {}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "private_weight", "dst": "partial"},
         "attributes": {}},
        {"opcode": "COMMIT", "operands": {"src": "partial", "dst": "Y"}, "attributes": {"output_dtype": "i32"}},
        {"opcode": "EVICT", "operands": {"handle": "private_weight"}, "attributes": {}}]}
    observed = source_observation(emit_interface_mlir(cb))
    assert observed["macs"] == 24 and observed["work_refusals"] == []


def test_elementwise_lhs_and_destination_shapes_do_not_invent_contraction_macs():
    from merlin.perf.candidate_source import source_observation
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    cb = {"target": "test_unit", "tensors": {
        name: {"shape": [17, 16], "dtype": "i8", "role": "output" if name == "Y" else "input"}
        for name in ("A", "B", "Y")}, "commands": [{"opcode": "RESIDUAL_ADD",
        "operands": {"lhs": "A", "rhs": "B", "dst": "Y"}, "attributes": {}}]}
    observed = source_observation(emit_interface_mlir(cb))
    assert observed["macs"] is None
    assert "RESIDUAL_ADD" in observed["work_refusals"][0]

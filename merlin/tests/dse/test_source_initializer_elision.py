"""Implicit initializer evidence never waives unrelated source or accumulator state."""
import copy
import json

import pytest

from merlin.perf.source_initializer_elision import prove_implicit_zero_initializer
from merlin.perf.source_program_pair import text_digest


SOURCE = '''module {func.func @work(%c:tensor<2x3xi8>,%a:tensor<2x5xi8>,%b:tensor<5x3xi8>)->tensor<2x3xi8>{
%z=arith.constant 0:i8
%init=linalg.fill ins(%z:i8) outs(%c:tensor<2x3xi8>)->tensor<2x3xi8>
%r=linalg.matmul ins(%a,%b:tensor<2x5xi8>,tensor<5x3xi8>) outs(%init:tensor<2x3xi8>)->tensor<2x3xi8>
func.return %r:tensor<2x3xi8>}}'''


def buffer(source=SOURCE):
    return {"tensors": {name: {} for name in ("C", "A", "B", "Y", "Y_acc")},
        "kernel_abi": {"kind": "whole_program", "outputs": ["Y"]},
        "params": {"global_program_plan": {"source_sha256": text_digest(source), "source_op_count": 3,
            "entry_bindings": ["C", "A", "B"], "output_bindings": ["Y"],
            "tasks": [{"task_index": 0, "kind": "contraction", "source_op_indices": [2],
                       "reads": ["A", "B"], "writes": ["Y", "Y_acc"], "accumulator_temporary": "Y_acc"}],
            "source_values": [{"op_index": 2, "result_index": 0, "tensor": "Y"}],
            "compiler_temporaries": [{"purpose": "accumulator_readout", "tensor": "Y_acc", "source_op_index": 2}]}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "B", "dst": "B_res"}, "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "B_res", "dst": "fresh"}},
            {"opcode": "COMMIT", "operands": {"src": "fresh", "dst": "Y_acc"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}}]}


def test_exact_zero_chain_is_separate_source_to_command_evidence():
    cb = buffer()
    original = copy.deepcopy(cb)
    proof = prove_implicit_zero_initializer(source_text=SOURCE, command_buffer=cb, source_op_index=2)
    assert proof["status"] == "proved", proof
    assert proof["implicit_source_op_indices"] == [0, 1]
    assert proof["fresh_accumulator"] == "fresh" and cb == original
    assert proof["emitted_machine_initialization"] == proof["numeric_equivalence"] == "UNPROVEN"
    assert not proof["declared_plan_modified"]


def test_fill_observed_as_second_return_is_not_single_use_elidable():
    source = SOURCE.replace("->tensor<2x3xi8>{", "->(tensor<2x3xi8>,tensor<2x3xi8>){", 1)
    source = source.replace("func.return %r:tensor<2x3xi8>",
                            "func.return %r,%init:tensor<2x3xi8>,tensor<2x3xi8>")
    proof = prove_implicit_zero_initializer(source_text=source, command_buffer=buffer(source), source_op_index=2)
    assert proof["status"] == "UNKNOWN" and "consumer" in proof["reason"]


@pytest.mark.parametrize("mutation", ["nonzero", "multiuse", "accumulator_input", "accumulate", "wrong_lhs", "wrong_rhs", "wrong_source_map", "extra_command", "extra_missing", "stale_source"])
def test_nonidentity_or_stale_source_refuses(mutation):
    source = SOURCE
    if mutation == "nonzero":
        source = source.replace("constant 0", "constant 7")
    if mutation == "multiuse":
        source = source.replace("%r=linalg.matmul", "%other=arith.addi %z,%z:i8\n%r=linalg.matmul")
    cb = buffer(source)
    plan = cb["params"]["global_program_plan"]
    if mutation == "accumulator_input":
        cb["commands"][1]["operands"]["dst"] = "C"
        cb["commands"][2]["operands"]["src"] = "C"
    elif mutation == "accumulate":
        cb["commands"][1]["attributes"] = {"accumulate": True}
    elif mutation == "wrong_lhs":
        cb["commands"][1]["operands"]["lhs"] = "C"
    elif mutation == "wrong_rhs":
        cb["commands"][0]["operands"]["src"] = "A"
    elif mutation == "wrong_source_map":
        plan["entry_bindings"] = ["C", "B", "A"]
    elif mutation == "extra_command":
        cb["commands"].append(copy.deepcopy(cb["commands"][1]))
    elif mutation == "extra_missing":
        plan["tasks"][0]["source_op_indices"] = []
    elif mutation == "stale_source":
        plan["source_sha256"] = "stale"
    assert prove_implicit_zero_initializer(source_text=source, command_buffer=cb, source_op_index=2)["status"] == "UNKNOWN"


def structural_artifact():
    cb = buffer()
    names = ["C", "A", "B", "Y", "Y_acc"]
    for name, shape in zip(names, ([2,3], [2,5], [5,3], [2,3], [2,3])):
        cb["tensors"][name] = {"shape": shape, "dtype": "i32" if name == "Y_acc" else "i8",
            "role": "input" if name in {"C", "A", "B"} else "output" if name == "Y" else "intermediate"}
    cb["kernel_abi"]["args"] = [{"tensor": name, "access": "read" if i < 3 else "write"}
                                 for i, name in enumerate(names)]
    plan = cb["params"]["global_program_plan"]
    plan.update(schema="mixed_program_plan_v1", schedule_instruction_count=3,
                prologue_instruction_range=[0,1], epilogue_instruction_range=[2,3])
    plan["tasks"][0].update(instruction_start=1, instruction_end=2)
    plan["compiler_temporaries"][0]["source_result_index"] = 0
    # This deliberately does not compute a MAC. The supplemental admission must
    # remain numerical-PROBE permission, never a numerical/target-route proof.
    llvm = '''builtin.module {llvm.func @kernel(%c:!llvm.ptr,%a:!llvm.ptr,%b:!llvm.ptr,%y:!llvm.ptr,%acc:!llvm.ptr){
%z=llvm.mlir.constant(0:i64):i64
%p=llvm.getelementptr %y[%z] {merlin.global_task=0:i64}:(!llvm.ptr,i64)->!llvm.ptr,i8
llvm.return}}'''
    return cb, llvm


def test_supplement_is_numerical_probe_permission_not_a_forged_plan_pass():
    from merlin.perf.source_initializer_elision import numerical_probe_admission
    from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
    from merlin.perf.source_program_pair import document_digest
    cb, llvm = structural_artifact()
    proof = verify_compiler_global_plan(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        candidate_sha256="a"*64, command_buffer_sha256=document_digest(cb))
    assert proof["problems"] == ["source operations are not fully covered: [0, 1]"]
    original = copy.deepcopy(proof)
    admission = numerical_probe_admission(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        declared_proof=proof, candidate_sha256="a"*64, source_op_index=2)
    assert admission["status"] == "source_bound_numerical_probe", admission
    assert proof == original and proof["status"] == "refused"
    assert admission["numeric_equivalence"] == "UNPROVEN"
    assert not admission["declared_plan_verified"] and not admission["target_route_verified"]
    assert not admission["global_cost_calibration"]


def test_wrapper_task_keys_survive_receipt_transport_without_changing_proof():
    from merlin.perf.source_initializer_elision import numerical_probe_admission
    from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
    from merlin.perf.source_program_pair import document_digest
    cb, llvm = structural_artifact()
    llvm = llvm.replace("%p=llvm.getelementptr", '%wrapper=llvm.getelementptr %y[%z] {merlin.global_task=-1:i64}:(!llvm.ptr,i64)->!llvm.ptr,i8\n%p=llvm.getelementptr')
    llvm = llvm.replace("llvm.return", '%end=llvm.getelementptr %y[%z] {merlin.global_task=-2:i64}:(!llvm.ptr,i64)->!llvm.ptr,i8\nllvm.return')
    proof = verify_compiler_global_plan(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        candidate_sha256="a"*64, command_buffer_sha256=document_digest(cb))
    transported = json.loads(json.dumps(proof))
    assert proof["problems"] == ["source operations are not fully covered: [0, 1]"]
    assert document_digest(proof) != document_digest(transported)
    admissions = [numerical_probe_admission(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        declared_proof=p, candidate_sha256="a"*64, source_op_index=2) for p in (proof, transported)]
    assert admissions[0] == admissions[1]
    assert admissions[0]["status"] == "source_bound_numerical_probe", admissions
    assert admissions[0]["declared_proof_sha256"] == document_digest(transported)
    transported["control_flow"]["status"] = "UNKNOWN"
    assert numerical_probe_admission(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        declared_proof=transported, candidate_sha256="a"*64, source_op_index=2)["status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation", ["unowned", "abi", "stale_llvm", "stale_proof"])
def test_other_cfg_abi_and_identity_defects_cannot_use_initializer_permission(mutation):
    from merlin.perf.source_initializer_elision import numerical_probe_admission
    from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
    from merlin.perf.source_program_pair import document_digest
    cb, llvm = structural_artifact()
    if mutation == "unowned":
        llvm = llvm.replace("{merlin.global_task=0:i64}", "")
    if mutation == "abi":
        cb["kernel_abi"]["args"].pop()
    proof = verify_compiler_global_plan(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        candidate_sha256="a"*64, command_buffer_sha256=document_digest(cb))
    if mutation == "stale_llvm":
        llvm += "\n"
    if mutation == "stale_proof":
        proof["status"] = "verified"
    admission = numerical_probe_admission(source_text=SOURCE, lowered_text=llvm, command_buffer=cb,
        declared_proof=proof, candidate_sha256="a"*64, source_op_index=2)
    assert admission["status"] == "UNKNOWN"

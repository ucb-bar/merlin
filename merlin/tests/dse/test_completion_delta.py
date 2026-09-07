"""Relative IR proof with independent completion ABIs and conservative visibility outcomes."""
import hashlib
import io

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i1, i32, i64
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.perf.completion_delta import CompletionContract, qualify_relative_completion_delta


def artifact(spec, *, completion="wait_tiles", host_value=7):
    block = Block(arg_types=[])
    rows = []
    for task, kind in spec:
        if kind == "host":
            block.add_op(llvm.ConstantOp(IntegerAttr(host_value, i32), i32))
            continue
        if kind == "host_store":
            count = llvm.ConstantOp(IntegerAttr(1, i64), i64)
            storage = llvm.AllocaOp(count, i32)
            value = llvm.ConstantOp(IntegerAttr(host_value, i32), i32)
            block.add_ops([count, storage, value, llvm.StoreOp(value, storage)])
            continue
        op = llvm.InlineAsmOp(completion if kind == "complete" else "device_work",
                              "~{memory}", [], [], has_side_effects=True)
        op.attributes["merlin.global_task"] = IntegerAttr(task, i64)
        block.add_op(op)
        rows.append({"class": kind})
    block.add_op(llvm.ReturnOp())
    module = ModuleOp([llvm.FuncOp("kernel", llvm.LLVMFunctionType([]), body=Region([block]))])
    return bundle(module, rows)


def bundle(module, rows):
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    text = stream.getvalue()
    sha = hashlib.sha256(text.encode()).hexdigest()
    plan = {"status": "verified", "candidate_lowered_sha256": sha,
            "candidate_command_buffer_sha256": "a" * 64, "source_sha256": "b" * 64}
    return ({"emission": {"candidate_lowered_sha256": sha},
             "diagnostics": {"verified_global_plan_emission": plan}},
            {"lowered_text": text, "candidate_lowered_sha256": sha,
             "candidate_command_buffer_sha256": "a" * 64,
             "decoded_trace": {"instructions": rows}, "parsed_lowered_module": module})


def contract(assembly):
    return CompletionContract(
        "fixture-completion-contract", {"source": "independent test ABI"},
        lambda row: row.get("class") == "complete",
        lambda op, row: row.get("class") == "complete" and op.asm_string.data == assembly
        and op.constraints.data == "~{memory}" and op.has_side_effects is not None)


def run(before, after, assembly="wait_tiles"):
    return qualify_relative_completion_delta(previous_analysis=before[0], current_analysis=after[0],
                                             previous_artifacts=before[1], current_artifacts=after[1],
                                             contract=contract(assembly))


@pytest.mark.parametrize("assembly", ["wait_tiles", "join_packets"])
def test_relative_completion_with_later_visibility_barrier(assembly):
    before = artifact([(0, "complete"), (0, "host"), (1, "complete"),
                       (2, "complete"), (2, "device")], completion=assembly)
    after = artifact([(0, "complete"), (0, "host"), (2, "complete"), (2, "device")],
                     completion=assembly)
    original_count = sum(1 for _ in before[1]["parsed_lowered_module"].walk())
    result = run(before, after, assembly)
    assert result["device_completion_redundancy"] == "verified"
    assert result["host_visibility"] == "verified"
    assert result["relative_synchronization_qualified"]
    assert result["numerical_equivalence"] == "NOT_ESTABLISHED"
    assert sum(1 for _ in before[1]["parsed_lowered_module"].walk()) == original_count


def test_device_completion_does_not_prove_later_host_write_visibility():
    before = artifact([(0, "complete"), (0, "host"), (1, "complete"), (2, "device")])
    after = artifact([(0, "complete"), (0, "host"), (2, "device")])
    result = run(before, after)
    assert result["device_completion_redundancy"] == "verified"
    assert result["host_visibility"] == "UNRESOLVED"
    assert not result["relative_synchronization_qualified"]


@pytest.mark.parametrize("assembly", ["wait_tiles", "join_packets"])
def test_terminal_exit_wrapper_proves_only_device_completion(assembly):
    before = artifact([(0, "complete"), (0, "host"), (-2, "complete")], completion=assembly)
    after = artifact([(0, "complete"), (0, "host")], completion=assembly)
    result = run(before, after, assembly)
    assert result["device_completion_redundancy"] == "verified"
    assert result["host_visibility"] == "UNRESOLVED"
    assert result["caller_visible_ordering"] == "UNKNOWN"
    assert not result["relative_synchronization_qualified"]
    assert result["status"] == "conditional_device_completion_delta_verified"
    assert result["deleted_completions"][0]["deletion_scope"] == "terminal_exit_wrapper"
    assert result["complete_ir_equal_after_only_identified_deletions"]


def test_terminal_exit_host_stores_do_not_acquire_caller_ordering_authority():
    before = artifact([(0, "complete"), (0, "host_store"), (-2, "complete")])
    after = artifact([(0, "complete"), (0, "host_store")])
    result = run(before, after)
    assert result["device_completion_redundancy"] == "verified"
    assert result["host_visibility"] == "UNRESOLVED"
    assert result["caller_visible_ordering"] == "UNKNOWN"
    assert "store" in result["deleted_completions"][0]["intervening_host_categories"]
    assert not result["relative_synchronization_qualified"]


@pytest.mark.parametrize("before_spec,after_spec", [
    ([(-1, "complete"), (0, "host")], [(0, "host")]),
    ([(0, "complete"), (-1, "complete")], [(0, "complete")]),
    ([(0, "complete"), (-3, "complete")], [(0, "complete")]),
    ([(0, "complete"), (-2, "complete"), (0, "host")], [(0, "complete"), (0, "host")]),
    ([(0, "complete"), (1, "device"), (-2, "complete")], [(0, "complete"), (1, "device")]),
])
def test_unsupported_wrapper_and_device_issue_are_not_partial_proofs(before_spec, after_spec):
    result = run(artifact(before_spec), artifact(after_spec))
    assert result["status"] == "unresolved"
    assert result["device_completion_redundancy"] == "UNRESOLVED"
    assert not result["relative_synchronization_qualified"]


def test_terminal_exit_unknown_completion_contract_is_not_partial_proof():
    before = artifact([(0, "complete"), (-2, "complete")])
    after = artifact([(0, "complete")])
    result = run(before, after, "different_target_completion")
    assert result["status"] == "unresolved"
    assert result["device_completion_redundancy"] == "UNRESOLVED"


def test_terminal_exit_changed_host_stores_are_not_partial_proof():
    before = artifact([(0, "complete"), (0, "host_store"), (-2, "complete")])
    after = artifact([(0, "complete"), (0, "host_store")], host_value=8)
    result = run(before, after)
    assert result["status"] == "unresolved"
    assert "complete LLVM differs" in result["reason"]
    assert result["device_completion_redundancy"] == "UNRESOLVED"


def test_host_arithmetic_changes_cannot_hide_behind_completion_deletion():
    before = artifact([(0, "complete"), (0, "host"), (1, "complete")])
    after = artifact([(0, "complete"), (0, "host")], host_value=8)
    result = run(before, after)
    assert result["status"] == "unresolved"
    assert "complete LLVM differs" in result["reason"]


def test_removed_device_work_is_not_a_completion_transformation():
    result = run(artifact([(0, "complete"), (1, "device")]), artifact([(0, "complete")]))
    assert result["status"] == "not_applicable"
    assert not result["relative_synchronization_qualified"]


def test_stale_verified_abi_does_not_qualify():
    before = artifact([(0, "complete"), (0, "host"), (1, "complete")])
    after = artifact([(0, "complete"), (0, "host")])
    after[0]["diagnostics"]["verified_global_plan_emission"]["candidate_command_buffer_sha256"] = "c" * 64
    result = run(before, after)
    assert result["status"] == "unresolved"
    assert "ABI contract is not bound" in result["reason"]


def test_a_deleted_barrier_cannot_discharge_another_deleted_barrier():
    before = artifact([(0, "complete"), (1, "complete"), (2, "complete"), (3, "device")])
    after = artifact([(0, "complete"), (3, "device")])
    result = run(before, after)
    assert result["status"] == "unresolved"
    assert "also deleted" in result["reason"]


@pytest.mark.parametrize("deleted_owner", [1, -2])
def test_lexically_preceding_completion_does_not_prove_cfg_domination(deleted_owner):
    entry, left, right, join = (Block(arg_types=[]) for _ in range(4))
    condition = llvm.ConstantOp(IntegerAttr(1, i1), i1)
    entry.add_ops([condition, llvm.CondBrOp(condition, left, [], right, [])])
    complete = llvm.InlineAsmOp("wait_tiles", "~{memory}", [], [], has_side_effects=True)
    complete.attributes["merlin.global_task"] = IntegerAttr(0, i64)
    left.add_ops([complete, llvm.BrOp(join)])
    right.add_op(llvm.BrOp(join))
    deleted = llvm.InlineAsmOp("wait_tiles", "~{memory}", [], [], has_side_effects=True)
    deleted.attributes["merlin.global_task"] = IntegerAttr(deleted_owner, i64)
    join.add_ops([deleted, llvm.ReturnOp()])
    module = ModuleOp([llvm.FuncOp("kernel", llvm.LLVMFunctionType([]),
                                  body=Region([entry, left, right, join]))])
    before = bundle(module, [{"class": "complete"}, {"class": "complete"}])
    changed = module.clone()
    command = [op for op in changed.walk() if op.name == "llvm.inline_asm"][-1]
    command.parent.erase_op(command)
    after = bundle(changed, [{"class": "complete"}])
    result = run(before, after)
    assert result["status"] == "unresolved"
    assert "does not dominate" in result["reason"]

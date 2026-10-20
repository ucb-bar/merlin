"""A private inverse, independent of the editable transformation's claims."""

import copy
import hashlib

import pytest

from merlin.perf.zero_initializer_evidence import check_zero_initializer_grouping
from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan

SOURCE = """module { func.func @forward(%a:tensor<3x5xi8>, %b:tensor<5x7xi8>) -> tensor<3x7xi32> {
  %z = arith.constant 0 : i32
  %init = "tensor.splat"(%z) : (i32) -> tensor<3x7xi32>
  %r = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
      affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}
      ins(%a,%b:tensor<3x5xi8>,tensor<5x7xi8>) outs(%init:tensor<3x7xi32>) {
    ^bb0(%x:i8,%y:i8,%s:i32):
      %xx = arith.extsi %x:i8 to i32
      %yy = arith.extsi %y:i8 to i32
      %p = arith.muli %xx,%yy:i32
      %v = arith.addi %s,%p:i32
      linalg.yield %v:i32
  } -> tensor<3x7xi32>
  return %r:tensor<3x7xi32>
}}"""


def pair(source=SOURCE):
    tensors = {
        "a": {"shape": [3, 5], "dtype": "i8", "role": "input"},
        "b": {"shape": [5, 7], "dtype": "i8", "role": "input"},
        "seed": {"shape": [3, 7], "dtype": "i32", "role": "intermediate"},
        "out": {"shape": [3, 7], "dtype": "i32", "role": "output"},
    }
    plan = {
        "schema": "mixed_program_plan_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_op_count": 3,
        "entry_bindings": ["a", "b"],
        "output_bindings": ["out"],
        "source_values": [
            {"op_index": 1, "result_index": 0, "tensor": "seed"},
            {"op_index": 2, "result_index": 0, "tensor": "out"},
        ],
        "compiler_temporaries": [],
        "prologue_instruction_range": [0, 1],
        "epilogue_instruction_range": [4, 5],
        "schedule_instruction_count": 5,
        "tasks": [
            {
                "task_index": index,
                "kind": "device" if index == 2 else "host",
                "source_op_indices": [index],
                "instruction_start": index + 1,
                "instruction_end": index + 2,
                "reads": ["a", "b", "seed"] if index == 2 else [],
                "writes": ["out"] if index == 2 else ["seed"] if index == 1 else [],
            }
            for index in range(3)
        ],
    }
    before = {
        "tensors": tensors,
        "commands": [{"opcode": "original work", "operands": {"output": "out"}}],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "read" if name in {"a", "b"} else "write"} for name in tensors],
            "outputs": ["out"],
        },
        "compiler_pointer_abi": {"version": 1, "arguments": list(tensors)},
        "params": {"global_program_plan": plan},
    }
    after = copy.deepcopy(before)
    del after["tensors"]["seed"]
    after["kernel_abi"]["args"] = [row for row in after["kernel_abi"]["args"] if row["tensor"] != "seed"]
    after["compiler_pointer_abi"]["arguments"].remove("seed")
    new = after["params"]["global_program_plan"]
    new["source_values"].pop(0)
    task = new["tasks"][-1]
    task.update(task_index=0, source_op_indices=[0, 1, 2], reads=["a", "b"], instruction_start=1, instruction_end=2)
    new.update(tasks=[task], epilogue_instruction_range=[2, 3], schedule_instruction_count=3)
    return before, after


def test_exact_inverse_keeps_complete_source_commands_and_types_without_importing_candidate():
    before, after = pair()
    originals = copy.deepcopy((before, after))
    result = check_zero_initializer_grouping(SOURCE, before, after)
    assert result["status"] == "verified_source_initializer_grouping"
    assert result["virtual_tensor_names"] == ["seed"]
    assert result["logical_initializer_bytes_removed"] == 84
    assert (before, after) == originals
    assert "independently" in result["scope"]
    assert check_zero_initializer_grouping(SOURCE, before, before)["groups"] == []


@pytest.mark.parametrize("change", ["command", "type", "input_order", "output_order", "host_work", "new_storage"])
def test_unrelated_program_changes_cannot_hide_behind_a_legal_zero_group(change):
    before, after = pair()
    if change == "command":
        after["commands"] = []
    elif change == "type":
        after["tensors"]["a"]["dtype"] = "i32"
    elif change == "input_order":
        after["compiler_pointer_abi"]["arguments"][:2] = ["b", "a"]
    elif change == "output_order":
        after["kernel_abi"]["outputs"] = []
    elif change == "host_work":
        after["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [1, 2]
    else:
        after["tensors"]["new"] = {"shape": [99], "dtype": "i32", "role": "intermediate"}
    with pytest.raises(ValueError):
        check_zero_initializer_grouping(SOURCE, before, after)


@pytest.mark.parametrize("zero", [1, -1])
def test_nonzero_initializer_with_self_consistent_source_hash_still_refuses(zero):
    source = SOURCE.replace("constant 0 : i32", f"constant {zero} : i32")
    before, after = pair(source)
    with pytest.raises(ValueError, match="unproved.*nonzero"):
        check_zero_initializer_grouping(source, before, after)


def test_unknown_source_effect_cannot_self_authorize_a_group():
    source = SOURCE.replace("linalg.generic {", "linalg.generic {unknown_effect=1:i32,")
    before, after = pair(source)
    with pytest.raises(ValueError, match="unproved"):
        check_zero_initializer_grouping(source, before, after)


@pytest.mark.parametrize("shared_scalar", [False, True], ids=["returned_initializer", "shared_scalar_constant"])
def test_closed_group_cannot_absorb_returned_storage_or_a_shared_scalar(shared_scalar):
    result = "tensor<3x7xi32>"
    source = SOURCE.replace(f"-> {result} {{", f"-> ({result},{result}) {{", 1)
    output = "extra" if shared_scalar else "seed"
    if shared_scalar:
        source = source.replace(
            "  return %r:", '  %extra = "tensor.splat"(%z) : (i32) -> tensor<3x7xi32>\n  return %r:'
        )
    source = source.replace(
        f"return %r:{result}", f"return %r,%{'extra' if shared_scalar else 'init'}:{result},{result}"
    )
    before, _ = pair(source)
    plan = before["params"]["global_program_plan"]
    if shared_scalar:
        before["tensors"][output] = {"shape": [3, 7], "dtype": "i32", "role": "output"}
        before["kernel_abi"]["args"].append({"tensor": output, "access": "write"})
        before["compiler_pointer_abi"]["arguments"].append(output)
        plan["source_values"].append({"op_index": 3, "result_index": 0, "tensor": output})
        plan["tasks"].append(
            dict(
                task_index=3,
                source_op_indices=[3],
                kind="host",
                reads=[],
                writes=[output],
                instruction_start=4,
                instruction_end=5,
            )
        )
        plan.update(source_op_count=4, epilogue_instruction_range=[5, 6], schedule_instruction_count=6)
    else:
        before["tensors"][output]["role"] = "output"
    before["kernel_abi"]["outputs"].append(output)
    plan["output_bindings"].append(output)
    after = copy.deepcopy(before)
    new = after["params"]["global_program_plan"]
    fused = new["tasks"][2]
    fused.update(
        task_index=0,
        source_op_indices=[0, 1, 2],
        reads=["a", "b"],
        writes=["seed", "out"],
        instruction_start=1,
        instruction_end=2,
    )
    remaining = new["tasks"][3:]
    if remaining:
        remaining[0].update(task_index=1, instruction_start=2, instruction_end=3)
    new.update(
        tasks=[fused, *remaining],
        epilogue_instruction_range=[2 + len(remaining), 3 + len(remaining)],
        schedule_instruction_count=3 + len(remaining),
    )
    assert validate_mixed_program_plan(source, before)["ok"] is True
    assert validate_mixed_program_plan(source, after)["ok"] is True
    assert check_zero_initializer_grouping(source, before, before)["groups"] == []
    with pytest.raises(ValueError, match="unproved"):
        check_zero_initializer_grouping(source, before, after)

"""The granted OOT kit numbers source operations exactly as whole-model admission does."""

import hashlib
import json
from copy import deepcopy

import pytest

from merlin.targetgen.oot_starterkit.plan import (
    main,
    source_operation_inventory,
    validate_mixed_program_plan,
)

SOURCE = """builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = tensor.empty() : tensor<2xi32>
    %1 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %1 : tensor<2xi32>
  }
}"""
LOWERED = """builtin.module {
  llvm.func @kernel(%0: !llvm.ptr) {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %0[%1]
      {merlin.global_task = 0 : i64, merlin.source_op_index = 1 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}"""


def _buffer():
    return {
        "tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "result", "access": "write"}],
            "outputs": ["result"],
        },
        "params": {
            "global_program_plan": {
                "schema": "mixed_program_plan_v1",
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                "source_op_count": 2,
                "tasks": [
                    {
                        "task_index": 0,
                        "kind": "host",
                        "source_op_indices": [0, 1],
                        "instruction_start": 1,
                        "instruction_end": 2,
                        "reads": [],
                        "writes": ["result"],
                    }
                ],
                "schedule_instruction_count": 3,
                "prologue_instruction_range": [0, 1],
                "epilogue_instruction_range": [2, 3],
                "entry_bindings": [],
                "source_values": [{"op_index": 1, "result_index": 0, "tensor": "result"}],
                "output_bindings": ["result"],
                "compiler_temporaries": [],
            }
        },
    }


def _split_output_tasks(*, writers: tuple[int, ...]) -> dict:
    cb = _buffer()
    plan = cb["params"]["global_program_plan"]
    plan["tasks"] = [
        {
            "task_index": index,
            "kind": "host",
            "source_op_indices": [index],
            "instruction_start": index + 1,
            "instruction_end": index + 2,
            "reads": [],
            "writes": ["result"] if index in writers else [],
        }
        for index in range(2)
    ]
    plan["schedule_instruction_count"] = 4
    plan["epilogue_instruction_range"] = [3, 4]
    return cb


def test_exact_direct_source_inventory_includes_init_and_pins_bytes():
    inventory = source_operation_inventory(SOURCE)
    assert inventory["source_op_count"] == 2
    assert [row["operation"] for row in inventory["operations"]] == ["tensor.empty", "arith.constant"]
    assert inventory["returns"][0]["source"] == {"op_index": 1, "result_index": 0}
    assert inventory["source_sha256"] != source_operation_inventory(SOURCE + "\n")["source_sha256"]
    renamed = source_operation_inventory(SOURCE.replace("@forward", "@model_entry"))
    assert renamed["entry"] == "model_entry"
    assert renamed["source_op_count"] == 2


@pytest.mark.parametrize("declaration", ["tensor<3xi32>", "tensor<2xi64>", "(tensor<2xi32>, tensor<2xi32>)", "()"])
def test_source_return_signature_disagreement_refuses_inventory_and_plan(declaration):
    from xdsl.utils.exceptions import VerifyException

    source = SOURCE.replace("@forward() -> tensor<2xi32>", "@forward() -> " + declaration)
    cb = _buffer()
    cb["params"]["global_program_plan"]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(VerifyException, match="function output types"):
        source_operation_inventory(source)
    result = validate_mixed_program_plan(source, cb)
    assert not result["ok"]
    assert "function output types" in " ".join(result["findings"])


def test_ordered_returns_keep_original_arguments_repeats_and_result_indices():
    source = """module {
      func.func @forward(%original: tensor<2xi32>) ->
        (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
        %first, %second = "independent.pair"(%original) :
          (tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
        func.return %second, %original, %first, %second, %original :
          tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
      }
    }"""
    inventory = source_operation_inventory(source)
    assert inventory["arguments"] == [{"shape": [2], "dtype": "i32"}]
    assert inventory["source_op_count"] == 1
    assert [row["source"] for row in inventory["returns"]] == [
        {"op_index": 0, "result_index": 1},
        {"arg_index": 0},
        {"op_index": 0, "result_index": 0},
        {"op_index": 0, "result_index": 1},
        {"arg_index": 0},
    ]
    assert all(row["shape"] == [2] and row["dtype"] == "i32" for row in inventory["returns"])
    reversed_inventory = source_operation_inventory(
        source.replace(
            "func.return %second, %original, %first, %second, %original",
            "func.return %original, %second, %first, %original, %second",
        )
    )
    assert reversed_inventory["arguments"] == inventory["arguments"]
    assert reversed_inventory["returns"] == [inventory["returns"][i] for i in [1, 0, 2, 4, 3]]


def test_actual_empty_return_remains_a_structural_zero_result_inventory():
    inventory = source_operation_inventory("module { func.func @forward() { func.return } }")
    assert inventory["arguments"] == [] and inventory["returns"] == []
    assert inventory["source_op_count"] == 0


def test_public_preflight_accepts_structural_plan_without_claiming_execution():
    result = validate_mixed_program_plan(SOURCE, _buffer(), LOWERED)
    assert result["ok"], result["findings"]
    assert result["scope"] == "public structural preflight only"
    cb = _buffer()
    del cb["params"]["global_program_plan"]["compiler_temporaries"]
    assert validate_mixed_program_plan(SOURCE, cb, LOWERED)["ok"]


def test_output_writer_requires_one_task_owning_its_exact_source_result():
    assert validate_mixed_program_plan(SOURCE, _split_output_tasks(writers=(1,)))["ok"]
    for writers in ((), (0,), (0, 1)):
        result = validate_mixed_program_plan(SOURCE, _split_output_tasks(writers=writers))
        assert not result["ok"]
        assert "output writer" in " ".join(result["findings"])


def test_non_output_temporary_writer_does_not_need_output_result_ownership():
    cb = _split_output_tasks(writers=(1,))
    cb["tensors"]["temporary"] = {"shape": [2], "dtype": "i32", "role": "intermediate"}
    cb["kernel_abi"]["args"].append({"tensor": "temporary", "access": "readwrite"})
    plan = cb["params"]["global_program_plan"]
    plan["compiler_temporaries"] = [
        {"tensor": "temporary", "source_op_index": 0, "source_result_index": 0, "purpose": "temporary storage"}
    ]
    plan["tasks"][0]["writes"].append("temporary")
    assert validate_mixed_program_plan(SOURCE, cb)["ok"]


def test_public_preflight_rejects_misnumbering_duplicate_region_and_crossing():
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [1, 1]
    assert not validate_mixed_program_plan(SOURCE, cb)["ok"]
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [1]
    assert "ownership" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["reads"] = ["missing"]
    assert "absent tensor" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])
    cb = _buffer()
    values = cb["params"]["global_program_plan"]["source_values"]
    values.append(dict(values[0]))
    assert "duplicate source binding" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])


def test_public_preflight_rejects_byte_drift_and_wrong_lowered_owner():
    cb = _buffer()
    assert not validate_mixed_program_plan(SOURCE + "\n", cb)["ok"]
    assert not validate_mixed_program_plan(
        SOURCE, cb, LOWERED.replace("merlin.global_task = 0", "merlin.global_task = 1")
    )["ok"]


def test_public_preflight_rejects_orphan_constant_task_markers():
    lowered = """builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %anchor = "llvm.mlir.constant"() <{value = 0 : i64}>
          {merlin.global_task = 0 : i64} : () -> i64
        llvm.return
      }
    }"""
    result = validate_mixed_program_plan(SOURCE, _buffer(), lowered)
    assert not result["ok"]
    assert "planned task emits no owned computation or control flow" in result["findings"]


def test_public_preflight_rejects_optional_path_only_task():
    lowered = """builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %condition = llvm.mlir.constant(true) : i1
        llvm.cond_br %condition, ^work, ^exit {merlin.global_task = -1 : i64}
      ^work:
        %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i32
        llvm.br ^exit {merlin.global_task = 0 : i64}
      ^exit:
        llvm.return
      }
    }"""
    result = validate_mixed_program_plan(SOURCE, _buffer(), lowered)
    assert not result["ok"]
    assert "a returning CFG path bypasses an entire planned task" in result["findings"]


def test_public_preflight_reads_declared_physical_shape_without_claiming_encoding_proof():
    cb = _buffer()
    cb["tensors"]["result"]["shape"] = [1, 2]
    cb["params"]["storage_encodings"] = {
        "result": {
            "schema": "grouped_axes_storage_v1",
            "logical_shape": [2],
            "physical_shape": [1, 2],
            "dtype": "i32",
            "axis_groups": [[], [0]],
            "strides_elements": [2, 1],
            "storage_elements": 2,
            "offset_elements": 0,
        }
    }
    assert validate_mixed_program_plan(SOURCE, cb)["ok"]
    cb["params"]["storage_encodings"]["result"]["logical_shape"] = [3]
    assert not validate_mixed_program_plan(SOURCE, cb)["ok"]


def test_compiler_temporary_keeps_exact_source_result_dtype():
    cb = _buffer()
    cb["tensors"]["tmp"] = {"shape": [2], "dtype": "i32", "role": "intermediate"}
    cb["kernel_abi"]["args"].insert(0, {"tensor": "tmp", "access": "readwrite"})
    plan = cb["params"]["global_program_plan"]
    plan["compiler_temporaries"] = [
        {"tensor": "tmp", "source_op_index": 0, "source_result_index": 0, "purpose": "source result staging"}
    ]
    plan["tasks"][0]["writes"].append("tmp")
    correct = validate_mixed_program_plan(SOURCE, cb)
    assert correct["ok"], correct["findings"]
    cb["tensors"]["tmp"]["dtype"] = "f32"
    assert "dtype" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])


def test_cli_inventory_and_validation(tmp_path, capsys):
    source = tmp_path / "capsule.interface.mlir"
    source.write_text(SOURCE)
    cb = tmp_path / "command_buffer.json"
    cb.write_text(json.dumps(_buffer()))
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text(LOWERED)
    assert main(["inventory", "--source", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["source_op_count"] == 2
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb), "--lowered-mlir", str(lowered)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    bad = deepcopy(_buffer())
    bad["params"]["global_program_plan"]["source_op_count"] = 1
    cb.write_text(json.dumps(bad))
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb)]) == 1
    capsys.readouterr()
    wrong_writer = _split_output_tasks(writers=(0,))
    cb.write_text(json.dumps(wrong_writer))
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb)]) == 1
    assert "output writer lacks exact source-result task ownership" in json.loads(capsys.readouterr().out)["findings"]


def test_cli_explains_unowned_hoisted_constant_without_changing_refusal(tmp_path, capsys):
    source = tmp_path / "capsule.interface.mlir"
    source.write_text(SOURCE)
    cb = tmp_path / "command_buffer.json"
    cb.write_text(json.dumps(_buffer()))
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text(
        """builtin.module {
          llvm.func @kernel(%p: !llvm.ptr) {
            %anchor = "llvm.mlir.constant"() <{value = 0 : i64}>
              {merlin.global_task = 0 : i64} : () -> i64
            llvm.return
          }
        }"""
    )
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb), "--lowered-mlir", str(lowered)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert "planned task emits no owned computation or control flow" in result["findings"]
    assert any("hoisted constant" in row and "source-owned" in row for row in result["authoring_guidance"])

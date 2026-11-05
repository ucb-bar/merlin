"""Composition preflight joins original source/ABI/storage owners, not execution."""

import hashlib
import json
import struct

import pytest
from merlin_experiments.phase0 import development_composition_preflight as C

SOURCE = """module {
  func.func @forward(%weight: tensor<2xi8>, %input: tensor<2xi8>) -> (tensor<2xi8>, tensor<2xi8>) {
    %first = arith.addi %weight, %input : tensor<2xi8>
    %second = arith.subi %first, %weight : tensor<2xi8>
    func.return %first, %second : tensor<2xi8>, tensor<2xi8>
  }
}"""
# These task-owned stores exercise structural correspondence only. They are
# deliberately not a numerical implementation and cannot issue a certificate.
LOWERED = """module {
  llvm.func @kernel(%w: !llvm.ptr, %x: !llvm.ptr, %y: !llvm.ptr, %z: !llvm.ptr) {
    %a = llvm.load %w {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64} : !llvm.ptr -> i8
    llvm.store %a, %y {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64} : i8, !llvm.ptr
    %b = llvm.load %y {merlin.global_task = 1 : i64, merlin.source_op_index = 1 : i64} : !llvm.ptr -> i8
    llvm.store %b, %z {merlin.global_task = 1 : i64, merlin.source_op_index = 1 : i64} : i8, !llvm.ptr
    llvm.return
  }
}"""


def fixture(tmp_path):
    plan = {
        "schema": "mixed_program_plan_v1",
        "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        "source_op_count": 2,
        "tasks": [
            {
                "task_index": 0,
                "kind": "host",
                "source_op_indices": [0],
                "instruction_start": 1,
                "instruction_end": 2,
                "reads": ["arg0", "arg1"],
                "writes": ["Y0"],
            },
            {
                "task_index": 1,
                "kind": "host",
                "source_op_indices": [1],
                "instruction_start": 2,
                "instruction_end": 3,
                "reads": ["Y0", "arg0"],
                "writes": ["Y1"],
            },
        ],
        "schedule_instruction_count": 4,
        "prologue_instruction_range": [0, 1],
        "epilogue_instruction_range": [3, 4],
        "entry_bindings": ["arg0", "arg1"],
        "source_values": [
            {"op_index": 0, "result_index": 0, "tensor": "Y0"},
            {"op_index": 1, "result_index": 0, "tensor": "Y1"},
        ],
        "output_bindings": ["Y0", "Y1"],
        "compiler_temporaries": [],
    }
    cb = {
        "tensors": {
            name: {"shape": [2], "dtype": "i8", "role": "input" if name.startswith("arg") else "output"}
            for name in ("arg0", "arg1", "Y0", "Y1")
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": name, "access": "read" if name.startswith("arg") else "write"}
                for name in ("arg0", "arg1", "Y0", "Y1")
            ],
            "outputs": ["Y0", "Y1"],
        },
        "params": {"global_program_plan": plan},
    }
    paths = {name: tmp_path / name for name in ("source", "command_buffer", "lowered", "weight_manifest", "weights")}
    paths["source"].write_text(SOURCE)
    paths["lowered"].write_text(LOWERED)
    paths["command_buffer"].write_text(json.dumps(cb))
    paths["weight_manifest"].write_text(
        json.dumps(
            {
                "0": {"kind": "param", "weight": "w", "shape": [2], "dtype": "int8"},
                "1": {"kind": "input", "name": "input"},
            }
        )
    )
    write_weights(paths["weights"], {"w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}, b"\xfe\x03")
    return paths


def write_weights(path, header, payload):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def check(paths, **kwargs):
    return C.check(
        **paths, **{"row_pitch_elements": 4, "max_storage_bytes": 1000, "max_artifact_bytes": 1 << 20, **kwargs}
    )


def test_complete_connected_multioutput_storage_is_more_than_weight_payload(tmp_path):
    paths = fixture(tmp_path)
    result = check(paths)
    assert result["status"] == "public_composition_build_preflight_pass", result
    assert result["source_operations"] == 2 and result["complete_argument_count"] == 4
    assert result["actual_captured_parameter_payload_bytes"] == 2
    assert result["compiled_const_bytes"] > 2 and result["compiled_mutable_bytes"] > 0
    assert result["output_bindings"] == ["Y0", "Y1"]
    assert result["numerical_equivalence"] == result["accelerator_execution"] == "UNVERIFIED"
    # The input and both outputs cannot disappear from a parameter-only budget.
    refused = check(paths, max_storage_bytes=2)
    assert refused["status"] == "refused" and "complete compiled" in refused["findings"][0]
    assert check(paths, max_storage_bytes=result["complete_compiled_storage_bytes"])["status"] != "refused"
    assert check(paths, max_storage_bytes=result["complete_compiled_storage_bytes"] - 1)["status"] == "refused"


def test_original_shared_leaf_cli_returns_only_structural_preflight(tmp_path, capsys):
    paths = fixture(tmp_path)
    argv = [value for name, path in paths.items() for value in ("--" + name.replace("_", "-"), str(path))]
    assert (
        C.main([*argv, "--row-pitch-elements", "4", "--max-storage-bytes", "1000", "--max-artifact-bytes", "1048576"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["formal_admission"] is False and result["timing"] == "UNKNOWN"
    assert result["source_returns"][0]["source"] == {"op_index": 0, "result_index": 0}
    assert result["source_returns"][1]["source"] == {"op_index": 1, "result_index": 0}


@pytest.mark.parametrize("change", ["source_bytes", "dropped_operation", "missing_liveout"])
def test_original_source_and_complete_return_owner_still_refuse(tmp_path, change):
    paths = fixture(tmp_path)
    cb = json.loads(paths["command_buffer"].read_bytes())
    if change == "source_bytes":
        paths["source"].write_text(SOURCE + "\n")
    elif change == "dropped_operation":
        cb["params"]["global_program_plan"]["tasks"][1]["source_op_indices"] = []
    else:
        cb["params"]["global_program_plan"]["output_bindings"] = ["Y0"]
    paths["command_buffer"].write_text(json.dumps(cb))
    result = check(paths)
    assert result["status"] == "refused" and not result["original_public_plan"]["ok"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_argument",
        "shape",
        "shape_bool",
        "input_shadow",
        "dtype",
        "missing_parameter",
        "extra_parameter",
        "truncated",
        "trailing_bytes",
        "offset_bool",
    ],
)
def test_complete_parameter_namespace_and_original_storage_are_checked(tmp_path, change):
    paths = fixture(tmp_path)
    manifest = json.loads(paths["weight_manifest"].read_bytes())
    header = {"w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}
    payload = b"\xfe\x03"
    if change == "missing_argument":
        del manifest["1"]
    elif change == "shape":
        manifest["0"]["shape"] = [1, 2]
    elif change == "shape_bool":
        manifest["0"]["shape"] = [True, 2]
    elif change == "input_shadow":
        manifest["1"]["weight"] = "w"
    elif change == "dtype":
        header["w"]["dtype"] = "U8"
    elif change == "missing_parameter":
        header = {}
    elif change == "extra_parameter":
        header["foreign"] = {"dtype": "I8", "shape": [1], "data_offsets": [2, 3]}
        payload += b"\x04"
    elif change == "truncated":
        payload = payload[:1]
    elif change == "trailing_bytes":
        payload += b"\x04"
    else:
        header["w"]["data_offsets"] = [False, 2]
    paths["weight_manifest"].write_text(json.dumps(manifest))
    write_weights(paths["weights"], header, payload)
    assert check(paths)["status"] == "refused"


def test_declared_shared_parameter_still_keeps_all_pointer_uses(tmp_path):
    paths = fixture(tmp_path)
    manifest = json.loads(paths["weight_manifest"].read_bytes())
    manifest["1"] = dict(manifest["0"])
    paths["weight_manifest"].write_text(json.dumps(manifest))
    result = check(paths)
    assert result["status"] == "public_composition_build_preflight_pass", result
    assert result["unique_captured_parameters"] == 1 and len(result["parameter_uses"]) == 2
    assert result["pointer_order"] == ["arg0", "arg1", "Y0", "Y1"]
    assert result["accelerator_execution"] == "UNVERIFIED"


def test_no_source_symlink_or_mid_preflight_artifact_mutation(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    link = tmp_path / "source-link"
    link.symlink_to(paths["source"])
    assert check({**paths, "source": link})["status"] == "refused"
    original = C.bundle_pack.plan

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        paths["weights"].write_bytes(paths["weights"].read_bytes()[:-1] + b"\x04")
        return value

    monkeypatch.setattr(C.bundle_pack, "plan", mutate)
    result = check(paths)
    assert result["status"] == "refused" and "changed during" in result["findings"][0]


@pytest.mark.parametrize("value", [True, 0, -1, 4.0])
def test_no_implicit_or_invalid_target_row_granule(tmp_path, value):
    assert check(fixture(tmp_path), row_pitch_elements=value)["status"] == "refused"

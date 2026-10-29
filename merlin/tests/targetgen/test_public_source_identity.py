"""A mixed-program plan names the source bytes its compiler was given.

Regression from fresh7 (``SY_act_act_i8_aligned``, final core grade, both arms):
the compiler received the public-ABI-enriched copy of the capsule source and
hashed it into ``global_program_plan.source_sha256``, but the oracle scope
validated that plan against the original file. Every enriched whole-program row
with scratch then failed L2 with "plan source_sha256 differs from exact source
bytes" (8 rows in fresh7ref, 14 in fresh7refcca). The fixture below is that
row's exact original source, public metadata and command-buffer plan.
"""

import copy
import hashlib

import pytest
import yaml

from merlin.targetgen.contract.source_ownership import SourceOwnershipError
from merlin.targetgen.public_source_abi import compiler_source, public_abi_source

SOURCE = """builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x32xi8>, %B0: tensor<32x32xi8>) -> (tensor<32x16xi32>) {
    %_component_0_empty = tensor.empty() : tensor<16x32xi32>
    %_component_0_zero = arith.constant 0 : i32
    %_component_0_init = linalg.fill ins(%_component_0_zero : i32) outs(%_component_0_empty : tensor<16x32xi32>) -> tensor<16x32xi32>
    %product = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A0, %B0 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%_component_0_init : tensor<16x32xi32>) attrs = {prov.region_id = "component_0", prov.op = "matmul", prov.family = "contraction"} {
      ^bb0(%_component_0_lhs: i8, %_component_0_rhs: i8, %_component_0_old: i32):
        %_component_0_lhs_wide = arith.extsi %_component_0_lhs : i8 to i32
        %_component_0_rhs_wide = arith.extsi %_component_0_rhs : i8 to i32
        %_component_0_product = arith.muli %_component_0_lhs_wide, %_component_0_rhs_wide : i32
        %_component_0_sum = arith.addi %_component_0_old, %_component_0_product : i32
        linalg.yield %_component_0_sum : i32
    } -> tensor<16x32xi32>
    %_component_1_flat = tensor.collapse_shape %product [[0, 1]] : tensor<16x32xi32> into tensor<512xi32>
    %moved = tensor.expand_shape %_component_1_flat [[0, 1]] output_shape [32, 16] : tensor<512xi32> into tensor<32x16xi32>
    func.return %moved : tensor<32x16xi32>
  }
}
"""
# Recorded in the fresh7 row's public-source-abi receipt (source_binding.json).
ORIGINAL_SHA256 = "bb2c61090acfb0b08b7900b9063ea46d9764e8e82098b7e13710d729d7c35c0e"
COMPILED_SHA256 = "d2b859e9c3d090037758ed17ece57a8dc4b2081dcbe85ac8ad5383750fc7d4cb"

INPUTS = [
    {"name": "A0", "role": "input", "shape": [16, 32], "dtype": "i8"},
    {"name": "B0", "role": "input", "shape": [32, 32], "dtype": "i8"},
]
METADATA = {
    "name": "SY_act_act_i8_aligned",
    "label": "public",
    "interface_mlir": "capsule.interface.mlir",
    "linalg_mlir": "capsule.interface.mlir",
    "inputs": INPUTS,
    "component_program": {
        "schema": "merlin.component_program.v2",
        "inputs": INPUTS,
        "outputs": [{"name": "Y0", "value": "moved", "actual_value": "moved", "shape": [32, 16], "dtype": "i32"}],
    },
}


def _task(index, kind, ops, regions, reads, writes):
    return {"task_index": index, "kind": kind, "source_op_indices": ops, "source_region_ids": regions,
            "instruction_start": index, "instruction_end": index + 1, "reads": reads, "writes": writes}


# The candidate's command buffer for that row (tensors, ABI and complete plan).
BUFFER = {
    "tensors": {
        "A0": {"shape": [16, 32], "dtype": "i8", "role": "input"},
        "B0": {"shape": [32, 32], "dtype": "i8", "role": "input"},
        "source3": {"shape": [16, 32], "dtype": "i32", "role": "intermediate"},
        "source4": {"shape": [512], "dtype": "i32", "role": "intermediate"},
        "Y0": {"shape": [32, 16], "dtype": "i32", "role": "output"},
    },
    "kernel_abi": {
        "kind": "whole_program",
        "args": [{"tensor": "A0", "access": "read"}, {"tensor": "B0", "access": "read"},
                 {"tensor": "source3", "access": "write"}, {"tensor": "source4", "access": "write"},
                 {"tensor": "Y0", "access": "write"}],
        "outputs": ["Y0"],
    },
    "params": {"global_program_plan": {
        "schema": "mixed_program_plan_v1",
        "source_sha256": COMPILED_SHA256,
        "source_op_count": 6,
        "tasks": [
            _task(0, "accelerator_dense", [0, 1, 2, 3], ["component_0"], ["A0", "B0"], ["source3"]),
            _task(1, "host_source", [4], [], ["source3"], ["source4"]),
            _task(2, "host_source", [5], [], ["source4"], ["Y0"]),
        ],
        "schedule_instruction_count": 3,
        "prologue_instruction_range": [0, 0],
        "epilogue_instruction_range": [3, 3],
        "entry_bindings": ["A0", "B0"],
        "source_values": [{"op_index": 3, "result_index": 0, "tensor": "source3"},
                          {"op_index": 4, "result_index": 0, "tensor": "source4"},
                          {"op_index": 5, "result_index": 0, "tensor": "Y0"}],
        "output_bindings": ["Y0"],
    }},
}


def _capsule(tmp_path, metadata=METADATA, source=SOURCE):
    (tmp_path / "capsule.interface.mlir").write_text(source, encoding="utf-8")
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return {**copy.deepcopy(metadata), "__dir__": str(tmp_path)}, tmp_path / "capsule.interface.mlir"


def _buffer(source_sha256=COMPILED_SHA256):
    cb = copy.deepcopy(BUFFER)
    cb["params"]["global_program_plan"]["source_sha256"] = source_sha256
    return cb


def test_compiler_source_carries_original_and_received_bytes(tmp_path):
    capsule, source = _capsule(tmp_path)
    selected = compiler_source(capsule, source)
    assert selected.original_bytes == SOURCE.encode() and selected.original_sha256 == ORIGINAL_SHA256
    assert selected.enriched and selected.compiler_sha256 == COMPILED_SHA256
    # The re-derivation and the bytes actually handed to the compiler cannot drift apart.
    with public_abi_source(capsule, source, tmp_path / "generated") as given:
        assert given.read_bytes() == selected.compiler_bytes


def test_oracle_scope_compares_plan_with_the_bytes_the_compiler_received(tmp_path):
    from merlin.targetgen.capsule_lane_evidence import oracle_source_scope

    capsule, _ = _capsule(tmp_path)
    with oracle_source_scope(capsule, _buffer(), None) as mutables:
        assert mutables == ("source3", "source4")


def test_unenriched_source_still_binds_to_its_exact_original_bytes(tmp_path):
    from merlin.targetgen.capsule_lane_evidence import oracle_source_scope

    capsule, source = _capsule(tmp_path, {**METADATA, "label": "hidden"})
    assert not compiler_source(capsule, source).enriched
    with oracle_source_scope(capsule, _buffer(ORIGINAL_SHA256), None) as mutables:
        assert mutables == ("source3", "source4")
    with pytest.raises(SourceOwnershipError, match="source_sha256"):
        with oracle_source_scope(capsule, _buffer(COMPILED_SHA256), None):
            pytest.fail("hidden capsule source is never enriched")


@pytest.mark.parametrize("case", ["original_hash", "changed_source", "changed_metadata", "unselected_bytes"])
def test_binding_still_proves_the_selected_public_source(tmp_path, case):
    from merlin.targetgen.capsule_lane_evidence import oracle_source_scope

    metadata, source, cb = METADATA, SOURCE, _buffer()
    if case == "original_hash":
        # The compiler never saw the original bytes; a claim about them is not like with like.
        cb = _buffer(ORIGINAL_SHA256)
    elif case == "changed_source":
        source = SOURCE.replace("arith.constant 0 : i32", "arith.constant 1 : i32")
    elif case == "changed_metadata":
        metadata = copy.deepcopy(METADATA)
        metadata["component_program"]["outputs"][0]["name"] = "Z0"
    else:
        cb = _buffer(hashlib.sha256(b"candidate-chosen source").hexdigest())
    capsule, _ = _capsule(tmp_path, metadata, source)
    with pytest.raises(SourceOwnershipError, match="source_sha256"):
        with oracle_source_scope(capsule, cb, None):
            pytest.fail("plan bound to bytes other than the selected public source")


def test_metadata_disagreeing_with_loaded_capsule_refuses(tmp_path):
    capsule, source = _capsule(tmp_path)
    capsule["inputs"] = copy.deepcopy(capsule["inputs"])
    capsule["inputs"][0]["name"] = "other"
    with pytest.raises(ValueError, match="loaded capsule"):
        compiler_source(capsule, source)

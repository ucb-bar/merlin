"""Fast compile-only guards for reusable host tensor storage.

These tests execute no target model, simulator, or FireSim job.  They exercise the
compiler's source-owned liveness, storage selection, receipt, and LLVM-dialect ABI.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlir_oot.gemmini_opt import Pipeline, _print


CANDIDATE = Path(__file__).resolve().parents[1]
REPO = CANDIDATE.parents[4]


def _compile(source: Path) -> tuple[dict, str]:
    pipe = Pipeline(source.read_text(encoding="utf-8"), enable_source_conv=True).run()
    assert pipe.declined is None
    assert pipe.plan is not None
    assert pipe.artifact is not None
    return pipe.plan.command_buffer, _print(pipe.artifact)


def test_host_normalization_uses_explicit_workspace_not_tensor_stack() -> None:
    cb, target = _compile(
        REPO / "merlin/contract/capsules/model_slices/"
        "SY_host_only_normalization/capsule.linalg.mlir")
    receipt = cb["params"]["host_storage"]

    assert receipt["schema"] == "whole_program_host_storage_v1"
    assert receipt["workspace_global_bytes"] > 0
    assert receipt["bounded_stack_frame_upper_bound_bytes"] == 0
    assert receipt["allocation_count"] > 0
    assert receipt["largest_owners"]
    assert all(row["storage"] == "reusable_workspace"
               for row in receipt["largest_owners"])
    assert "llvm.alloca" not in target
    assert 'linkage = #llvm.linkage<"internal">' in target
    assert "__merlin_host_workspace" in target
    assert "merlin.host_workspace_bytes" in target


def test_mixed_lane_workspace_does_not_extend_kernel_pointer_abi() -> None:
    cb, target = _compile(
        REPO / "merlin/contract/capsules/model/"
        "M3_host_island_seam_gemmini/capsule.interface.mlir")
    receipt = cb["params"]["host_storage"]
    task_kinds = [task["kind"]
                  for task in cb["params"]["global_program_plan"]["tasks"]]

    assert task_kinds == ["contraction", "host", "contraction"]
    assert receipt["workspace_global_bytes"] > 0
    assert receipt["kernel_abi_pointer_added"] is False
    assert receipt["workspace_linkage"] == "internal_global"
    assert not any(arg["tensor"] == "__merlin_host_workspace"
                   for arg in cb["kernel_abi"]["args"])
    assert "__merlin_host_workspace" in target


def test_tensor_live_across_host_mesh_host_migration_is_explicitly_spilled() -> None:
    cb, target = _compile(
        CANDIDATE / "tests/fixtures/host_value_across_lane_migration.mlir")
    plan = cb["params"]["global_program_plan"]
    receipt = cb["params"]["host_storage"]

    assert [task["kind"] for task in plan["tasks"]] == [
        "contraction", "host", "contraction", "host"]
    assert plan["host_tensor_spills"] == [{
        "tensor": "t1",
        "source_op_index": 5,
        "source_result_index": 0,
        "bytes": 128,
    }]
    assert plan["tasks"][1]["writes"] == ["t1"]
    assert "t1" in plan["tasks"][3]["reads"]
    assert receipt["host_tensor_spills"] == plan["host_tensor_spills"]
    assert receipt["host_tensor_spill_bytes"] == 128
    assert receipt["kernel_abi_pointer_added"] is False
    assert "llvm.alloca" not in target


def test_receipt_is_deterministic_for_identical_source() -> None:
    source = (REPO / "merlin/contract/capsules/model_slices/"
              "SY_host_lane_reduction_f32/capsule.linalg.mlir")
    first_cb, first_target = _compile(source)
    second_cb, second_target = _compile(source)

    assert json.dumps(first_cb["params"]["host_storage"], sort_keys=True) == json.dumps(
        second_cb["params"]["host_storage"], sort_keys=True)
    assert first_target == second_target

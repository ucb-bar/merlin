"""Actual changed IR analysis is descriptive and conservatively invalidated."""

from dataclasses import replace

import pytest

from merlin.perf import pass_observation as OBS
from merlin.transforms.certificates import parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline

SOURCE = """builtin.module {
  func.func @f(%x: i8) -> i8 {
    %zero = arith.constant 0 : i8
    %result = arith.addi %x, %zero : i8
    func.return %result : i8
  }
}"""


def test_actual_ir_pass_counts_and_invalidation_do_not_supply_cycle_prices():
    result = apply_common_pipeline(parse_ir(SOURCE), PipelineOptions(32, reuse_buffers=False), plan_only=True)
    observed = OBS.observe_changes(result.changes, profile_sha256="a" * 64)
    assert observed["ranking_authority"] == "none"
    assert observed["capsule_impact"].startswith("UNKNOWN")
    assert observed["compiler_tool_profile_sha256"] == "a" * 64
    assert len(observed["observer_sha256"]) == 64
    row = observed["passes"][0]
    assert row["static_operation_delta"] == -1
    assert row["after"]["operation_counts"].get("arith.addi", 0) == 0
    assert row["invalidated_analyses"] == list(OBS.ANALYSES)
    assert row["recomputed_analyses"] == [
        "effects",
        "storage",
        "dependencies",
        "aliasing",
        "liveness",
        "logical_pressure",
    ]
    assert row["before"]["max_block_live_values"] > row["after"]["max_block_live_values"]
    assert row["before"]["max_block_peak_known_logical_bytes"] > row["after"]["max_block_peak_known_logical_bytes"]
    assert row["before"]["cycles"] == row["after"]["physical_traffic"] == "UNKNOWN"


def test_disconnected_pipeline_cannot_be_reported_as_one_optimization():
    result = apply_common_pipeline(parse_ir(SOURCE), PipelineOptions(32, reuse_buffers=False), plan_only=True)
    change = result.changes[0]
    disconnected = replace(change, before_ir=SOURCE + "\n")
    with pytest.raises(ValueError, match="connected IR"):
        OBS.observe_changes((change, disconnected), profile_sha256="a" * 64)


def test_unknown_allocations_and_effects_remain_unknown():
    source = """builtin.module {
      func.func @f(%n: index) {
        %a = memref.alloc(%n) : memref<?xi8>
        memref.dealloc %a : memref<?xi8>
        func.return
      }
    }"""
    row = OBS.metrics(source)
    assert row["unknown_allocations"] == 1
    assert row["planned_block_reservation_bytes"] is None
    assert row["storage_unknowns"]
    assert row["physical_traffic"] == "UNKNOWN"
    assert row["unknown_logical_sizes"] > 0
    assert "not physical" in row["pressure_scope"]


def test_ir_and_operation_budgets_are_enforced(monkeypatch):
    with pytest.raises(ValueError, match="byte budget"):
        OBS.metrics(" " * (OBS.MAX_IR_BYTES + 1))
    monkeypatch.setattr(OBS, "MAX_OPERATIONS", 2)
    with pytest.raises(ValueError, match="operation budget"):
        OBS.metrics(SOURCE)

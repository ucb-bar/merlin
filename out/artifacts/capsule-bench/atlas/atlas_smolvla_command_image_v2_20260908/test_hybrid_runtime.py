"""Focused tests for the fail-closed Atlas SmolVLA hybrid scheduler."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from submission.mlir_oot.hybrid_runtime import allocate_intervals


ROOT = Path(__file__).resolve().parent
PLAN_ROOT = ROOT / "whole_capture_plan"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_interval_allocator_respects_inclusive_lifetimes_and_reuses_storage() -> None:
    result = allocate_intervals([
        {"name": "a", "start": 0, "end": 2, "bytes": 33},
        {"name": "b", "start": 2, "end": 3, "bytes": 32},
        {"name": "c", "start": 3, "end": 4, "bytes": 16},
        {"name": "d", "start": 5, "end": 5, "bytes": 64},
    ], alignment=32)
    by_name = {row["name"]: row for row in result["allocations"]}
    assert by_name["a"]["offset"] == 0
    assert by_name["b"]["offset"] == 64  # endpoint 2 overlaps a
    assert by_name["c"]["offset"] == 0   # a is dead, b is still live
    assert by_name["d"]["offset"] == 0
    assert result["peak_bytes"] == 96
    assert result["saved_bytes"] > 0


def test_saved_hybrid_schedule_is_complete_ordered_and_fail_closed() -> None:
    schedule = load(PLAN_ROOT / "hybrid_schedule.json")
    assert schedule["status"] == "e2e_blocked_fail_closed"
    assert schedule["runnable_e2e"] is False
    assert schedule["coverage"] == {
        "bounded_host_regions_implemented": 22,
        "layout_bridge_candidates": 2033,
        "materialized_copy_bridges": 112,
        "partition_host_region_overlap": ["conv_0"],
        "proven_metadata_aliases": 1675,
        "qualified_accelerator_partitions": 3,
        "semantic_host_required_regions": 2430,
        "strided_broadcast_bridges": 246,
        "structural_accelerator_partitions": 391,
    }
    assert schedule["fail_closed"]["missing_host_semantics"] == 2408
    assert schedule["fail_closed"]["unqualified_accelerator_partitions"] == 388
    assert schedule["fail_closed"]["unrealized_layout_bridges"] == 358
    assert schedule["conversion_boundaries"] == {
        "by_conversion": {
            "device_requantize": 88,
            "device_to_host_dequantize": 391,
            "host_to_device_quantize": 694,
            "host_to_device_quantize_bias": 77,
        },
        "count": 1250,
        "qualified": 12,
    }
    assert len(schedule["events"]) == 6104
    assert [row["event_index"] for row in schedule["events"]] == list(range(6104))
    assert schedule["device_activation_arena"]["allocation_count"] == 391
    assert schedule["device_activation_arena"]["reuse_count"] > 0
    assert schedule["device_activation_arena"]["peak_bytes"] < (
        schedule["device_activation_arena"]["naive_no_reuse_bytes"]
    )


def test_bounded_real_chain_replays_host_semantics_and_retains_scoped_evidence() -> None:
    chain = load(PLAN_ROOT / "hybrid_schedule.json")["bounded_chain"]
    assert chain["status"] == "host_replay_matches_retained_qualified_rtl_chain"
    assert chain["partition_path"] == ["atlas_p0243", "atlas_p0244"]
    assert chain["host_bridge_region_count"] == 27
    assert chain["p0244_activation_shape"] == [50, 1440]
    assert chain["p0244_activation_f32_sha256"] == (
        "23457ea06c994031379dcfc4491e866b1f9e5558a2ad4c5ad5a310366145bcce"
    )
    assert chain["retained_results"][1]["dispatches"] == 3
    assert "not whole-model execution" in chain["claim"]


def test_hybrid_schedule_is_byte_stable_across_rebuilds() -> None:
    paths = [PLAN_ROOT / "hybrid_schedule.json", PLAN_ROOT / "hybrid_schedule_summary.json"]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before

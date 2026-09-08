"""Tests for conservative full-graph partition boundaries."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import parse_verified  # noqa: E402
from mlir_oot.full_graph import inventory_full_graph  # noqa: E402


def test_full_capture_partition_inventory_is_fail_closed() -> None:
    capture = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    report = inventory_full_graph(parse_verified(capture.read_text(encoding="utf-8")))
    assert report["logical_regions"] == 4930
    assert report["physical_contractions"] == {
        "batched_matmul": 88,
        "rank2_matmul": 303,
        "total": 391,
    }
    assert report["regions_by_partition_class"] == {
        "host_required": 2430,
        "layout_bridge_candidate": 2033,
        "mesh_candidate": 467,
    }
    assert report["regions_by_semantic_capability"]["host_required"] == 2131
    assert report["host_required_breakdown"] == {
        "effective_total": 2430,
        "emitter_shape_or_dtype_refused": 299,
        "no_semantic_emitter": 2131,
    }
    assert report["candidate_island_count"] == 698
    assert sum(
        island["region_count"] for island in report["candidate_islands"]
    ) == 2500
    assert report["host_required_by_semantic"]["dtype_cast"] == 472
    assert report["host_required_by_semantic"]["mul"] == 536

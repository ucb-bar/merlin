#!/usr/bin/env python3
"""Offline fail-closed verifier for the census and paired L2 receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> None:
    coverage = _load("coverage.json")
    receipt = _load("l2_receipt.json")
    census = coverage["capture_census"]
    families = census["families"]
    assert families == {
        "attention": 84, "contraction": 1457, "elementwise_map": 5780, "movement": 3469,
        "normalization": 1341, "reduction": 915, "unclassified": 140,
    }
    assert sum(families.values()) == census["total_regions"] == 13186
    assert census["classified_regions"] == 13046
    comparison = coverage["search_comparison"]
    assert len(comparison["pre_wire_14_capsules"]) == 14
    assert comparison["pre_wire_family_counts"] == {"contraction": 14}
    assert len(comparison["post_wire_15_capsules"]) == 15
    assert comparison["post_wire_family_counts"] == {"contraction": 14, "elementwise_map": 1}
    assert coverage["wired_probe"]["exact_occurrences_in_declared_captures"] == 99
    assert coverage["fail_closed"]["application_regions_physically_qualified_by_this_audit"] == 0

    positive = _load(receipt["positive"]["receipt"])
    negative = _load(receipt["negative_control"]["receipt"])
    assert _sha(HERE / receipt["positive"]["receipt"]) == receipt["positive"]["receipt_sha256"]
    assert _sha(HERE / receipt["negative_control"]["receipt"]) == receipt["negative_control"]["receipt_sha256"]
    assert positive["numeric"]["status"] == "pass" and positive["numeric"]["mismatch_count"] == 0
    assert negative["numeric"]["status"] == "fail" and negative["numeric"]["mismatch_count"] == 1
    assert positive["oracle"] == negative["oracle"] == {
        "kind": "cyclotron_perf_model", "derived_from_rtl": False,
    }
    assert positive["oracle_scope"] == negative["oracle_scope"] \
        == "L2 Cyclotron functional/performance model; not GSIM and not physical RTL"
    assert _sha(HERE / "l2_positive/command_buffer.json") \
        == _sha(HERE / "l2_negative/command_buffer.json") \
        == receipt["same_public_command_buffer_sha256"]
    assert _sha(HERE / "l2_positive/lowered.llvm.mlir") \
        == _sha(HERE / "l2_negative/lowered.llvm.mlir") \
        == receipt["same_lowered_llvm_mlir_sha256"]
    print("PASS: exact census and fail-capable L2 receipt pair verified; no GSIM/physical claim")


if __name__ == "__main__":
    verify()

"""Fail-closed regressions for deterministic whole-capture partition planning."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN_ROOT = ROOT / "whole_capture_plan"


def test_whole_capture_plan_compiles_only_real_structural_contractions() -> None:
    subprocess.run([sys.executable, str(ROOT / "build_partition_plan.py")],
                   check=True, capture_output=True, text=True, timeout=60)
    plan = json.loads((PLAN_ROOT / "partition_plan.json").read_text())
    assert plan["partition_count"] == 391
    assert plan["source_semantics"] == {
        "addmm": 77,
        "batch_matmul": 88,
        "convolution_im2col_matmul": 1,
        "matmul": 225,
    }
    assert plan["kernel_variant_count"] == 28
    assert plan["compile_coverage"] == {
        "capture_semantics_executable_partitions": 0,
        "imem_fit_structural_partitions": 391,
        "structural_partitions_total": 391,
        "unique_kernel_variants_fitting_imem": 28,
        "unique_kernel_variants_total": 28,
    }
    assert plan["structurally_lowerable_partition_count"] == 391
    assert plan["capture_semantics_executable_partition_count"] == 0
    assert all(p["image"]["fits_imem"] for p in plan["partitions"])
    assert all(p["image"]["command_count"] > 0 for p in plan["partitions"])
    assert all(p["structurally_lowerable"] for p in plan["partitions"])
    assert not any(p["capture_semantics_executable"] for p in plan["partitions"])
    assert plan["host_required"] == {
        "effective_total": 2430,
        "emitter_shape_or_dtype_refused": 299,
        "no_semantic_emitter": 2131,
    }


def test_dependencies_lifetimes_and_abis_are_stable_and_explicit() -> None:
    plan = json.loads((PLAN_ROOT / "partition_plan.json").read_text())
    assert len(plan["accelerator_dependency_edges"]) == 88
    assert plan["maximal_accelerator_island_count"] == 303
    assert sum(i["partition_count"] for i in plan["maximal_accelerator_islands"]) == 391
    first = plan["partitions"][1]
    assert first["capture_regions"] == ["matmul_0", "add_3"]
    assert first["kernel_id"] == "matmul_1024_768_768_bias"
    assert [v["origin"]["kind"] for v in first["abi"]["inputs"]] == [
        "host_region",
        "function_argument",
        "function_argument",
    ]
    assert first["abi"]["outputs"][0]["consumers"][0]["region_id"] == "view_6"
    assert first["abi"]["outputs"][0]["frontier_consumers"] == [{
        "kind": "accelerator_partition",
        "op_index": 255,
        "partition_id": "atlas_p0004",
        "region_id": "matmul_3",
    }]
    lifetimes = json.loads((PLAN_ROOT / "lifetime_manifest.json").read_text())
    assert len(lifetimes["outputs"]) == 391
    assert all(row["last_frontier_use_op_index"] >= row["definition_op_index"]
               for row in lifetimes["outputs"])


def test_no_host_region_is_silently_promoted_to_a_command_image() -> None:
    plan = json.loads((PLAN_ROOT / "partition_plan.json").read_text())
    planned_regions = {r for p in plan["partitions"] for r in p["capture_regions"]}
    assert len(planned_regions) == 468
    assert "dtype_cast_0" not in planned_regions
    assert "layer_norm_0" not in planned_regions
    assert "softmax_0" not in planned_regions
    assert plan["limitations"][1].startswith("host-required regions retain capture semantics")


def test_plan_and_manifests_are_byte_stable_across_rebuilds() -> None:
    names = (
        "partition_plan.json",
        "dependency_manifest.json",
        "lifetime_manifest.json",
        "abi_manifest.json",
    )

    def hashes() -> dict[str, str]:
        return {
            name: hashlib.sha256((PLAN_ROOT / name).read_bytes()).hexdigest()
            for name in names
        }

    subprocess.run([sys.executable, str(ROOT / "build_partition_plan.py")],
                   check=True, capture_output=True, text=True, timeout=60)
    before = hashes()
    subprocess.run([sys.executable, str(ROOT / "build_partition_plan.py")],
                   check=True, capture_output=True, text=True, timeout=60)
    assert hashes() == before

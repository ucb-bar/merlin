#!/usr/bin/env python3
"""Inventory capture-rooted and qualified-predecessor Atlas partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "whole_capture_plan/partition_plan.json"
OUTPUT = ROOT / "capture_binding_inventory.json"
QUALIFIED = ("atlas_p0098", "atlas_p0243", "atlas_p0244")
DIRECT_BINDABLE = ("atlas_p0098", "atlas_p0243")
ZERO_COMPUTE_BRIDGES = {
    "tensor.expand_shape", "tensor.collapse_shape", "tensor.cast",
    "linalg.transpose", "linalg.copy",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_inventory() -> dict:
    plan = load(PLAN)
    direct_roots = []
    direct_bindable = []
    direct_blocked = []
    qualified_edges = []
    for partition in plan["partitions"]:
        origins = [entry["origin"] for entry in partition["abi"]["inputs"]]
        if all(origin["kind"] == "function_argument" for origin in origins):
            direct_roots.append(partition["partition_id"])
            bridges = [bridge for origin in origins for bridge in origin.get("bridges", [])]
            unsupported = [
                bridge for bridge in bridges
                if bridge.get("execution") == "host_preprocess"
                or bridge.get("op") not in ZERO_COMPUTE_BRIDGES
            ]
            if unsupported:
                direct_blocked.append({
                    "partition_id": partition["partition_id"],
                    "fqn": partition["fqn"],
                    "reason": "capture-rooted but requires non-view host preprocessing",
                    "unsupported_bridges": unsupported,
                })
            else:
                direct_bindable.append(partition["partition_id"])
        predecessor_inputs = [
            origin for origin in origins
            if origin["kind"] == "accelerator_partition"
            and origin.get("partition_id") in QUALIFIED
        ]
        if predecessor_inputs:
            qualified_edges.append({
                "partition_id": partition["partition_id"],
                "qualified_predecessors": sorted({
                    origin["partition_id"] for origin in predecessor_inputs
                }),
            })

    if direct_roots != ["atlas_p0000", "atlas_p0098", "atlas_p0243"]:
        raise ValueError("capture-rooted partition inventory changed")
    if direct_bindable != list(DIRECT_BINDABLE):
        raise ValueError("directly bindable partition inventory changed")
    if qualified_edges:
        raise ValueError("a new direct qualified-to-accelerator edge needs explicit review")
    return {
        "schema": "atlas_smolvla_capture_binding_inventory_v1",
        "claim": "binding inventory only; no whole-model or additional numeric qualification claim",
        "structural_partitions_total": plan["partition_count"],
        "qualified_capture_semantics": list(QUALIFIED),
        "qualified_capture_semantics_count": len(QUALIFIED),
        "all_inputs_rooted_at_capture_arguments": direct_roots,
        "directly_bindable_without_host_computation": direct_bindable,
        "capture_rooted_but_blocked": direct_blocked,
        "direct_accelerator_successors_of_qualified_partitions": qualified_edges,
        "explicit_host_bridge_candidate": {
            "predecessor": "atlas_p0243",
            "candidate": "atlas_p0244",
            "host_regions": [
                "iota_38", "compare_2", "dtype_cast_242", "mul_293", "add_198",
                "sub_33", "dtype_cast_243", "mul_294", "sub_34", "select_30",
                "pow_65", "mul_295", "elementwise_2", "mul_296", "mul_297",
                "mul_298", "unsqueeze_198", "unsqueeze_199", "mul_299", "sin_32",
                "cos_32", "cat_50", "dtype_cast_244", "unsqueeze_200", "expand_147",
                "cat_51", "view_789",
            ],
            "status": "qualified_as_three_alias_free_n_slice_dispatches",
            "device_dispatches": [[0, 256], [256, 512], [512, 720]],
            "full_image_blocker": (
                "the adopted GSIM harness masks DRAM addresses to 1 MiB; the unsliced "
                "command buffer spans 1,182,240 bytes and would silently alias"
            ),
            "calibration_headroom": (
                "each slice uses explicit E4M3 code_cap=16 to keep quant-domain partial "
                "sums inside the demonstrated accumulator range"
            ),
            "result": "capture_semantics_action_time_mlp_in/result.json",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = build_inventory()
    encoded = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != encoded:
            raise ValueError("saved capture binding inventory is stale")
    else:
        OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

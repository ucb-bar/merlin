#!/usr/bin/env python3
"""Build the fail-closed whole-capture hybrid schedule and bounded-chain witness."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
PLAN_ROOT = ROOT / "whole_capture_plan"
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import parse_verified  # noqa: E402
from mlir_oot.hybrid_runtime import build_hybrid_schedule  # noqa: E402
from mlir_oot.host_semantics import HostSemanticLane, array_sha256  # noqa: E402
from run_capture_partition import (  # noqa: E402
    PARTITIONS,
    _load_capture_values,
    _select_partition,
    sha256_bytes,
)


QUALIFIED = frozenset(binding["partition_id"] for binding in PARTITIONS.values())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_chain_witness() -> tuple[dict, set[str]]:
    """Replay the real host bridge and bind it to retained qualified RTL outputs."""
    binding = PARTITIONS["action_time_mlp_in"]
    partition = _select_partition(binding)
    activation, _, _, source, _ = _load_capture_values(partition, CAPTURE, binding)
    calibration_path = ROOT / binding["output_dir"] / "calibration.json"
    result_path = ROOT / binding["output_dir"] / "result.json"
    calibration = load(calibration_path)
    result = load(result_path)
    source_input = source["input"]
    saved_input = calibration["source_tensors"]["input"]
    actual_hash = sha256_bytes(activation.astype("<f4", copy=False).tobytes())
    if actual_hash != saved_input["device_chain_sha256"]:
        raise ValueError("replayed bounded host bridge differs from saved p0244 calibration")
    if not result.get("acceptance", {}).get("passed"):
        raise ValueError("p0244 retained qualification no longer passes")

    bridge_regions = list(source_input["host_bridge"]["capture_regions"])
    origin = partition["abi"]["inputs"][0]["origin"]
    bridge_regions.append(origin["region_id"])
    bridge_regions.extend(item["region_id"] for item in origin.get("bridges", []))
    bridge_regions = list(dict.fromkeys(bridge_regions))
    receipts = [load(ROOT / path) for path in result["raw_gsim_receipts"]]
    if not all(r.get("assertion_clean") and r.get("stderr_observation") == "empty"
               for r in receipts):
        raise ValueError("p0244 retained dispatch evidence is not assertion-clean")
    return ({
        "schema": "atlas_bounded_real_hybrid_chain_v1",
        "status": "host_replay_matches_retained_qualified_rtl_chain",
        "claim": (
            "fresh deterministic host-bridge replay joined to retained RTL evidence; "
            "device partitions were not rerun and this is not whole-model execution"
        ),
        "partition_path": ["atlas_p0243", "atlas_p0244"],
        "host_bridge_regions": bridge_regions,
        "host_bridge_region_count": len(bridge_regions),
        "host_bridge_f32_sha256": source_input["host_bridge"]["raw_sha256"],
        "p0244_activation_shape": list(activation.shape),
        "p0244_activation_f32_sha256": actual_hash,
        "saved_calibration": {
            "path": calibration_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(calibration_path),
        },
        "retained_results": [
            {
                "partition_id": "atlas_p0243",
                "path": source_input["predecessor"]["result"],
                "sha256": source_input["predecessor"]["result_sha256"],
                "device_output_sha256": source_input["predecessor"][
                    "device_output_sha256"
                ],
            },
            {
                "partition_id": "atlas_p0244",
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(result_path),
                "device_output_sha256": result["device_output"]["raw_sha256"],
                "dispatches": len(receipts),
                "cycles": result["cycles"],
            },
        ],
        "events": [
            "retained_assertion_clean_atlas_p0243",
            "device_to_host_dequantize",
            "execute_27_region_host_bridge",
            "host_to_device_quantize",
            "retained_assertion_clean_atlas_p0244_three_dispatches",
            "device_to_host_dequantize",
        ],
    }, set(bridge_regions))


def generic_host_chain_witness(workload) -> dict:
    """Execute a capture-discovered, dependency-carrying pointwise chain twice."""
    lane = HostSemanticLane(workload)
    region_ids = lane.discover_contiguous_chain()
    first_values = lane.seed_external_values(region_ids)
    seeds = [
        {
            "ordinal": index,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
        for index, value in enumerate(first_values.values())
    ]
    first_outputs = []
    for region_id in region_ids:
        value = lane.execute(region_id, first_values)
        output_record = {
            "region_id": region_id,
            "semantic": lane.programs[region_id].semantic,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
            "finite": bool(value.dtype == np.bool_ or np.all(np.isfinite(value))),
        }
        if value.dtype == np.bool_:
            output_record["true_elements"] = int(np.count_nonzero(value))
        first_outputs.append(output_record)

    second_values = lane.seed_external_values(region_ids)
    second_hashes = []
    for region_id in region_ids:
        second_hashes.append(array_sha256(lane.execute(region_id, second_values)))
    first_hashes = [row["sha256"] for row in first_outputs]
    if second_hashes != first_hashes:
        raise ValueError("fresh host semantic chain is not deterministic")

    produced = set()
    dependency_edges = 0
    for region_id in region_ids:
        program = lane.programs[region_id]
        dependency_edges += sum(value in produced for value in program.generic.inputs)
        produced.update(value for op in program.operations for value in op.results)
    return {
        "schema": "atlas_real_capture_host_semantic_chain_v1",
        "status": "fresh_numeric_execution_exactly_replayed",
        "claim": "host pointwise execution only; no device execution and not whole-model E2E",
        "selection": (
            "longest consecutive qualified capture-region run with an SSA dependency and "
            "at most 1000000 external input elements"
        ),
        "region_ids": region_ids,
        "semantics": [lane.programs[region_id].semantic for region_id in region_ids],
        "region_count": len(region_ids),
        "dependency_edges": dependency_edges,
        "fresh_input_rule": "deterministic ordinal-dependent nonzero coordinate pattern",
        "fresh_inputs": seeds,
        "outputs": first_outputs,
        "replay_hashes_equal": True,
    }


def main() -> int:
    source_path = CAPTURE / "model.mlir"
    source = source_path.read_text(encoding="utf-8")
    plan = load(PLAN_ROOT / "partition_plan.json")
    inventory = load(ROOT / "full_capture_partition_inventory.json")
    workload = parse_verified(source)
    chain, bounded_regions = bounded_chain_witness()
    host_chain = generic_host_chain_witness(workload)
    schedule = build_hybrid_schedule(
        workload, plan, inventory,
        qualified_partitions=set(QUALIFIED),
        bounded_host_regions=bounded_regions,
        alignment=32,
    )
    schedule["capture"].update({
        "path": source_path.relative_to(REPO).as_posix(),
        "bytes": len(source.encode()),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
    })
    schedule["bounded_chain"] = chain
    schedule["generic_host_chain"] = host_chain
    schedule["device_activation_arena"]["alignment_source"] = (
        "the existing Atlas command-buffer allocator's 32-byte tensor-base alignment"
    )
    schedule_path = PLAN_ROOT / "hybrid_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "atlas_hybrid_capture_schedule_summary_v1",
        "status": schedule["status"],
        "runnable_e2e": schedule["runnable_e2e"],
        "claim": schedule["claim"],
        "coverage": schedule["coverage"],
        "fail_closed": {
            key: value for key, value in schedule["fail_closed"].items()
            if not key.endswith("_ids")
        },
        "conversion_boundaries": schedule["conversion_boundaries"],
        "device_activation_arena": {
            key: value for key, value in schedule["device_activation_arena"].items()
            if key != "allocations"
        },
        "event_count": len(schedule["events"]),
        "bounded_chain": chain,
        "generic_host_chain": host_chain,
        "full_schedule": {
            "path": schedule_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(schedule_path),
        },
    }
    summary_path = PLAN_ROOT / "hybrid_schedule_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed whole-capture host/device schedule and activation-arena planning.

This module does not pretend that a structural contraction is executable.  It orders the verified
capture's semantic regions, layout bridges, conversion boundaries, and accelerator partitions, then
marks every event with the evidence needed to execute it.  Missing host semantics or partition
calibration makes the whole schedule non-runnable.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import math

from .frontend import _str_attr
from .full_graph import _partition_class, _tensor_dtype, _tensor_shape
from .host_semantics import HostSemanticLane, LayoutBridgeLane, signature_sha256
from .accelerator_semantics import AcceleratorContractLane, contract_sha256


_PROVEN_ALIAS_SEMANTICS = frozenset({"view", "unsqueeze"})
_STRIDED_ALIAS_SEMANTICS = frozenset({"expand"})
_MATERIALIZED_BRIDGE_SEMANTICS = frozenset({"copy"})


def _align(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def allocate_intervals(intervals: list[dict], *, alignment: int) -> dict:
    """Assign reusable offsets to inclusive live intervals with deterministic first fit."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    ordered = sorted(intervals, key=lambda row: (
        int(row["start"]), int(row["end"]), str(row["name"])
    ))
    active: list[dict] = []
    allocations: list[dict] = []
    peak = 0
    reused = 0
    for row in ordered:
        start, end, size = int(row["start"]), int(row["end"]), int(row["bytes"])
        if start < 0 or end < start or size <= 0:
            raise ValueError(f"invalid live interval {row}")
        # Inclusive endpoints: an output whose last use is event N cannot share storage with a
        # definition at event N.
        active = [item for item in active if int(item["end"]) >= start]
        occupied = sorted(active, key=lambda item: int(item["offset"]))
        offset = 0
        for item in occupied:
            offset = _align(offset, alignment)
            if offset + size <= int(item["offset"]):
                break
            offset = max(offset, int(item["offset"]) + int(item["bytes"]))
        offset = _align(offset, alignment)
        prior_peak = peak
        peak = max(peak, offset + size)
        if offset + size <= prior_peak:
            reused += 1
        allocation = {
            **row,
            "start": start,
            "end": end,
            "bytes": size,
            "offset": offset,
        }
        active.append(allocation)
        allocations.append(allocation)

    # Prove the allocator rather than trusting its construction.
    for index, lhs in enumerate(allocations):
        for rhs in allocations[index + 1:]:
            live_overlap = not (lhs["end"] < rhs["start"] or rhs["end"] < lhs["start"])
            storage_overlap = not (
                lhs["offset"] + lhs["bytes"] <= rhs["offset"]
                or rhs["offset"] + rhs["bytes"] <= lhs["offset"]
            )
            if live_overlap and storage_overlap:
                raise ValueError(f"live allocations overlap: {lhs['name']} and {rhs['name']}")
    naive = sum(_align(int(row["bytes"]), alignment) for row in ordered)
    return {
        "schema": "device_activation_interval_allocation_v1",
        "interval_semantics": "inclusive capture operation ordinals",
        "alignment_bytes": alignment,
        "allocations": allocations,
        "allocation_count": len(allocations),
        "peak_bytes": peak,
        "naive_no_reuse_bytes": naive,
        "saved_bytes": naive - peak,
        "reuse_count": reused,
    }


def _region_inventory(workload) -> tuple[list[dict], int]:
    funcs = [op for op in workload.module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        raise ValueError(f"expected one func.func, found {len(funcs)}")
    block = funcs[0].body.blocks[0]
    rows: OrderedDict[str, dict] = OrderedDict()
    for index, op in enumerate(block.ops):
        region_id = _str_attr(op, "prov.region_id")
        if not region_id:
            continue
        row = rows.setdefault(region_id, {
            "region_id": region_id,
            "semantic": _str_attr(op, "prov.op"),
            "first_op_index": index,
            "last_op_index": index,
            "dtypes": set(),
            "shapes": set(),
            "mlir_ops": Counter(),
        })
        if row["semantic"] != _str_attr(op, "prov.op"):
            raise ValueError(f"inconsistent semantic annotation in {region_id}")
        row["last_op_index"] = index
        row["mlir_ops"][op.name] += 1
        for value in (*op.operands, *op.results):
            dtype = _tensor_dtype(value)
            shape = _tensor_shape(value)
            if dtype is not None:
                row["dtypes"].add(dtype)
            if shape is not None:
                row["shapes"].add(shape)
    result = []
    for row in rows.values():
        category, reason = _partition_class(row["semantic"], row["dtypes"], row["shapes"])
        result.append({
            **row,
            "dtypes": sorted(row["dtypes"]),
            "shapes": [list(shape) for shape in sorted(row["shapes"])],
            "mlir_ops": dict(sorted(row["mlir_ops"].items())),
            "category": category,
            "category_reason": reason,
        })
    return result, len(list(block.ops))


def _alias_status(row: dict) -> tuple[str, bool, str]:
    semantic = row["semantic"]
    op_names = set(row["mlir_ops"])
    sizes = {math.prod(shape) for shape in row["shapes"]}
    if semantic in _PROVEN_ALIAS_SEMANTICS:
        allowed = {"tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"}
        proved = bool(op_names) and op_names <= allowed and len(sizes) == 1
        return (
            "metadata_alias" if proved else "unproven_alias",
            proved,
            "all tensor shapes have equal element count and only reshape/cast ops are present"
            if proved else "reshape/cast alias proof did not hold",
        )
    if semantic in _STRIDED_ALIAS_SEMANTICS:
        return (
            "strided_broadcast_requires_descriptor",
            False,
            "broadcast may be a zero-stride view, but the partition ABI requires contiguous tensors",
        )
    if semantic in _MATERIALIZED_BRIDGE_SEMANTICS:
        return (
            "materialized_copy_required",
            False,
            "copy has value semantics and cannot be erased as an alias",
        )
    return "unproven_alias", False, "layout-bridge semantic has no proof rule"


def _boundary_kind(entry: dict) -> str:
    origin = entry["origin"]["kind"]
    if origin == "accelerator_partition":
        return "device_requantize"
    if entry["device_dtype"] == "bf16":
        return "host_to_device_quantize_bias"
    return "host_to_device_quantize"


def build_hybrid_schedule(
    workload,
    partition_plan: dict,
    full_inventory: dict,
    *,
    qualified_partitions: set[str],
    bounded_host_regions: set[str],
    command_buffers: dict[str, dict],
    alignment: int,
) -> dict:
    """Build a complete deterministic schedule skeleton and fail-closed readiness verdict."""
    regions, top_level_ops = _region_inventory(workload)
    host_lane = HostSemanticLane(workload)
    layout_lane = LayoutBridgeLane(workload)
    accelerator_lane = AcceleratorContractLane(
        workload, partition_plan, command_buffers
    )
    by_region = {row["region_id"]: row for row in regions}
    observed_classes = Counter(row["category"] for row in regions)
    if dict(sorted(observed_classes.items())) != full_inventory["regions_by_partition_class"]:
        raise ValueError("live capture region classes differ from the saved inventory")
    if len(regions) != int(full_inventory["logical_regions"]):
        raise ValueError("live capture region count differs from the saved inventory")

    partition_regions: dict[str, str] = {}
    for partition in partition_plan["partitions"]:
        for region_id in partition["capture_regions"]:
            prior = partition_regions.setdefault(region_id, partition["partition_id"])
            if prior != partition["partition_id"]:
                raise ValueError(f"region {region_id} belongs to multiple partitions")
    missing_mesh = sorted(
        row["region_id"] for row in regions
        if row["category"] == "mesh_candidate" and row["region_id"] not in partition_regions
    )
    if missing_mesh:
        raise ValueError(f"mesh candidate regions lack a physical partition: {missing_mesh[:3]}")

    events: list[dict] = []
    aliases = Counter()
    materialized_layout = Counter()
    missing_host = []
    for row in regions:
        if row["category"] == "layout_bridge_candidate":
            source_status, executable, reason = _alias_status(row)
            aliases[source_status] += 1
            signature = layout_lane.signature_for(row["region_id"])
            if not executable and signature is not None:
                executable = True
                status = "host_materialized_layout_bridge"
                reason = (
                    "complete layout signature accepted for distinct contiguous host materialization"
                )
                materialized_layout[signature["materialization"]] += 1
            else:
                status = source_status
            event = {
                "capture_op_index": row["first_op_index"],
                "kind": "layout_bridge",
                "region_id": row["region_id"],
                "semantic": row["semantic"],
                "status": status,
                "executable": executable,
                "reason": reason,
                "source_requirement": source_status,
            }
            if signature is not None:
                event["layout_signature_sha256"] = signature_sha256(signature)
                event["materialization"] = signature["materialization"]
            events.append(event)
        elif row["category"] == "host_required":
            signature = host_lane.signature_for(row["region_id"])
            executable = signature is not None
            if not executable:
                missing_host.append(row["region_id"])
            event = {
                "capture_op_index": row["first_op_index"],
                "kind": "host_region",
                "region_id": row["region_id"],
                "semantic": row["semantic"],
                "status": ("extracted_host_semantics_qualified" if executable
                           else "missing_host_semantics"),
                "executable": executable,
                "reason": (
                    "complete operation signature accepted by the fail-closed host lane"
                    if executable else host_lane.rejections.get(
                        row["region_id"], row["category_reason"]
                    )
                ),
            }
            if signature is not None:
                # Bind the event to the complete extracted signature without
                # duplicating tens of thousands of affine-map lines in the
                # schedule.  The digest is over canonical JSON and is rebuilt
                # from the source capture on every schedule generation.
                event["operation_signature_sha256"] = signature_sha256(signature)
            events.append(event)

    missing_partitions = []
    boundaries: list[dict] = []
    for partition in partition_plan["partitions"]:
        partition_id = partition["partition_id"]
        qualified = partition_id in qualified_partitions
        contract = accelerator_lane.signature_for(partition_id)
        contract_qualified = contract is not None
        if not qualified:
            missing_partitions.append(partition_id)
        capture_index = int(partition["capture_op_index"])
        for input_index, entry in enumerate(partition["abi"]["inputs"]):
            boundary_kind = _boundary_kind(entry)
            conversion_contract_qualified = contract_qualified and (
                boundary_kind != "device_requantize"
                or entry["origin"].get("partition_id") in accelerator_lane.contracts
            )
            boundary = {
                "capture_op_index": capture_index,
                "kind": "conversion_boundary",
                "direction": "to_device",
                "conversion": boundary_kind,
                "partition_id": partition_id,
                "tensor": entry["name"],
                "capture_type": entry["capture_type"],
                "device_dtype": entry["device_dtype"],
                "device_bytes": entry["device_bytes"],
                "origin": entry["origin"],
                "status": (
                    "qualified" if qualified else
                    "conversion_semantics_qualified_pending_physical_partition"
                    if conversion_contract_qualified else "missing_conversion_contract"
                ),
                "executable": qualified,
                "conversion_semantics_qualified": conversion_contract_qualified,
                "input_index": input_index,
            }
            events.append(boundary)
            boundaries.append(boundary)
        partition_event = {
            "capture_op_index": capture_index,
            "kind": "accelerator_partition",
            "partition_id": partition_id,
            "capture_regions": partition["capture_regions"],
            "kernel_id": partition["kernel_id"],
            "status": ("qualified_capture_semantics" if qualified
                       else "static_command_contract_qualified_pending_capture_numeric"
                       if contract_qualified else "structural_only_unqualified"),
            "executable": qualified,
            "command_contract_qualified": contract_qualified,
        }
        if contract is not None:
            partition_event["command_contract_sha256"] = contract_sha256(contract)
        else:
            partition_event["command_contract_rejection"] = (
                accelerator_lane.rejections[partition_id]
            )
        events.append(partition_event)
        output = partition["abi"]["outputs"][0]
        boundary = {
            "capture_op_index": int(partition["lifetime"]["definition_op_index"]),
            "kind": "conversion_boundary",
            "direction": "to_host",
            "conversion": "device_to_host_dequantize",
            "partition_id": partition_id,
            "tensor": output["name"],
            "capture_type": output["capture_type"],
            "device_dtype": output["device_dtype"],
            "device_bytes": output["device_bytes"],
            "status": (
                "qualified" if qualified else
                "conversion_semantics_qualified_pending_physical_partition"
                if contract_qualified else "missing_conversion_contract"
            ),
            "executable": qualified,
            "conversion_semantics_qualified": contract_qualified,
        }
        events.append(boundary)
        boundaries.append(boundary)

    phase_order = {
        ("conversion_boundary", "to_device"): 0,
        ("accelerator_partition", ""): 1,
        ("conversion_boundary", "to_host"): 2,
        ("layout_bridge", ""): 3,
        ("host_region", ""): 4,
    }
    events.sort(key=lambda event: (
        int(event["capture_op_index"]),
        phase_order[(event["kind"], event.get("direction", ""))],
        str(event.get("partition_id", "")),
        int(event.get("input_index", -1)),
        str(event.get("region_id", "")),
    ))
    for index, event in enumerate(events):
        event["event_index"] = index

    intervals = [{
        "name": partition["partition_id"] + ":Y0",
        "partition_id": partition["partition_id"],
        "start": int(partition["lifetime"]["definition_op_index"]),
        "end": int(partition["lifetime"]["last_frontier_use_op_index"]),
        "bytes": int(partition["abi"]["outputs"][0]["device_bytes"]),
    } for partition in partition_plan["partitions"]]
    arena = allocate_intervals(intervals, alignment=alignment)

    host_count = observed_classes["host_required"]
    alias_count = observed_classes["layout_bridge_candidate"]
    overlap = sorted(set(partition_regions) & {
        row["region_id"] for row in regions if row["category"] == "host_required"
    })
    blocking_aliases = sum(
        event["kind"] == "layout_bridge" and not event["executable"]
        for event in events
    )
    failures = {
        "missing_host_semantics": len(missing_host),
        "unqualified_accelerator_partitions": len(missing_partitions),
        "unqualified_accelerator_command_contracts": len(accelerator_lane.rejections),
        "unrealized_layout_bridges": blocking_aliases,
        "missing_physical_event_runtime": 1,
    }
    missing_host_by_semantic = Counter(by_region[region_id]["semantic"] for region_id in missing_host)
    runnable = not any(failures.values())
    prior_bounded = sum(
        row["category"] == "host_required" and row["region_id"] in bounded_host_regions
        for row in regions
    )
    qualified_by_semantic = Counter(
        program.semantic for region_id, program in host_lane.programs.items()
        if by_region.get(region_id, {}).get("category") == "host_required"
    )
    layout_shape_classes: OrderedDict[tuple, dict] = OrderedDict()
    for region_id, program in layout_lane.programs.items():
        signature = program.signature
        map_key = tuple(
            (item["kind"], item.get("position"), item.get("value"))
            for item in signature["input_map"]
        )
        key = (
            signature["semantic"], signature["dtype"],
            tuple(signature["input_shape"]), tuple(signature["output_shape"]),
            map_key, signature["materialization"],
        )
        entry = layout_shape_classes.setdefault(key, {
            "semantic": signature["semantic"],
            "dtype": signature["dtype"],
            "input_shape": signature["input_shape"],
            "output_shape": signature["output_shape"],
            "input_map": signature["input_map"],
            "materialization": signature["materialization"],
            "count": 0,
            "region_ids": [],
        })
        entry["count"] += 1
        entry["region_ids"].append(region_id)
    return {
        "schema": "atlas_hybrid_capture_schedule_v1",
        "status": "e2e_runnable" if runnable else "e2e_blocked_fail_closed",
        "claim": "deterministic schedule and symbolic allocation; not whole-model execution",
        "runnable_e2e": runnable,
        "capture": {
            "top_level_operations": top_level_ops,
            "logical_regions": len(regions),
        },
        "coverage": {
            "structural_accelerator_partitions": len(partition_plan["partitions"]),
            "qualified_accelerator_partitions": len(qualified_partitions),
            "static_command_contract_partitions_qualified": len(accelerator_lane.contracts),
            "semantic_host_required_regions": host_count,
            "host_signature_regions_implemented": host_count - len(missing_host),
            "host_signature_regions_by_semantic": dict(sorted(qualified_by_semantic.items())),
            "previous_bounded_host_regions_implemented": prior_bounded,
            "previous_missing_host_semantics": host_count - prior_bounded,
            "missing_host_semantics_reduction": (
                (host_count - prior_bounded) - len(missing_host)
            ),
            "layout_bridge_candidates": alias_count,
            "proven_metadata_aliases": aliases["metadata_alias"],
            "strided_broadcast_bridges": aliases["strided_broadcast_requires_descriptor"],
            "materialized_copy_bridges": aliases["materialized_copy_required"],
            "host_materialized_layout_bridges": sum(materialized_layout.values()),
            "host_materialized_layout_bridges_by_rule": dict(sorted(materialized_layout.items())),
            "qualified_layout_bridges": alias_count - blocking_aliases,
            "partition_host_region_overlap": overlap,
        },
        "fail_closed": {
            **failures,
            "missing_host_semantics_by_semantic": dict(sorted(missing_host_by_semantic.items())),
            "missing_host_region_ids": missing_host,
            "unqualified_partition_ids": missing_partitions,
        },
        "conversion_boundaries": {
            "count": len(boundaries),
            "by_conversion": dict(sorted(Counter(
                boundary["conversion"] for boundary in boundaries
            ).items())),
            "qualified": sum(boundary["executable"] for boundary in boundaries),
            "semantics_qualified": sum(
                boundary["conversion_semantics_qualified"] for boundary in boundaries
            ),
        },
        "accelerator_contract_census": accelerator_lane.census(),
        "device_activation_arena": arena,
        "layout_bridge_census": {
            "shape_class_count": len(layout_shape_classes),
            "shape_classes": list(layout_shape_classes.values()),
            "by_semantic": dict(sorted(Counter(
                program.semantic for program in layout_lane.programs.values()
            ).items())),
            "by_materialization": dict(sorted(Counter(
                program.signature["materialization"]
                for program in layout_lane.programs.values()
            ).items())),
        },
        "events": events,
    }

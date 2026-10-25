"""Compact same-ROI optimization diagnostics from existing public projections.

No execution, model prices, physical counters or qualification are produced here.
The caller retains custody of the original source geometry and observation.
"""

from __future__ import annotations

import math

from .scoped_resources import SCHEMA as RESOURCE_SCHEMA

SCHEMA = "merlin.same_roi_optimization_feedback.v1"


def _count(value):
    return value if type(value) is int and 0 <= value < 2**64 else None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def project(performance):
    """Summarize one producer-bound source roofline and scoped resource projection.

    Spike-derived model cycles remain modeled. A tier's cycle count or a
    whole-trace booking is never substituted for this exact counter-PC window.
    This is descriptive arithmetic over projections, not independent custody.
    """
    result = dict(
        schema=SCHEMA,
        status="UNKNOWN",
        scope_sha256=None,
        model_sha256=None,
        elf_sha256=None,
        cycles=dict(value=None, source_kind="unknown", unit="model_cycles", measured_cycles=None),
        work=dict(
            source_useful_macs=None,
            observed_shape_macs=None,
            nominal_padded_macs=None,
            useful_fraction_of_nominal_padded_work=None,
            structural_peak_macs_per_cycle=None,
            structural_compute_floor_cycles=None,
            compute_ceiling_attainment=None,
            structural_floor_to_modeled_cycles=None,
            source_kind="unknown",
            meaning=("source useful MACs / (structural peak MACs per cycle * modeled ROI cycles); "
                     "not measured array occupancy"),
        ),
        traffic=[],
        commands={},
        host_instruction_counts={},
        counts_source_kind="unknown",
        reservation_status="UNKNOWN",
        reservations=[],
        overlaps=[],
        physical=dict(
            source_kind="unknown",
            array_occupancy=None,
            bandwidth=None,
            dependency_stall_cycles=None,
            accelerator_queue_pressure=None,
            storage_byte_highwater=None,
            host_preparation_cycles=None,
            host_dispatch_cycles=None,
            host_synchronization_wait_cycles=None,
        ),
        suggestions=[],
        resource_cycles_additive=False,
        qualification_issued=False,
    )
    scoped = performance.get("scoped_resource_feedback") or {}
    cycles = _count(scoped.get("estimated_cycles_same_window"))
    pins = ("scope_sha256", "model_sha256", "elf_sha256")
    valid = (
        scoped.get("schema") == RESOURCE_SCHEMA
        and scoped.get("status") == "modeled"
        and scoped.get("source_kind") == "modeled"
        and cycles is not None
        and cycles > 0
        and cycles == performance.get("estimated_whole_soc_cycles")
        and all(_sha(scoped.get(key)) for key in pins)
        and all(scoped[key] == performance.get(key) for key in pins)
    )
    if not valid:
        result["reason"] = (
            "Matching complete scoped resource feedback and modeled ROI cycles are unavailable; "
            "whole-trace and other tier cycles are excluded."
        )
        return result
    result.update(status="modeled", **{key: scoped[key] for key in pins})
    result["cycles"].update(value=cycles, source_kind="modeled")
    roofline = performance.get("structural_roofline") or {}
    compute = scoped.get("compute") or {}
    useful = _count(roofline.get("useful_macs"))
    observed = _count(compute.get("useful_macs"))
    padded = _count(compute.get("nominal_padded_macs"))
    peak = _count(roofline.get("structural_peak_macs_per_cycle"))
    floor = _number(roofline.get("structural_compute_floor_cycles"))
    result["work"].update(
        source_useful_macs=useful,
        observed_shape_macs=observed,
        nominal_padded_macs=padded,
        useful_fraction_of_nominal_padded_work=_ratio(useful, padded),
        structural_peak_macs_per_cycle=peak if peak else None,
        structural_compute_floor_cycles=floor,
        compute_ceiling_attainment=_ratio(useful, peak * cycles if peak else None),
        structural_floor_to_modeled_cycles=_ratio(floor, cycles),
        source_kind="source_derived_and_observed_counts" if useful is not None else "source_work_unknown",
    )
    result["work"]["model_below_structural_floor"] = floor > cycles if floor is not None else None
    logical = roofline.get("logical_request_ideal_service") or {}
    for row in (scoped.get("traffic") or [])[:8]:
        size = _count(row.get("bytes"))
        direction = {"device_requested_read": "read", "device_requested_write": "write"}.get(row.get("boundary"))
        demand = _count((logical.get(direction) or {}).get("requested_bytes")) if direction else None
        result["traffic"].append(
            dict(
                boundary=str(row.get("boundary", "unknown"))[:96],
                bytes=size,
                bytes_per_modeled_cycle=_ratio(size, cycles),
                requested_bytes_per_source_useful_mac=_ratio(size, useful) if direction else None,
                requested_to_source_logical_ratio=_ratio(size, demand),
                source_kind=row.get("source_kind", "unknown"),
                physical_bandwidth=False,
            )
        )
    commands = scoped.get("commands") or {}
    result["commands"] = {
        key: _count(commands.get(key)) for key in ("total", "config", "load", "preload", "compute", "store")
    }
    result["commands"]["target_fences"] = _count(commands.get("fences"))
    result["commands"]["host_fences"] = _count(
        (scoped.get("host", {}).get("instruction_counts") or {}).get("cpu.fence")
    )
    result["counts_source_kind"] = "observed_same_window_instructions_and_requests"
    # Keep primary instruction classes: alphabetical truncation mostly exposed
    # zero counters and branch subdivisions while hiding integer/load/fence work.
    host_counts = scoped.get("host", {}).get("instruction_counts") or {}
    primary = [(key, value) for key, value in host_counts.items()
               if key.startswith("cpu.") and key.count(".") == 1
               and key not in {"cpu.branch_taken", "cpu.branch_not_taken"} and _count(value) is not None]
    primary.sort(key=lambda row: (-row[1], row[0]))
    result["host_instruction_counts"] = dict(primary[:8])
    result["host_instruction_classes_omitted"] = max(0, len(primary) - 8)
    if scoped.get("activity_status") == "modeled":
        result["reservation_status"] = "modeled"
        resources = sorted(
            scoped.get("resources") or [], key=lambda row: -(_number(row.get("capacity_normalized_activity")) or 0)
        )
        for row in resources[:8]:
            result["reservations"].append(
                dict(
                    resource_id=row.get("resource_id"),
                    name=str(row.get("name", "unknown"))[:96],
                    reservation_active_fraction=_number(row.get("active_fraction")),
                    capacity_reservation_fraction=_number(row.get("capacity_normalized_activity")),
                    modeled_average_waiting_entries=_number(row.get("average_modeled_waiting_entries")),
                    modeled_queue_active_fraction=_ratio(_count(row.get("queue_active_cycles")), cycles),
                    peak_generic_queue_entries=_count(row.get("peak_queue_entries")),
                    source_kind="modeled",
                )
            )
        result["reservation_rows_omitted"] = max(0, len(resources) - 8)
        result["overlaps"] = [
            {key: row.get(key) for key in ("left", "right", "active_cycles", "fraction_of_window", "source_kind")}
            for row in (scoped.get("overlaps") or [])[:4]
        ]
    result["reservation_meaning"] = (
        "Capacity reservations and generic start queues; not physical utilization, ROB occupancy or stalls. "
        "Zero generic queue entries do not establish absent hardware queue pressure."
    )
    suggestions = result["suggestions"]
    if useful is not None and padded is not None and padded > useful:
        suggestions.append(
            dict(
                opportunity="reduce_nominal_padded_work",
                capability="source-exact tiling or tail transformation",
                required_proof="Preserve source arithmetic, output types, aliases and whole-call completion.",
            )
        )
    if result["commands"].get("target_fences") or result["commands"].get("host_fences"):
        suggestions.append(
            dict(
                opportunity="inspect_synchronization_frequency",
                capability="dependency and completion aware scheduling",
                required_proof="Keep every required operand lifetime and real target completion boundary.",
            )
        )
    if any((row["requested_to_source_logical_ratio"] or 0) > 1 for row in result["traffic"]):
        suggestions.append(
            dict(
                opportunity="inspect_requested_traffic_amplification",
                capability="source and capacity aware reuse or layout transformation",
                required_proof=("Prove source aliases, buffer lifetimes and target capacity; "
                                "requested bytes are not physical traffic."),
            )
        )
    if any((row["modeled_average_waiting_entries"] or 0) > 0 for row in result["reservations"]):
        suggestions.append(
            dict(
                opportunity="inspect_modeled_resource_contention",
                capability="legal dependency ordering and staging",
                required_proof=("Preserve selected target issue/completion semantics; "
                                "modeled queues do not prove hardware stalls."),
            )
        )
    result["objective"] = (
        "Compare complete same-ROI latency under the same evaluator; diagnostic ratios are not objectives to maximize."
    )
    return result

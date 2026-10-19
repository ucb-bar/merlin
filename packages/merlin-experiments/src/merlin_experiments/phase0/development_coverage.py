"""Exact public source coverage and conditional logical resource accounting."""

from __future__ import annotations

import math


def resources(description, basis):
    """Price source payload only; neither physical placement nor a full floor."""
    if basis is None:
        return {"status": "UNKNOWN", "reason": "no selected facts-derived capacity basis"}
    if not isinstance(basis, dict) or any(
        type(basis.get(key)) is not int or basis[key] < 1
        for key in (
            "tile_rows",
            "tile_cols",
            "operand_bytes",
            "accumulate_bytes",
            "operand_store_bytes",
            "accumulator_bytes",
            "peak_macs_per_cycle",
        )
    ):
        raise ValueError("logical resource accounting requires the original positive facts basis")
    operands = sum(math.prod(row["shape"]) for row in description["inputs"])
    wide = sum(math.prod(row["shape"]) for row in description["outputs"] if row["dtype"] == "int32")
    narrow = sum(math.prod(row["shape"]) for row in description["outputs"] if row["dtype"] == "int8")
    operand_bytes = operands * basis["operand_bytes"]
    accumulator_bytes = wide * basis["accumulate_bytes"]
    return {
        "status": "conditional_logical_payload_only",
        "input_elements": operands,
        "wide_output_elements": wide,
        "narrow_output_elements": narrow,
        "logical_operand_bytes": operand_bytes,
        "logical_accumulator_output_bytes": accumulator_bytes,
        "operand_store_ratio": operand_bytes / basis["operand_store_bytes"],
        "accumulator_store_ratio": accumulator_bytes / basis["accumulator_bytes"],
        "simultaneous_liveness_proven": False,
        "physical_transfer_granularity": "UNKNOWN",
        "padded_work": "UNKNOWN until an emitted schedule is checked",
        "roofline": {
            "status": "UNKNOWN",
            "conditional_useful_compute_component_cycles": math.ceil(
                description.get("integer_scalar_products", description["scalar_products"])
                / basis["peak_macs_per_cycle"]
            ),
            "missing": [
                "physical_transfers",
                "issue",
                "host_packing_layout_and_arithmetic",
                "legal_overlap",
                "readout",
                "emitted_schedule_and_measured_cycles",
            ],
            "attainment": None,
        },
    }


def matrix(members):
    rows = []
    for member in members:
        request = member["request"]
        declaration = request.get("declaration", {})
        capacity = request.get("capacity", {})
        description = member.get("description", {})
        rows.append(
            {
                "ordinal": member["ordinal"],
                "identity": member["request_sha256"],
                "family": request.get("family", capacity.get("family", declaration.get("pattern", "unavailable"))),
                "regime": capacity.get("regime", request.get("basis", "explicit_public_development")),
                "role": request.get("role", "development_performance"),
                "pattern": declaration.get("pattern"),
                "geometry": declaration.get("geometry", request.get("shape")),
                "layout": request.get("layout", declaration.get("pattern")),
                "purposes": request.get("purposes", [capacity["exercises"]] if "exercises" in capacity else []),
                "mechanism_basis": capacity.get("why", request.get("basis")),
                "original_capacity_working_set": capacity.get("working_set"),
                "source_sha256": member.get("loader_identity", {}).get("sha256"),
                "reference_sha256": member.get("reference_identity", {}).get("sha256"),
                "numeric_policy": member.get("numeric_policy", declaration.get("arithmetic")),
                "required_accelerator_work": member.get("required_accelerator_work"),
                "contractions": description.get("contractions"),
                "logical_contractions": description.get("logical_contractions", description.get("contractions")),
                "pending_contract": description.get("pending_contract", []),
                "score_pressure": description.get("score_pressure"),
                "optimization_mechanisms": description.get("optimization_mechanisms", []),
                "source_status": member["status"],
                "reason": member.get("reason"),
                "baseline": member["baseline"],
                "resource_pressure": member.get("resource_pressure", {"status": "UNKNOWN"}),
                "expected_cost": member.get("expected_cost", {"target_cycles": None}),
            }
        )
    return {
        "schema": "merlin.phase0.public_development_source_coverage.v1",
        "scope": "source_reference_coverage_not_accelerator_or_performance_admission",
        "members": rows,
        "unavailable_members": [
            row["identity"] for row in rows if row["source_status"] != "source_and_independent_reference_frozen"
        ],
        "pending_contract_members": [row["identity"] for row in rows if row["pending_contract"]],
        "remaining_global_gates": [
            "private_exclusion",
            "software_and_hardware_correspondence",
            "sealed_capture",
            "baseline_execution",
            "qualified_timing",
            "frozen_objective",
        ],
        "required_pattern_gaps": {
            "complete_reduced_models": "connected source patterns omit full residual/MLP/normalization/softmax graphs",
            "selected_scalar_grouped_readout": "exact target software/readout correspondence remains unadmitted",
            "bias_activation_pooling": "no reviewed exact source adaptation in this new factory roster",
            "queue_config_fence_overlap": "capacity probes require emitted instruction/effect and timing qualification",
            "physical_residency_dataflow": "original facts-derived requests retained; selected schedule not executed",
        },
    }

"""Readiness of actual reference analyses, independent of numerical grading."""

from __future__ import annotations

from collections import Counter

SCHEMA = "merlin.reference_readiness.v1"


def assess(reports):
    """Require useful comparisons for every included accelerator region.

    The operator supplies reports freshly produced by the existing analyzer.
    Static coverage alone is never a ready verdict. An unresolved region stays
    in the denominator rather than disappearing through a relevance guess.
    """
    regions, failures, inputs = [], [], []
    seen_inputs = set()
    for report in reports:
        pin = report.get("input")
        if pin:
            inputs.append(pin)
        identity = (pin or {}).get("path"), (pin or {}).get("sha256")
        if not all(identity) or identity in seen_inputs:
            failures.append(dict(kind="missing_or_duplicate_input", input=pin))
        seen_inputs.add(identity)
        selected = report.get("regions", [])
        ids = [row.get("stable_id", row.get("region_id")) for row in selected]
        if (
            type(report.get("total")) is not int
            or report["total"] != len(selected)
            or any(identifier is None for identifier in ids)
            or len(set(ids)) != len(ids)
        ):
            failures.append(dict(kind="incomplete_or_duplicate_region_coverage", input=pin))
        coverage = report.get("coverage") or {}
        sites = [row.get("accelerator_sites") for row in selected]
        if (
            type(coverage.get("accelerator_sites")) is not int
            or coverage["accelerator_sites"] <= 0
            or coverage["accelerator_sites"] != coverage.get("selected_region_sites")
            or any(type(value) is not int or value < 0 for value in sites)
            or sum(value for value in sites if type(value) is int) != coverage["accelerator_sites"]
        ):
            failures.append(dict(kind="unaccounted_instruction_sites", input=pin))
        for region in selected:
            projection = region.get("mechanism_projection") or {}
            cca = region.get("cca") or {}
            axes = (cca.get("provenance") or {}).get("axis_evidence") or {}
            supported = [
                axis
                for axis, fact in axes.items()
                if fact.get("status") in ("observed", "inferred") and fact.get("evidence")
            ]
            missing = []
            if projection.get("semantic_region_status") != "established":
                missing.append("complete_upstream_semantic_region")
            if projection.get("source_binary_correspondence") != "established":
                missing.append("source_binary_correspondence")
            if not supported:
                missing.append("supported_cca_axes")
            if projection.get("compatible_comparison") != "established":
                missing.append("compatible_reference_candidate_comparison")
            if projection.get("unclassified_instruction_classes"):
                missing.append("selected_isa_instruction_roles")
            regions.append(
                dict(
                    input=pin,
                    region=region.get("stable_id", region.get("region_id")),
                    accelerator_sites=region.get("accelerator_sites"),
                    supported_axes=supported,
                    missing=missing,
                    ready=not missing,
                )
            )
    if not regions:
        failures.append(dict(kind="no_analyzed_regions"))
    gaps = Counter(gap for row in regions for gap in row["missing"])
    ready = bool(regions) and not failures and all(row["ready"] for row in regions)
    return dict(
        schema=SCHEMA,
        status="ready" if ready else "blocked",
        region_count=len(regions),
        ready_regions=sum(row["ready"] for row in regions),
        inputs=inputs,
        failures=failures,
        missing=dict(gaps),
        regions=regions,
        authority="operator diagnostic; original semantic/producer/comparison records must be rechecked",
        performance_qualification=False,
        launch_ready=ready,
    )


def require_ready(document):
    """Reject stale/incomplete shapes as well as a blocked operator report."""
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or document.get("status") != "ready"
        or document.get("launch_ready") is not True
        or not document.get("region_count")
        or document.get("ready_regions") != document.get("region_count")
        or document.get("failures")
        or document.get("missing")
        or not document.get("regions")
        or len(document["regions"]) != document["region_count"]
        or any(row.get("ready") is not True or row.get("missing") for row in document["regions"])
    ):
        raise ValueError("reference analysis has unresolved readiness obligations")
    return document

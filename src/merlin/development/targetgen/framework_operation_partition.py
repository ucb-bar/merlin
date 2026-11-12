"""Frontend-first registry partitions, distinct from lowered provenance fanout."""

from __future__ import annotations

import copy
from collections import Counter

_STAGES = ("original", "quantized", "prepared")


def attach_frontend_partitions(universe: dict, source_trace: dict) -> dict:
    """Use verified graph operator identities without replacing them with decompositions."""
    result = copy.deepcopy(universe)
    lowered_keys = (
        "observed_frontend_operators",
        "observed_registered_aten_operators",
        "unobserved_registered_aten_operators",
        "unlisted_frontend_operators",
        "accelerator_candidate_frontend_operators",
        "unresolved_frontend_operators",
        "count_basis",
    )
    result["lowered_provenance"] = {key: result[key] for key in lowered_keys}
    registered = set(result.get("registered_aten_operators") or [])
    catalog_available = result.get("registered_aten_operators") is not None
    stages = {}
    for stage in _STAGES:
        snapshot = source_trace.get("graphs", {}).get(stage) or {}
        observed = snapshot.get("status") == "verified"
        counts = snapshot.get("by_target") if observed else None
        names = set(counts or {})
        version = (snapshot.get("runtime_versions") or {}).get("torch")
        catalog_matches = catalog_available and version is not None and version == result.get("torch_version")
        stages[stage] = {
            "status": "observed" if observed else "unknown",
            "graph_sha256": snapshot.get("sha256"),
            "torch_version": version,
            "static_call_site_count": snapshot.get("call_count"),
            "operator_counts": copy.deepcopy(counts),
            "operator_names": sorted(names) if observed else None,
            "observed_registered_aten_operators": sorted(names & registered) if observed and catalog_matches else None,
            "unobserved_registered_aten_operators": sorted(registered - names)
            if observed and catalog_matches
            else None,
            "unlisted_frontend_operators": sorted(names - registered) if observed and catalog_matches else None,
            "registry_comparison_status": "matched"
            if catalog_matches
            else "unknown"
            if not version or not catalog_available
            else "version_mismatch",
            "count_basis": "exact static call sites in this captured graph, not lowered annotation occurrences",
        }
    result["stages"] = stages
    original = stages["original"]
    result["frontend_denominator_stage"] = "original" if original["status"] == "observed" else "unknown"
    if original["status"] == "observed":
        result["observed_frontend_operators"] = [
            {"operator": name, "static_call_site_count": count}
            for name, count in sorted(original["operator_counts"].items())
        ]
        for key in (
            "observed_registered_aten_operators",
            "unobserved_registered_aten_operators",
            "unlisted_frontend_operators",
        ):
            result[key] = original[key]
        # Source-call acceleration needs all its lowered obligations, not just a
        # similarly named or partially accelerated decomposed operator.
        result["accelerator_candidate_frontend_operators"] = None
        result["unresolved_frontend_operators"] = None
        result["count_basis"] = original["count_basis"]
    return result


def combine_frontend_partitions(universe: dict, applications: dict) -> dict:
    """Aggregate source stages only when every selected application supplies that stage."""
    graphs = {}
    for stage in _STAGES:
        partitions = [application["framework_universe"]["stages"][stage] for application in applications.values()]
        known = bool(partitions) and all(partition["status"] == "observed" for partition in partitions)
        versions = {partition["torch_version"] for partition in partitions}
        counts = Counter()
        if known:
            for partition in partitions:
                counts.update(partition["operator_counts"])
        graphs[stage] = {
            "status": "verified" if known else "unknown",
            "by_target": dict(sorted(counts.items())) if known else None,
            "call_count": sum(counts.values()) if known else None,
            "runtime_versions": {"torch": next(iter(versions))} if len(versions) == 1 else {},
        }
    return attach_frontend_partitions(universe, {"graphs": graphs})

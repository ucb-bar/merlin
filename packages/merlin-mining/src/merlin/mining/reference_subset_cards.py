"""Bounded observation cards and diversity features for explicitly selected references."""

from __future__ import annotations

import json
import math
from collections import Counter


def known_leaves(value, prefix=""):
    """Keep false and zero; null is absence of a reconstructed fact."""
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            if key != "provenance":
                result.update(known_leaves(item, prefix + "." + key if prefix else key))
        return result
    return {prefix: value} if value is not None else {}


def canonical(analysis):
    return {key: value for key, value in analysis["cca"].items() if key != "provenance"}


def bucket(value):
    return min(max(0, int(value)).bit_length(), 16)


def value_bin(value):
    if type(value) is bool:
        return "bool:" + str(value)
    if type(value) is int:
        return "integer:" + str(-bucket(-value) if value < 0 else bucket(value))
    if isinstance(value, str):
        return "enum:" + value
    if isinstance(value, float) and math.isfinite(value):
        return "float"
    return None


def features(analysis):
    """No candidate, timing, model-gap labels or target-specific field-name rules."""
    projection = analysis["mechanism_projection"]
    result = {"role:" + key for key, n in projection.get("role_sites", {}).items() if n}
    result.update("class:" + key for key, n in projection.get("instruction_classes", {}).items() if n)
    for field in projection.get("field_fingerprints", []):
        label = field["instruction"] + "." + field["field"]
        unresolved, missing = field.get("unresolved_sites", 0), field.get("missing_sites", 0)
        known = field["sites"] - unresolved
        status = "resolved" if not unresolved and not missing else "partial" if known else "unresolved"
        result.add("field:" + label + ":" + status)
        result.add("field:" + label + ":distinct-bin:" + str(bucket(field["distinct_values"])))
    for label, values in projection.get("field_profiles", {}).items():
        for row in values:
            category = value_bin(row["value"])
            if category is not None:
                result.add("field:" + label + ":sample-value-bin:" + category)
    result.add("structure:site-bin:" + str(bucket(analysis["accelerator_sites"])))
    result.add("structure:loop-bin:" + str(bucket(analysis.get("cfg", {}).get("natural_loops", 0))))
    result.add("structure:fence-bin:" + str(bucket(projection.get("fence_sites", 0))))
    result.add("structure:counted-control:" + projection.get("counted_control", {}).get("status", "unknown"))
    return result


def card(occurrence, analysis):
    projection = analysis["mechanism_projection"]
    observations = projection.get("mechanism_observations", {})
    fingerprints = projection.get("field_fingerprints", [])
    return dict(
        occurrence_id=occurrence["id"],
        artifact_id=occurrence["artifact_id"],
        address=occurrence["address"],
        size_bytes=occurrence["size_bytes"],
        accelerator_sites=occurrence["accelerator_sites"],
        boundary_status=occurrence["boundary_status"],
        canonical_facts=known_leaves(analysis["cca"]),
        role_static_sites=projection.get("role_sites", {}),
        instruction_class_static_sites=projection.get("instruction_classes", {}),
        field_count=len(fingerprints),
        field_sites=sum(row["sites"] for row in fingerprints),
        unresolved_field_sites=sum(row.get("unresolved_sites", 0) for row in fingerprints),
        missing_field_sites=sum(row.get("missing_sites", 0) for row in fingerprints),
        field_fingerprint_status=projection.get("field_fingerprint_status"),
        counted_control=projection.get("counted_control", {}),
        schedule_complete=analysis.get("schedule_model", {}).get("complete", False),
        repetition_census=observations.get("repetition_census"),
        invariant_fields=observations.get("invariant_fields", []),
        repeated_operand_groups=observations.get("repeated_operand_groups", []),
        evidence_id=analysis["evidence"]["evidence_id"],
        unknowns=analysis["cca"].get("provenance", {}).get("unknowns", []),
        qualification="static observations, conditional on selected ISA; no numerical or performance qualification",
    )


def coverage(units):
    cards = [item for unit in units for item in unit["cards"]]
    signatures = Counter(json.dumps(item, sort_keys=True) for unit in units for item in unit["canonical"])
    return dict(
        unique_bodies=len(units),
        occurrences=len(cards),
        model_occurrences=dict(sorted(Counter(item["artifact_id"] for item in cards).items())),
        static_accelerator_sites=sum(item["accelerator_sites"] for item in cards),
        raw_code_bytes=sum(unit["size_bytes"] for unit in units),
        canonical_signatures=len(signatures),
        largest_canonical_signature_occurrences=max(signatures.values(), default=0),
        instruction_classes=sorted({name for item in cards for name in item["instruction_class_static_sites"]}),
        roles=sorted({name for item in cards for name in item["role_static_sites"]}),
        observed_features=len(set().union(*(unit["features"] for unit in units))) if units else 0,
        field_sites=sum(item["field_sites"] for item in cards),
        unresolved_field_sites=sum(item["unresolved_field_sites"] for item in cards),
        missing_field_sites=sum(item["missing_field_sites"] for item in cards),
        complete_schedules=sum(bool(item["schedule_complete"]) for item in cards),
        feature_absence="not a proof of semantic absence; projections and value populations can be bounded",
    )


def diversity_order(units):
    """One declared diversity-first curriculum, not a random-subset or cost ablation."""
    remaining = sorted(units, key=lambda unit: unit["kernel_id"])
    selected, seen, models = [], set(), set()
    while remaining:

        def score(unit):
            unseen = unit["features"] - seen
            roles = {name.removeprefix("role:") for name in unit["features"] if name.startswith("role:")}
            return (
                bool(set(unit["models"]) - models),
                bool(roles - {"config", "sync"}),
                sum(name.startswith("role:") for name in unseen),
                sum(name.startswith("class:") for name in unseen),
                sum(name.startswith("field:") for name in unseen),
                sum(name.startswith("structure:") for name in unseen),
                -unit["size_bytes"],
            )

        chosen = max(remaining, key=score)
        chosen["selection_score"] = score(chosen)
        chosen["new_features"] = sorted(chosen["features"] - seen)
        selected.append(chosen)
        seen.update(chosen["features"])
        models.update(chosen["models"])
        remaining.remove(chosen)
    return selected

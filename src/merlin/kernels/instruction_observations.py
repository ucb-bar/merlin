"""Typed, witnessed instruction observations carried by CCA beside semantic facets.

Static field populations establish observations, not source equivalence, storage
lifetimes, dynamic traffic or compiler optimization levers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class OperandFieldObservation:
    instruction: str
    field: str
    sites: int
    missing_sites: int
    unresolved_sites: int
    distinct_values: int | None
    sequence_sha256: str | None
    population_sha256: str | None
    value_set_sha256: str | None
    # Canonical scalar JSON preserves distinctions between false, 0 and 0.0.
    samples: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if (
            not self.instruction
            or not self.field
            or len(self.samples) > 16
            or any(type(n) is not int or n < 0 for n in (self.sites, self.missing_sites, self.unresolved_sites))
            or self.unresolved_sites > self.sites
        ):
            raise ValueError("invalid bounded operand field observation")
        for value, count in self.samples:
            if len(value) > 4096 or type(count) is not int or count < 1:
                raise ValueError("invalid bounded operand sample")
            parsed = json.loads(value)
            if type(parsed) not in (str, int, bool, float):
                raise ValueError("operand samples require recovered scalar values")
            json.dumps(parsed, allow_nan=False)

    def brief(self):
        return dict(
            sites=self.sites,
            missing_sites=self.missing_sites,
            unresolved_sites=self.unresolved_sites,
            distinct_values=self.distinct_values,
            samples=[dict(value=json.loads(value), sites=count) for value, count in self.samples[:4]],
        )


@dataclass(frozen=True)
class InstructionObservations:
    count_basis: str = "listed_instruction_sites"
    evidence_id: str = ""
    decoder_inputs: tuple[tuple[str, str], ...] = ()
    classes: tuple[tuple[str, int], ...] = ()
    roles: tuple[tuple[str, int], ...] = ()
    fields: tuple[OperandFieldObservation, ...] = ()
    field_roster_status: str = "unknown"
    fence_sites: int | None = None
    instruction_class_sequence_sha256: str | None = None
    operand_sequence_sha256: str | None = None
    operand_multiset_sha256: str | None = None
    unknowns: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            len(self.fields) > 1024
            or len(self.classes) > 1024
            or len(self.roles) > 1024
            or len({(f.instruction, f.field) for f in self.fields}) != len(self.fields)
        ):
            raise ValueError("invalid bounded instruction observation roster")


def field_roster_status(projection):
    """``complete`` only when the selected RTL layout roster is fully projected.

    The observed-field budget alone proves only that every decoded field was
    retained; it says nothing about RTL fields the protocol never projects.
    """
    observed = projection.get("field_fingerprint_status", "unknown")
    roster = projection.get("operand_layout_roster")
    if observed != "complete":
        return observed
    if not isinstance(roster, dict) or roster.get("status") not in ("complete", "declared_subset"):
        return "observed_fields_only"
    return roster["status"]


def from_projection(projection, *, evidence_id, unknowns=()):
    """Consume original ISA observer output without interpreting instruction spellings."""
    profiles = projection.get("field_profiles", {})
    fields = tuple(
        OperandFieldObservation(
            instruction=f["instruction"],
            field=f["field"],
            sites=f["sites"],
            missing_sites=f["missing_sites"],
            unresolved_sites=f["unresolved_sites"],
            distinct_values=f["distinct_values"],
            sequence_sha256=f.get("sequence_sha256"),
            population_sha256=f.get("population_sha256"),
            value_set_sha256=f.get("value_set_sha256"),
            samples=tuple(
                (json.dumps(row["value"], sort_keys=True, separators=(",", ":"), allow_nan=False), row["sites"])
                for row in profiles.get(f["instruction"] + "." + f["field"], [])[:4]
            ),
        )
        for f in projection.get("field_fingerprints", [])
    )
    return InstructionObservations(
        count_basis=projection.get("count_basis", "listed_instruction_sites"),
        evidence_id=evidence_id,
        decoder_inputs=tuple(sorted((row["path"], row["sha256"]) for row in projection.get("decoder_inputs", []))),
        classes=tuple(sorted(projection.get("instruction_classes", {}).items())),
        roles=tuple(sorted(projection.get("role_sites", {}).items())),
        fields=fields,
        field_roster_status=field_roster_status(projection),
        fence_sites=projection.get("fence_sites"),
        instruction_class_sequence_sha256=projection.get("instruction_class_sequence_sha256"),
        operand_sequence_sha256=projection.get("operand_sequence_sha256"),
        operand_multiset_sha256=projection.get("operand_multiset_sha256"),
        unknowns=tuple(unknowns),
    )


def compare_observations(reference, candidate, *, limit=8, requested_fields=()):
    """Compare recovered values with explicit missingness and independent authority."""
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("instruction comparison requires a finite bound")
    if (
        not isinstance(requested_fields, (tuple, list))
        or len(requested_fields) > 8
        or any(type(name) is not str or not name or len(name) > 512 for name in requested_fields)
    ):
        raise ValueError("instruction comparison requires bounded field selectors")
    unknown = dict(
        status="unknown",
        compared_fields=0,
        field_differences=[],
        semantic_alignment="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )
    if reference is None or candidate is None or not reference.evidence_id or not candidate.evidence_id:
        return dict(unknown, reason="instruction_observations_unavailable")
    if (
        not reference.decoder_inputs
        or reference.decoder_inputs != candidate.decoder_inputs
        or reference.count_basis != candidate.count_basis
    ):
        return dict(unknown, reason="selected_decoder_or_count_scope_unmatched")
    left = {(f.instruction, f.field): f for f in reference.fields}
    right = {(f.instruction, f.field): f for f in candidate.fields}
    differences, unavailable, compared = [], [], 0
    for key in sorted(left.keys() | right.keys()):
        if requested_fields and ".".join(key) not in requested_fields:
            continue
        a, b = left.get(key), right.get(key)
        usable = (
            a is not None
            and b is not None
            and a.sites > a.unresolved_sites
            and b.sites > b.unresolved_sites
            and (a.value_set_sha256 or a.sequence_sha256)
            and (b.value_set_sha256 or b.sequence_sha256)
        )
        if not usable:
            unavailable.append(dict(instruction=key[0], field=key[1], reason="recovered_values_unavailable"))
            continue
        compared += 1
        reasons = []
        complete = not any((a.missing_sites, a.unresolved_sites, b.missing_sites, b.unresolved_sites))
        if complete and a.value_set_sha256 and b.value_set_sha256 and a.value_set_sha256 != b.value_set_sha256:
            reasons.append("recovered_values_changed")
        elif a.samples and b.samples and {v for v, _ in a.samples} != {v for v, _ in b.samples}:
            reasons.append("observed_value_samples_changed")
        if a.population_sha256 and b.population_sha256 and a.population_sha256 != b.population_sha256:
            reasons.append("recovered_population_changed")
        if (a.sites, a.missing_sites, a.unresolved_sites) != (b.sites, b.missing_sites, b.unresolved_sites):
            reasons.append("site_counts_or_missingness_changed")
        if a.sequence_sha256 and b.sequence_sha256 and a.sequence_sha256 != b.sequence_sha256:
            reasons.append(
                "recovered_sequence_changed"
                if a.value_set_sha256 and b.value_set_sha256
                else "recovered_sequence_changed_population_unknown"
            )
        if reasons:
            differences.append(
                dict(
                    instruction=key[0],
                    field=key[1],
                    reference=a.brief(),
                    candidate=b.brief(),
                    reasons=reasons,
                    completeness="partial_recovered_values"
                    if any((a.missing_sites, a.unresolved_sites, b.missing_sites, b.unresolved_sites))
                    or not (a.value_set_sha256 and b.value_set_sha256)
                    else "complete_observed_field_values",
                    evidence=[reference.evidence_id + "#" + ".".join(key), candidate.evidence_id + "#" + ".".join(key)],
                )
            )
    differences.sort(
        key=lambda row: (
            not any(
                reason in row["reasons"] for reason in ("recovered_values_changed", "observed_value_samples_changed")
            ),
            row["completeness"] != "complete_observed_field_values",
            row["instruction"],
            row["field"],
        )
    )
    available_names = {".".join(key) for key in left.keys() | right.keys()}
    unavailable.extend(
        dict(field=name, reason="requested_field_unobserved")
        for name in requested_fields
        if name not in available_names
    )
    return dict(
        status="compared_observations" if compared else "unknown",
        compared_fields=compared,
        total_fields=compared + len(unavailable),
        requested_fields=list(requested_fields),
        unknown_fields=len(unavailable),
        unknown_field_examples=unavailable[:4],
        field_differences=differences[:limit],
        omitted_field_differences=max(0, len(differences) - limit),
        count_basis=reference.count_basis,
        reference_field_roster=reference.field_roster_status,
        candidate_field_roster=candidate.field_roster_status,
        reference_roles=dict(reference.roles),
        candidate_roles=dict(candidate.roles),
        reference_unknowns=list(reference.unknowns),
        candidate_unknowns=list(candidate.unknowns),
        semantic_alignment="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
        interpretation=(
            "Recovered instruction observations; field differences are not optimization goals or semantic gaps."
        ),
    )


def compact_comparison(comparison):
    """Keep field values visible within the native reply; full evidence stays in detail."""
    result = {key: value for key, value in comparison.items() if key != "field_differences"}
    rows = []
    for original in comparison.get("field_differences", [])[:4]:
        row = dict(original)
        for side in ("reference", "candidate"):
            row[side] = dict(
                original[side],
                samples=[
                    dict(sample, value=(sample["value"][:160] + " [see detail]"), display_truncated=True)
                    if isinstance(sample["value"], str) and len(sample["value"]) > 160
                    else sample
                    for sample in original[side]["samples"][:4]
                ],
            )
        rows.append(row)
    result["field_differences"] = rows
    result["omitted_field_differences"] = comparison.get("omitted_field_differences", 0) + max(
        0, len(comparison.get("field_differences", [])) - len(rows)
    )
    return result

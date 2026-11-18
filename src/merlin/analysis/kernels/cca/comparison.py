"""Multi-level CCA comparator — structured, typed divergences (target-agnostic).

Given an expert CCA and ours (both lifted to the Common Compute Abstraction, ideally at the asm
level — the authoritative substrate), emit ``Divergence`` records: one per populated facet field
that differs. These feed the action catalog (``action_catalog.py``), which maps each to a typed
compiler change. Comparison is per-facet, so a target only diffs the facets it has (vector for
RVV, spatial for a systolic array, …) — nothing RVV-specific here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, get_args, get_type_hints

from merlin.analysis.kernels.cca.abstraction import CCA

# ---- cca_compare -------------------------------------------------------------------------------


@dataclass
class Divergence:
    axis: str  # "compute.contraction_form", "vector.lmul", ...
    expert: Any
    ours: Any
    backend: str
    evidence: list[str] = field(default_factory=list)  # kernel ids justifying the expert value


def _populated_pairs(fa, fb) -> dict[str, tuple]:
    if fa is None or fb is None:
        return {}
    out = {}
    for k, va in asdict(fa).items():
        vb = asdict(fb).get(k)
        if va is not None and vb is not None and va != vb:
            out[k] = (va, vb)
    return out


def uncomparable_axes(expert: CCA, ours: CCA) -> list[tuple[str, str]]:
    """Axes one side populated and the other did NOT -- returned as ``(axis, which_side_is_missing)``.

    ``compare`` deliberately emits no divergence when either side is None: it cannot claim a gap it
    cannot see, and inventing one would be worse. But SILENCE is not the same as "no gap", and the
    difference matters because the loop can only discover what it is shown. This module already knew
    that hazard and patched it for ONE axis -- see the MR-aware special case in ``compare``, added
    because "an UNBLOCKED kernel lifts to register_block=None, so 'expert blocks to MR=k, ours
    doesn't' never surfaced". Every other axis kept the blindness.

    OBSERVED: selecting a dtype-matched int8 expert (correctly, to stop a cross-dtype comparison
    manufacturing fake divergences) took the divergence count 8 -> 5. Two of the three that vanished
    did so because the qd8 fixture lifts ``compute.register_block=None`` and no ``memory`` facet at
    all -- so real gaps on register blocking and access pattern became invisible rather than resolved.
    Trading visible noise for invisible blindness is the wrong trade; this makes the blindness visible.

    Report this next to ``compare``'s output, the way ``build_catalog`` reports unrouted divergences.
    """
    out: list[tuple[str, str]] = []
    for facet in _facet_names():
        fe, fo = getattr(expert, facet, None), getattr(ours, facet, None)
        if fe is None and fo is None:
            continue
        if fe is None or fo is None:
            # the whole facet is missing on one side; name it once rather than per field
            out.append((facet, "expert" if fe is None else "ours"))
            continue
        de, do = asdict(fe), asdict(fo)
        for k in sorted(set(de) | set(do)):
            ve, vo = de.get(k), do.get(k)
            if (ve is None) != (vo is None):
                out.append((f"{facet}.{k}", "expert" if ve is None else "ours"))
    return out


def _facet_names() -> tuple[str, ...]:
    """Facet fields of :class:`CCA`, reflected rather than hardcoded.

    This list used to be a literal, which meant a newly added facet was silently never compared --
    the lift would populate it, the contract would demand a route for it, and `compare` would still
    report no divergence, so the beam would never see it. Reflecting keeps a new facet wired by
    construction. (Found the hard way: `region` was invisible here even after being lifted.)

    A facet is identified STRUCTURALLY -- a field whose declared type is one of the ``*Facet``
    dataclasses -- rather than by subtracting a hardcoded list of non-facets. That subtraction had
    the mirror image of the same bug: adding a plain scalar field to ``CCA`` (``scope``) put it in
    this list, and ``compare`` then tried to read facet fields off a string. A list of what to
    EXCLUDE must be updated by whoever adds a non-facet; deriving what to INCLUDE need not be.
    """
    from dataclasses import fields as _fields
    from dataclasses import is_dataclass

    from merlin.analysis.kernels.cca import abstraction as _cca

    facet_types = {n for n, o in vars(_cca).items() if is_dataclass(o) and n.endswith("Facet")}
    return tuple(f.name for f in _fields(CCA) if any(t in str(f.type) for t in facet_types))


def compare(expert: CCA, ours: CCA, *, evidence: list[str] | None = None) -> list[Divergence]:
    """expert-vs-ours CCA -> typed Divergences (the authoritative gap when both are asm-lifted).

    REFUSES a cross-scope comparison. The same axis name asks a different question at each scope --
    ``dispatch.n_dispatches`` means "launches in this region" at kernel scope and "launches in the
    model" at program scope -- so comparing across scopes yields divergences that are real numbers
    answering different questions, which is worse than no comparison because they look routable.
    """
    es, os_ = getattr(expert, "scope", "kernel"), getattr(ours, "scope", "kernel")
    if es != os_:
        raise ValueError(
            f"refusing to compare a {es!r}-scoped CCA against a {os_!r}-scoped one: the same axis "
            f"means a different question at each scope, so the differences would not be divergences"
        )
    ev = evidence or [expert.provenance.get("source", "expert")]
    backend = (expert.backend or ours.backend or ["?"])[0]
    out: list[Divergence] = []
    for facet in _facet_names():
        for k, (ve, vo) in _populated_pairs(getattr(expert, facet), getattr(ours, facet)).items():
            out.append(Divergence(axis=f"{facet}.{k}", expert=ve, ours=vo, backend=backend, evidence=list(ev)))

    # register_block (MR) is the #1 GEMM data-movement decision and the one we were structurally
    # blind to: _populated_pairs only fires when BOTH sides report it, but an UNBLOCKED kernel lifts
    # to register_block=None (== MR 1, no row reuse), so "expert blocks to MR=k, ours doesn't" never
    # surfaced. Compare MR-aware (None -> MR 1) so the gap becomes a real, routable divergence.
    def _mr(cca):
        rb = getattr(cca.compute, "register_block", None) if cca and cca.compute else None
        if isinstance(rb, (tuple, list)) and rb and isinstance(rb[0], int):
            return rb[0]
        return 1

    emr, omr = _mr(expert), _mr(ours)
    if emr > omr and not any(d.axis == "compute.register_block" for d in out):
        out.append(
            Divergence(
                axis="compute.register_block",
                expert=expert.compute.register_block,
                ours=(ours.compute.register_block if (ours and ours.compute) else None),
                backend=backend,
                evidence=list(ev),
            )
        )
    return out


@dataclass(frozen=True)
class EvidenceComparison:
    """Supported structural comparisons, never an agreement or qualification verdict."""

    status: str
    divergences: tuple[Divergence, ...]
    compared_axes: tuple[str, ...]
    unknowns: tuple[dict, ...]
    advisory_scope: str = "Structural evidence only; semantic, numerical and performance equality are not established."

    @property
    def denominator(self) -> int:
        return len(self.compared_axes)

    def to_dict(self) -> dict:
        return dict(asdict(self), denominator=self.denominator)


_RESIDENCY_AXES = frozenset(
    (
        "memory.onchip_resident",
        "memory.panel_reuse",
        "compute.accumulator_resident",
        "spatial.accumulator_resident",
        "simt.smem_resident",
        "communication.resident_across_calls",
    )
)
_COUNT_AXES = frozenset(
    (
        "dispatch.n_dispatches",
        "dispatch.n_fences",
        "dispatch.config_fraction",
        "dispatch.dma_issue_to_wait",
        "compute.reduction_partition",
    )
)


def _axis_evidence(cca, axis):
    evidence = cca.provenance.get("axis_evidence", {})
    row = evidence.get(axis, {}) if isinstance(evidence, dict) else {}
    return row if isinstance(row, dict) else {}


def _supported(row):
    witnesses = row.get("evidence")
    return (
        row.get("status") in ("observed", "inferred")
        and isinstance(witnesses, (tuple, list))
        and bool(witnesses)
        and all(type(witness) is str and bool(witness.strip()) for witness in witnesses)
    )


def _known_scope(value):
    if isinstance(value, dict):
        return bool(value) and all(type(key) is str and key and _known_scope(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_known_scope(item) for item in value)
    return (
        type(value) in (int, bool)
        or type(value) is str
        and bool(value.strip())
        and value.strip().lower() not in ("unknown", "?")
    )


def compare_evidence(expert: CCA, ours: CCA, *, evidence: list[str] | None = None) -> EvidenceComparison:
    """Compare only populated axes with explicit status, witnesses and compatible scopes.

    Both absent values remain unknown. Residency and dispatch metrics additionally
    require matching per-axis ``axis_evidence[axis]['scope']`` records; a global
    residency label alone cannot prove matching lifetimes or count boundaries.
    Callers retain responsibility for source fusion/numerics, target identity and
    trusted artifact binding. ``compared`` means a nonzero denominator, not equality.
    """
    if type(expert) is not CCA or type(ours) is not CCA:
        raise ValueError("evidence comparison requires existing typed CCAs")
    unknowns, identity_mismatch = [], False
    for key in ("op", "scope"):
        values = (getattr(expert, key), getattr(ours, key))
        missing = tuple(
            side
            for side, value in zip(("expert", "ours"), values)
            if type(value) is not str or value.strip().lower() in ("", "unknown", "?")
        )
        if missing or values[0] != values[1]:
            identity_mismatch |= not missing
            unknowns.append(
                dict(
                    axis=key,
                    reason="identity_unknown" if missing else "identity_mismatch",
                    missing_sides=missing,
                    expert=values[0],
                    ours=values[1],
                )
            )
    problems = (expert.scope_problems(), ours.scope_problems())
    if any(problems):
        unknowns.append(dict(axis="scope", reason="invalid_scope", expert=problems[0], ours=problems[1]))
    blocked = bool(unknowns)
    eligible, witnesses = [], {}
    annotations = get_type_hints(CCA)
    for facet in _facet_names():
        candidates = get_args(annotations[facet]) or (annotations[facet],)
        schema = next(candidate for candidate in candidates if is_dataclass(candidate))
        for item in fields(schema):
            axis = facet + "." + item.name
            values = tuple(getattr(getattr(cca, facet), item.name, None) for cca in (expert, ours))
            rows = tuple(_axis_evidence(cca, axis) for cca in (expert, ours))
            missing = tuple(side for side, value in zip(("expert", "ours"), values) if value is None)
            reason = "value_unavailable" if missing else "comparison_identity_unresolved" if blocked else None
            if reason is None:
                missing = tuple(side for side, row in zip(("expert", "ours"), rows) if not _supported(row))
                reason = "axis_evidence_unavailable" if missing else None
            scopes = tuple(row.get("scope") for row in rows)
            if reason is None:
                required = axis in _RESIDENCY_AXES | _COUNT_AXES or any(scope is not None for scope in scopes)
                missing = tuple(
                    side
                    for side, scope in zip(("expert", "ours"), scopes)
                    if not isinstance(scope, (str, dict)) or not _known_scope(scope)
                )
                if required and missing:
                    reason = "axis_scope_unavailable"
                elif required and scopes[0] != scopes[1]:
                    reason = "axis_scope_mismatch"
                else:
                    legacy = tuple(cca.provenance.get("residency_scope") for cca in (expert, ours))
                    if axis in _RESIDENCY_AXES and all(legacy) and legacy[0] != legacy[1]:
                        reason = "residency_scope_mismatch"
            if reason:
                unknowns.append(
                    dict(axis=axis, reason=reason, missing_sides=missing, expert_scope=scopes[0], ours_scope=scopes[1])
                )
            else:
                eligible.append(axis)
                witnesses[axis] = tuple(dict.fromkeys((*rows[0]["evidence"], *rows[1]["evidence"])))
    divergences = tuple(
        replace(row, evidence=list(witnesses[row.axis]))
        for row in (() if blocked else compare(expert, ours, evidence=evidence))
        if row.axis in witnesses
    )
    status = "mismatch" if identity_mismatch else "compared" if eligible else "unknown"
    return EvidenceComparison(status, divergences, tuple(sorted(eligible)), tuple(unknowns))


# ---- instruction_observations ------------------------------------------------------------------
# Typed, witnessed instruction observations carried by CCA beside semantic facets.
#
# Static field populations establish observations, not source equivalence, storage
# lifetimes, dynamic traffic or compiler optimization levers.


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

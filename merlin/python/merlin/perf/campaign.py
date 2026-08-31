"""Pure, target-agnostic decisions for a performance-capsule campaign.

This module does not discover capsules, run an oracle, or name a machine.  It is the narrow boundary
between those mechanisms and a claim: a declaration plus derived traits determines applicability;
exactly enumerated measurements determine whether the screen ran; a measured negative control shows
that the falsifier was capable of firing; and :mod:`merlin.perf.differential` decides whether the
unpriced work cancels before an expensive tier may be bought.

The distinctions are deliberate.  A refuted trait is a proved inapplicability, an unknown trait is a
missing fact, and a missing emitter is implementation work.  Likewise, a falsifier that ran and did
not fire makes a family ``INERT``; a falsifier that was not observable makes it ``REFUSED``.  None of
those states is a zero score or a pass.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from merlin.perf import differential
from merlin.perf.envelope import Composed
from merlin.perf.falsifier import DID_NOT_RISE, UNDETERMINABLE, ABDecision
from merlin.perf.oracle_cost import CostSample, ProbeKind
from merlin.perf.profile import TRAITS

__all__ = [
    "BLOCKED_UNIMPLEMENTED",
    "COMPLETE",
    "ELIGIBLE",
    "INERT",
    "PROMOTED",
    "REFUSED",
    "SKIPPED_INAPPLICABLE_FALSE",
    "SKIPPED_INAPPLICABLE_UNKNOWN",
    "CampaignDeclarationError",
    "CampaignEvidenceError",
    "FamilyDecision",
    "FamilyState",
    "FalsifierEvidence",
    "Measurement",
    "ReplicaIdentity",
    "assess_eligibility",
    "complete_family",
    "decide_promotion",
    "validate_declaration",
    "validate_measurements",
]


class CampaignDeclarationError(ValueError):
    """A performance-family declaration cannot mean what it claims to mean."""


class CampaignEvidenceError(ValueError):
    """Runtime evidence is absent, duplicated, or insufficient for the requested decision."""


class FamilyState(str, Enum):
    """Every state a family may occupy; skipped and refused are never folded into completion."""

    ELIGIBLE = "eligible"
    SKIPPED_INAPPLICABLE_FALSE = "skipped_inapplicable_false"
    SKIPPED_INAPPLICABLE_UNKNOWN = "skipped_inapplicable_unknown"
    BLOCKED_UNIMPLEMENTED = "blocked_unimplemented"
    INERT = "INERT"
    REFUSED = "REFUSED"
    PROMOTED = "promoted"
    COMPLETE = "complete"


ELIGIBLE = FamilyState.ELIGIBLE
SKIPPED_INAPPLICABLE_FALSE = FamilyState.SKIPPED_INAPPLICABLE_FALSE
SKIPPED_INAPPLICABLE_UNKNOWN = FamilyState.SKIPPED_INAPPLICABLE_UNKNOWN
BLOCKED_UNIMPLEMENTED = FamilyState.BLOCKED_UNIMPLEMENTED
INERT = FamilyState.INERT
REFUSED = FamilyState.REFUSED
PROMOTED = FamilyState.PROMOTED
COMPLETE = FamilyState.COMPLETE


@dataclass(frozen=True)
class FamilyDecision:
    """One licensed campaign transition, with the evidence needed to audit it."""

    family: str
    state: FamilyState
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise CampaignDeclarationError("a family decision must name its family")
        if not isinstance(self.state, FamilyState):
            raise CampaignEvidenceError(f"family decision state must be a FamilyState, got {self.state!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise CampaignEvidenceError("a family decision must state its reason")
        if not isinstance(self.details, Mapping):
            raise CampaignEvidenceError("family decision details must be a mapping")

    @property
    def can_run_tier1(self) -> bool:
        return self.state is ELIGIBLE

    @property
    def can_run_tier2(self) -> bool:
        return self.state is PROMOTED

    @property
    def is_complete(self) -> bool:
        return self.state is COMPLETE

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "state": self.state.value, "reason": self.reason,
                "details": dict(self.details)}


@dataclass(frozen=True, order=True)
class ReplicaIdentity:
    """The exact identity of one expected query; no field is inferred from its result path."""

    family: str
    member: str
    tier: str
    replica: str

    def __post_init__(self) -> None:
        for key, value in (("family", self.family), ("member", self.member),
                           ("tier", self.tier), ("replica", self.replica)):
            if not isinstance(value, str) or not value.strip():
                raise CampaignDeclarationError(
                    f"replica identity {key} must be a non-empty string, got {value!r}")

    @property
    def label(self) -> str:
        return "/".join((self.family, self.member, self.tier, self.replica))

    def as_dict(self) -> dict[str, str]:
        return {"family": self.family, "member": self.member,
                "tier": self.tier, "replica": self.replica}


@dataclass(frozen=True)
class Measurement:
    """One measured query, including the concurrency required to interpret its wall time."""

    identity: ReplicaIdentity
    parameters: Mapping[str, Any]
    seconds: float
    cycles: int
    words: int
    concurrency: int

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise CampaignEvidenceError("measurement parameters must be a mapping")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise CampaignEvidenceError(
                f"measurement {self.identity.label} cycles must be a positive integer")
        if isinstance(self.words, bool) or not isinstance(self.words, int) or self.words < 0:
            raise CampaignEvidenceError(
                f"measurement {self.identity.label} words must be a non-negative integer")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
            raise CampaignEvidenceError(
                f"measurement {self.identity.label} seconds must be a non-negative number")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int):
            raise CampaignEvidenceError(
                f"measurement {self.identity.label} concurrency must be an integer >= 1")
        # CostSample owns the shared wall-time/concurrency contract. Constructing one here means a
        # campaign record cannot evolve a weaker, subtly incompatible copy of that validation.
        try:
            self.to_cost_sample()
        except (TypeError, ValueError) as exc:
            raise CampaignEvidenceError(
                f"measurement {self.identity.label} is not a valid cost sample: {exc}") from exc

    def to_cost_sample(self) -> CostSample:
        return CostSample(seconds=self.seconds, cycles=self.cycles, words=self.words,
                          concurrency=self.concurrency, kind=ProbeKind.CORPUS,
                          label=self.identity.label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict(),
            "parameters": dict(self.parameters),
            "cost": {
                "seconds": self.seconds,
                "cycles": self.cycles,
                "words": self.words,
                "concurrency": self.concurrency,
                "kind": ProbeKind.CORPUS.value,
            },
        }


@dataclass(frozen=True)
class FalsifierEvidence:
    """The run-authored falsifier verdict for one measured replica."""

    identity: ReplicaIdentity
    negative_control: bool
    fired: bool | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.negative_control, bool):
            raise CampaignEvidenceError("negative_control must be exactly True or False")
        if self.fired is not None and not isinstance(self.fired, bool):
            raise CampaignEvidenceError("falsifier fired must be True, False, or None")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise CampaignEvidenceError("falsifier evidence must state its instrument-level reason")

    @classmethod
    def from_ab_decision(cls, identity: ReplicaIdentity, decision: ABDecision, *,
                         negative_control: bool) -> "FalsifierEvidence":
        """Translate the existing eta A/B verdict without treating correctness rejection as a fire."""
        if not isinstance(decision, ABDecision):
            raise CampaignEvidenceError("falsifier evidence must come from an ABDecision")
        established = decision.bit_exact is True and decision.invariants_held is True
        if not established or decision.eta.state == UNDETERMINABLE:
            fired: bool | None = None
        else:
            fired = decision.eta.state == DID_NOT_RISE
        return cls(identity=identity, negative_control=negative_control,
                   fired=fired, reason=decision.reason)

    def as_dict(self) -> dict[str, Any]:
        return {"identity": self.identity.as_dict(), "negative_control": self.negative_control,
                "fired": self.fired, "reason": self.reason}


def validate_declaration(performance: Mapping[str, Any]) -> str:
    """Validate the portion of a performance block the decision engine consumes."""
    if not isinstance(performance, Mapping):
        raise CampaignDeclarationError("performance declaration must be a mapping")
    family = performance.get("family")
    if not isinstance(family, str) or not family.strip():
        raise CampaignDeclarationError("performance declaration must name a family")

    falsifier = performance.get("falsifier")
    if not isinstance(falsifier, Mapping):
        raise CampaignDeclarationError(f"family {family!r} must declare a falsifier mapping")
    if "fired" in falsifier:
        raise CampaignDeclarationError(
            f"family {family!r} authors falsifier.fired; that verdict is written by the run, "
            "never by the capsule declaration")
    observation = falsifier.get("observation")
    if not isinstance(observation, str) or not observation.strip():
        raise CampaignDeclarationError(f"family {family!r} falsifier must name its observation")
    negative_control = falsifier.get("negative_control")
    if not isinstance(negative_control, str) or not negative_control.strip():
        raise CampaignDeclarationError(
            f"family {family!r} falsifier must name its negative-control member")

    gate = performance.get("gate") or {}
    if not isinstance(gate, Mapping):
        raise CampaignDeclarationError(f"family {family!r} gate must be a mapping")
    required = gate.get("traits") or []
    if not isinstance(required, list) or any(
            not isinstance(name, str) or not name.strip() for name in required):
        raise CampaignDeclarationError(f"family {family!r} gate.traits must be a list of names")
    duplicates = sorted({name for name in required if required.count(name) > 1})
    if duplicates:
        raise CampaignDeclarationError(
            f"family {family!r} repeats gate trait(s) {duplicates}; each fact is asked once")
    unknown = sorted(set(required) - set(TRAITS))
    if unknown:
        raise CampaignDeclarationError(
            f"family {family!r} names unknown performance trait(s) {unknown}; allowed: {list(TRAITS)}")

    emitter = performance.get("emitter")
    if not isinstance(emitter, Mapping) or not str(emitter.get("status") or "").strip():
        raise CampaignDeclarationError(f"family {family!r} must declare emitter.status")
    return family


def assess_eligibility(performance: Mapping[str, Any], *, traits: Mapping[str, bool | None],
                       emitter_implemented: bool | None) -> FamilyDecision:
    """Classify trait applicability before any oracle budget is spent."""
    family = validate_declaration(performance)
    if not isinstance(traits, Mapping):
        raise CampaignEvidenceError("derived traits must be a mapping")
    if emitter_implemented is not None and not isinstance(emitter_implemented, bool):
        raise CampaignEvidenceError("emitter_implemented must be True, False, or None")

    required = list(((performance.get("gate") or {}).get("traits") or []))
    states: dict[str, bool | None] = {}
    for name in required:
        value = traits.get(name)
        if value is not None and not isinstance(value, bool):
            raise CampaignEvidenceError(
                f"derived trait {name!r} must be exactly True, False, or None, got {value!r}")
        states[name] = value
    refuted = [name for name, value in states.items() if value is False]
    unknown = [name for name, value in states.items() if value is None]
    details: dict[str, Any] = {
        "traits": states,
        "refuted_traits": refuted,
        "unknown_traits": unknown,
        "negative_control": str((performance.get("falsifier") or {})["negative_control"]),
    }
    if refuted:
        return FamilyDecision(
            family, SKIPPED_INAPPLICABLE_FALSE,
            f"derived trait(s) {refuted} are False; the family is proved inapplicable",
            details)
    if unknown:
        return FamilyDecision(
            family, SKIPPED_INAPPLICABLE_UNKNOWN,
            f"derived trait(s) {unknown} are UNKNOWN; missing evidence is not absence",
            details)

    emitter_status = str((performance.get("emitter") or {}).get("status"))
    details["emitter_status"] = emitter_status
    details["emitter_implemented"] = emitter_implemented
    if emitter_implemented is not True:
        why = "is not implemented" if emitter_implemented is False else "was not established ready"
        return FamilyDecision(
            family, BLOCKED_UNIMPLEMENTED,
            f"emitter {emitter_status!r} {why}; implementation work is not target inapplicability",
            details)
    return FamilyDecision(family, ELIGIBLE,
                          "every required trait is True and the emitter is implemented", details)


def _duplicates(values: Iterable[ReplicaIdentity]) -> list[ReplicaIdentity]:
    seen: set[ReplicaIdentity] = set()
    repeated: set[ReplicaIdentity] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def _labels(values: Iterable[ReplicaIdentity]) -> list[str]:
    return [identity.label for identity in sorted(values)]


def validate_measurements(measurements: Sequence[Measurement], *,
                          expected_identities: Sequence[ReplicaIdentity],
                          fitted_parameters: Sequence[str] = ()) -> tuple[Measurement, ...]:
    """Require the exact replica set and at least two distinct points on every fitted axis."""
    rows = tuple(measurements)
    expected = tuple(expected_identities)
    if not expected:
        raise CampaignEvidenceError("campaign declares zero expected replica identities")
    if any(not isinstance(identity, ReplicaIdentity) for identity in expected):
        raise CampaignEvidenceError("every expected replica identity must be a ReplicaIdentity")
    repeated_expected = _duplicates(expected)
    if repeated_expected:
        raise CampaignDeclarationError(
            f"campaign repeats expected replica identity/identities {_labels(repeated_expected)}")
    if any(not isinstance(row, Measurement) for row in rows):
        raise CampaignEvidenceError("every result must be a Measurement")
    repeated = _duplicates(row.identity for row in rows)
    if repeated:
        raise CampaignEvidenceError(f"duplicate replica measurement(s): {_labels(repeated)}")

    expected_set, observed_set = set(expected), {row.identity for row in rows}
    missing, unexpected = expected_set - observed_set, observed_set - expected_set
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing replica measurement(s) {_labels(missing)}")
        if unexpected:
            parts.append(f"unexpected replica measurement(s) {_labels(unexpected)}")
        raise CampaignEvidenceError("; ".join(parts))

    fitted = tuple(fitted_parameters)
    if any(not isinstance(name, str) or not name.strip() for name in fitted):
        raise CampaignDeclarationError("fitted parameter names must be non-empty strings")
    if len(set(fitted)) != len(fitted):
        raise CampaignDeclarationError("a fitted parameter may only be declared once")
    for name in fitted:
        absent = [row.identity.label for row in rows if name not in row.parameters]
        if absent:
            raise CampaignEvidenceError(
                f"fitted parameter {name!r} is absent from measurement(s) {absent}")
        distinct: list[Any] = []
        for row in rows:
            value = row.parameters[name]
            if not any(value == previous for previous in distinct):
                distinct.append(value)
        if len(distinct) < 2:
            raise CampaignEvidenceError(
                f"fitted parameter {name!r} has {len(distinct)} distinct point(s); at least two "
                "distinct measurements are required per fitted parameter")
    return rows


def _refused(decision: FamilyDecision, reason: str, **details: Any) -> FamilyDecision:
    merged = dict(decision.details)
    merged.update(details)
    return FamilyDecision(decision.family, REFUSED, reason, merged)


def decide_promotion(
    eligibility: FamilyDecision,
    measurements: Sequence[Measurement],
    *,
    expected_identities: Sequence[ReplicaIdentity],
    falsifier_evidence: Sequence[FalsifierEvidence],
    base: Composed,
    candidate: Composed,
    fitted_parameters: Sequence[str] = (),
    demands_base: Mapping[str, Any] | None = None,
    demands_candidate: Mapping[str, Any] | None = None,
) -> FamilyDecision:
    """Promote only a complete screen whose negative control fired and whose unknowns cancel."""
    if not isinstance(eligibility, FamilyDecision):
        raise CampaignEvidenceError("promotion requires a FamilyDecision from assess_eligibility")
    if eligibility.state is not ELIGIBLE:
        return eligibility
    try:
        rows = validate_measurements(
            measurements, expected_identities=expected_identities,
            fitted_parameters=fitted_parameters)
    except CampaignEvidenceError as exc:
        return _refused(eligibility, f"screen evidence refused: {exc}", stage="screen")
    wrong_family = [row.identity.label for row in rows
                    if row.identity.family != eligibility.family]
    if wrong_family:
        return _refused(
            eligibility,
            f"screen evidence belongs to another family: {wrong_family}", stage="screen")

    evidence = tuple(falsifier_evidence)
    if any(not isinstance(row, FalsifierEvidence) for row in evidence):
        return _refused(eligibility, "falsifier evidence contains a non-evidence record",
                        stage="screen")
    repeated = _duplicates(row.identity for row in evidence)
    if repeated:
        return _refused(
            eligibility, f"falsifier evidence repeats replica(s) {_labels(repeated)}",
            stage="screen")
    measured_ids = {row.identity for row in rows}
    unattached = [row.identity for row in evidence if row.identity not in measured_ids]
    if unattached:
        return _refused(
            eligibility,
            f"falsifier evidence has no matching measurement for replica(s) {_labels(unattached)}",
            stage="screen")
    controls = [row for row in evidence if row.negative_control]
    if not controls:
        return _refused(
            eligibility,
            "no measured negative-control falsifier evidence exists; an empty check cannot promote",
            stage="screen")
    declared_control = str(eligibility.details.get("negative_control") or "")
    mislabelled = [row.identity for row in controls if row.identity.member != declared_control]
    if mislabelled:
        return _refused(
            eligibility,
            f"falsifier evidence labels undeclared member(s) {_labels(mislabelled)} as the negative "
            f"control; the declaration names {declared_control!r}",
            stage="screen")
    undetermined = [row for row in controls if row.fired is None]
    if undetermined:
        return _refused(
            eligibility,
            "negative-control falsifier evidence is undetermined: "
            + "; ".join(row.reason for row in undetermined),
            stage="screen", falsifier=[row.as_dict() for row in controls])
    if not any(row.fired is True for row in controls):
        reason = "; ".join(row.reason for row in controls)
        return FamilyDecision(
            eligibility.family, INERT,
            f"the measured negative control never fired; the family is instrument-inert: {reason}",
            {**dict(eligibility.details), "stage": "screen", "falsifier_fired": False,
             "falsifier": [row.as_dict() for row in controls],
             "measurements": [row.as_dict() for row in rows]})

    comparable, comparison_reason = differential.comparable(
        base, candidate, demands_a=demands_base, demands_b=demands_candidate)
    if not comparable:
        return _refused(
            eligibility, f"differential comparison refused: {comparison_reason}", stage="screen",
            falsifier_fired=True, falsifier=[row.as_dict() for row in controls],
            measurements=[row.as_dict() for row in rows],
            comparison={"comparable": False, "reason": comparison_reason})
    return FamilyDecision(
        eligibility.family, PROMOTED,
        "the negative-control falsifier fired and the unpriced demands provably cancel",
        {**dict(eligibility.details), "stage": "screen", "falsifier_fired": True,
         "falsifier": [row.as_dict() for row in controls],
         "measurements": [row.as_dict() for row in rows],
         "comparison": {"comparable": True, "reason": comparison_reason}})


def complete_family(
    promoted: FamilyDecision,
    measurements: Sequence[Measurement],
    *,
    expected_identities: Sequence[ReplicaIdentity],
    certification_passed: bool | None,
    certification_reason: str,
    fitted_parameters: Sequence[str] = (),
) -> FamilyDecision:
    """Complete a promoted family only after its exact certifying set ran and passed."""
    if not isinstance(promoted, FamilyDecision) or promoted.state is not PROMOTED:
        raise CampaignEvidenceError("only a promoted family can enter certification")
    if certification_passed is not None and not isinstance(certification_passed, bool):
        raise CampaignEvidenceError("certification_passed must be True, False, or None")
    if not isinstance(certification_reason, str) or not certification_reason.strip():
        raise CampaignEvidenceError("certification must state its reason")
    try:
        rows = validate_measurements(
            measurements, expected_identities=expected_identities,
            fitted_parameters=fitted_parameters)
    except CampaignEvidenceError as exc:
        return _refused(promoted, f"certification evidence refused: {exc}", stage="certify")
    wrong_family = [row.identity.label for row in rows
                    if row.identity.family != promoted.family]
    if wrong_family:
        return _refused(
            promoted, f"certification evidence belongs to another family: {wrong_family}",
            stage="certify")
    if certification_passed is not True:
        state = "failed" if certification_passed is False else "was not established"
        return _refused(
            promoted, f"certification {state}: {certification_reason}", stage="certify",
            certification_passed=certification_passed,
            certifying_measurements=[row.as_dict() for row in rows])
    return FamilyDecision(
        promoted.family, COMPLETE, f"certification passed: {certification_reason}",
        {**dict(promoted.details), "stage": "complete", "certification_passed": True,
         "certifying_measurements": [row.as_dict() for row in rows]})

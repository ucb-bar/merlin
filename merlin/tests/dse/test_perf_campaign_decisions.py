"""Performance-family promotion is evidence-driven and fail-closed.

These are pure decision tests: no target, compiler, or simulator is selected.  The campaign runner
will supply declarations, derived traits, measurements, and structural bounds; this module decides
which state that evidence licenses.
"""
from __future__ import annotations

import pytest

from merlin.perf.campaign import (
    BLOCKED_UNIMPLEMENTED,
    COMPLETE,
    ELIGIBLE,
    INERT,
    PROMOTED,
    REFUSED,
    SKIPPED_INAPPLICABLE_FALSE,
    SKIPPED_INAPPLICABLE_UNKNOWN,
    CampaignDeclarationError,
    CampaignEvidenceError,
    FalsifierEvidence,
    Measurement,
    ReplicaIdentity,
    assess_eligibility,
    complete_family,
    decide_promotion,
    validate_measurements,
)
from merlin.perf.envelope import UNKNOWN, Composed, Composition
from merlin.perf.falsifier import EtaObservation, ab_decision
from merlin.perf.oracle_cost import CostSample


def _declaration(*traits: str) -> dict:
    return {
        "family": "PX",
        "falsifier": {
            "observation": "candidate must change the measured instrument",
            "negative_control": "base",
        },
        "gate": {"traits": list(traits)},
        "emitter": {"status": "existing", "entry": "package.module.emitter"},
    }


def _identity(member: str, replica: str, tier: str = "screen") -> ReplicaIdentity:
    return ReplicaIdentity(family="PX", member=member, tier=tier, replica=replica)


def _measurement(identity: ReplicaIdentity, point: int, *, concurrency: int = 1) -> Measurement:
    return Measurement(identity=identity, parameters={"depth": point}, seconds=0.25,
                       cycles=100 + point, words=12, concurrency=concurrency)


def _bound(partial: float, *, demand_unknown: bool = False) -> Composed:
    unresolved = ("unpriced",) if demand_unknown else ()
    return Composed(cycles=UNKNOWN if unresolved else partial, partial_cycles=partial,
                    floor_cycles=0.0, operator=Composition.SUM, eta=0.0,
                    overlap_saving=0.0, unresolved=unresolved, workload_fixed_cycles=0)


def _eligible():
    return assess_eligibility(_declaration("explicit_dma"),
                              traits={"explicit_dma": True}, emitter_implemented=True)


def _screen_evidence(*, fired: bool | None = True, reason: str = "control rejected"):
    expected = (_identity("base", "r0"), _identity("candidate", "r0"))
    measured = (_measurement(expected[0], 1), _measurement(expected[1], 2))
    falsifier = (FalsifierEvidence(identity=expected[0], negative_control=True,
                                   fired=fired, reason=reason),)
    return expected, measured, falsifier


def test_authored_falsifier_verdict_is_rejected_even_when_it_says_not_run() -> None:
    declaration = _declaration()
    declaration["falsifier"]["fired"] = "not_run"
    with pytest.raises(CampaignDeclarationError, match="written by the run"):
        assess_eligibility(declaration, traits={}, emitter_implemented=True)


def test_trait_false_unknown_and_missing_are_distinct_from_eligible() -> None:
    declaration = _declaration("explicit_dma", "explicit_completion")

    refuted = assess_eligibility(
        declaration, traits={"explicit_dma": False, "explicit_completion": None},
        emitter_implemented=True)
    assert refuted.state == SKIPPED_INAPPLICABLE_FALSE
    assert refuted.details["traits"] == {"explicit_dma": False, "explicit_completion": None}

    unknown = assess_eligibility(
        declaration, traits={"explicit_dma": True}, emitter_implemented=True)
    assert unknown.state == SKIPPED_INAPPLICABLE_UNKNOWN
    assert unknown.details["unknown_traits"] == ["explicit_completion"]

    eligible = assess_eligibility(
        declaration, traits={"explicit_dma": True, "explicit_completion": True},
        emitter_implemented=True)
    assert eligible.state == ELIGIBLE and eligible.can_run_tier1


def test_unknown_trait_names_and_non_tri_state_values_fail_closed() -> None:
    with pytest.raises(CampaignDeclarationError, match="unknown performance trait"):
        assess_eligibility(_declaration("made_up"), traits={"made_up": True},
                           emitter_implemented=True)
    with pytest.raises(CampaignEvidenceError, match="True, False, or None"):
        assess_eligibility(_declaration("explicit_dma"), traits={"explicit_dma": 1},
                           emitter_implemented=True)


def test_an_applicable_family_with_no_emitter_is_blocked_not_inapplicable() -> None:
    decision = assess_eligibility(_declaration("explicit_dma"),
                                  traits={"explicit_dma": True}, emitter_implemented=False)
    assert decision.state == BLOCKED_UNIMPLEMENTED
    assert decision.details["emitter_status"] == "existing"


def test_measurements_require_exact_replica_identities_and_two_points_per_fit() -> None:
    expected = (_identity("base", "r0"), _identity("candidate", "r0"))
    rows = (_measurement(expected[0], 1), _measurement(expected[1], 2))
    assert validate_measurements(rows, expected_identities=expected,
                                 fitted_parameters=("depth",)) == rows

    with pytest.raises(CampaignEvidenceError, match="missing replica"):
        validate_measurements(rows[:1], expected_identities=expected,
                              fitted_parameters=("depth",))
    with pytest.raises(CampaignEvidenceError, match="duplicate replica"):
        validate_measurements((rows[0], rows[0], rows[1]), expected_identities=expected,
                              fitted_parameters=("depth",))
    with pytest.raises(CampaignEvidenceError, match="at least two distinct measurements"):
        validate_measurements((_measurement(expected[0], 1), _measurement(expected[1], 1)),
                              expected_identities=expected, fitted_parameters=("depth",))


def test_measurement_is_costsample_compatible_and_never_drops_concurrency() -> None:
    row = _measurement(_identity("base", "r0"), 1, concurrency=3)
    sample = row.to_cost_sample()
    assert isinstance(sample, CostSample)
    assert sample.concurrency == row.as_dict()["cost"]["concurrency"] == 3
    assert sample.label == "PX/base/screen/r0"

    with pytest.raises(CampaignEvidenceError, match="concurrency"):
        _measurement(_identity("base", "bad"), 1, concurrency=0)


def test_missing_or_undetermined_negative_control_evidence_is_refused() -> None:
    expected, rows, evidence = _screen_evidence()
    common = dict(expected_identities=expected, fitted_parameters=("depth",),
                  base=_bound(10), candidate=_bound(8))

    no_control = decide_promotion(_eligible(), rows, falsifier_evidence=(), **common)
    assert no_control.state == REFUSED and "negative-control" in no_control.reason

    unknown = decide_promotion(
        _eligible(), rows,
        falsifier_evidence=(FalsifierEvidence(expected[0], True, None, "instrument unread"),),
        **common)
    assert unknown.state == REFUSED and "undetermined" in unknown.reason

    not_a_control = decide_promotion(
        _eligible(), rows,
        falsifier_evidence=(FalsifierEvidence(expected[0], False, True, "ordinary member"),),
        **common)
    assert not_a_control.state == REFUSED and "negative-control" in not_a_control.reason

    mislabelled = decide_promotion(
        _eligible(), rows,
        falsifier_evidence=(FalsifierEvidence(expected[1], True, True, "wrong member"),),
        **common)
    assert mislabelled.state == REFUSED and "undeclared member" in mislabelled.reason


def test_a_measured_falsifier_that_never_fires_is_inert_with_its_reason() -> None:
    expected, rows, evidence = _screen_evidence(fired=False, reason="counter did not move")
    decision = decide_promotion(
        _eligible(), rows, expected_identities=expected, fitted_parameters=("depth",),
        falsifier_evidence=evidence, base=_bound(10), candidate=_bound(8))
    assert decision.state == INERT and not decision.can_run_tier2
    assert "counter did not move" in decision.reason


def test_fired_control_still_refuses_when_unpriced_demands_do_not_cancel() -> None:
    expected, rows, evidence = _screen_evidence()
    decision = decide_promotion(
        _eligible(), rows, expected_identities=expected, fitted_parameters=("depth",),
        falsifier_evidence=evidence, base=_bound(10, demand_unknown=True),
        candidate=_bound(8, demand_unknown=True), demands_base={"unpriced": 1},
        demands_candidate={"unpriced": 2})
    assert decision.state == REFUSED and "different work" in decision.reason
    assert decision.details["falsifier_fired"] is True


def test_only_fired_control_plus_comparable_schedules_promotes() -> None:
    expected, rows, evidence = _screen_evidence()
    decision = decide_promotion(
        _eligible(), rows, expected_identities=expected, fitted_parameters=("depth",),
        falsifier_evidence=evidence, base=_bound(10, demand_unknown=True),
        candidate=_bound(8, demand_unknown=True), demands_base={"unpriced": 1},
        demands_candidate={"unpriced": 1})
    assert decision.state == PROMOTED and decision.can_run_tier2
    assert decision.details["comparison"]["comparable"] is True


def test_existing_eta_decision_converts_to_a_negative_control_reading() -> None:
    base = EtaObservation("base", 0, 8, ("a", "b"), work="same", detail="measured")
    unchanged = EtaObservation("control", 0, 8, ("a", "b"), work="same", detail="measured")
    improved = EtaObservation("candidate", 4, 8, ("a", "b"), work="same", detail="measured")

    fired = FalsifierEvidence.from_ab_decision(
        _identity("control", "r0"), ab_decision(base, unchanged, bit_exact=True,
                                                 invariants_held=True), negative_control=True)
    did_not_fire = FalsifierEvidence.from_ab_decision(
        _identity("candidate", "r0"), ab_decision(base, improved, bit_exact=True,
                                                   invariants_held=True), negative_control=False)
    assert fired.fired is True
    assert did_not_fire.fired is False


def test_completion_requires_promotion_exact_certifying_replicas_and_a_pass() -> None:
    expected, rows, evidence = _screen_evidence()
    promoted = decide_promotion(
        _eligible(), rows, expected_identities=expected, fitted_parameters=("depth",),
        falsifier_evidence=evidence, base=_bound(10), candidate=_bound(8))
    cert_expected = (_identity("candidate", "c0", "certify"),
                     _identity("candidate", "c1", "certify"))
    cert_rows = (_measurement(cert_expected[0], 1), _measurement(cert_expected[1], 2))

    done = complete_family(promoted, cert_rows, expected_identities=cert_expected,
                           fitted_parameters=("depth",), certification_passed=True,
                           certification_reason="prediction held")
    assert done.state == COMPLETE and done.is_complete

    missing = complete_family(promoted, cert_rows[:1], expected_identities=cert_expected,
                              fitted_parameters=("depth",), certification_passed=True,
                              certification_reason="prediction held")
    assert missing.state == REFUSED and "missing replica" in missing.reason

    failed = complete_family(promoted, cert_rows, expected_identities=cert_expected,
                             fitted_parameters=("depth",), certification_passed=False,
                             certification_reason="layer-scale prediction was falsified")
    assert failed.state == REFUSED and "falsified" in failed.reason

    with pytest.raises(CampaignEvidenceError, match="only a promoted family"):
        complete_family(_eligible(), cert_rows, expected_identities=cert_expected,
                        fitted_parameters=("depth",), certification_passed=True,
                        certification_reason="prediction held")

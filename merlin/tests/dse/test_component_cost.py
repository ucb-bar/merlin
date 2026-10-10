"""Complete-program accounting and independently held screening gates."""

from dataclasses import replace

import pytest

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
    complete_component_cost,
    validate_complete_cost_report,
)
from merlin.perf.component_screen import (
    ComponentOpportunity,
    ComponentScreenPolicy,
    cheapest_distinguishing_experiment,
    qualify_component_screen,
    rank_component_opportunities,
    validate_component_screen_report,
)
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def calibration(target, rate=1):
    receipts = [sha([target, index]) for index in range(4)]
    return {
        "schema": "phase2_host_analytical_calibration_v1",
        "target_sha256": target,
        "evidence_sha256s": receipts,
        "composition": {"operator": "sum", "eta": 0, "provenance_sha256": receipts[0]},
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0,
        "features": [
            {
                "id": kind,
                "pointer": "/count",
                "resource": kind,
                "kind": kind,
                "cycles_per_unit": {"lo": rate, "hi": rate},
                "physical_bytes_per_unit": 4 if kind != "compute" else None,
                "commands_per_unit": 1 if kind != "compute" else None,
                "transitions_per_unit": 1 if kind == "encoding" else None,
                "provenance_sha256": receipts[i + 1],
            }
            for i, kind in enumerate(("compute", "movement", "encoding"))
        ],
    }


def observation(target, scope):
    return ComponentFeatureObservation(
        sha("compiler"),
        sha("member"),
        sha("corpus"),
        target,
        scope.sha256,
        sha("domain"),
        sha("elf"),
        sha("deps"),
        sha("inputs"),
        (sha("observed"),),
        (
            ComponentCostRegion("cold", COMPLETE_STAGES, ("compute",), {"count": 13}, accounting="inclusive"),
            ComponentCostRegion("nested", ("packing",), ("movement",), {"count": 3}, parent="cold"),
        ),
        (ComponentCostRegion("warm", COMPLETE_STAGES, ("compute",), {"count": 7}),),
        "PASS",
        legality_status="PASS",
    )


@pytest.mark.parametrize("target,rate", [(sha("geometry-a"), 2), (sha("geometry-b"), 7)])
def test_cold_warm_complete_cost_uses_derived_rates_and_does_not_double_count(target, rate):
    scope = ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy"))
    totals, report = complete_component_cost(
        observation(target, scope), calibration(target, rate), scope=scope, qualified_domains=(sha("domain"),)
    )
    assert totals["cold"].lo == 13 * rate and totals["warm"].lo == 7 * rate
    assert report["regimes"]["cold"]["regions"][1]["contained"]
    assert validate_complete_cost_report(report) == report
    report["regimes"]["cold"]["total"]["lo"] += 3
    with pytest.raises(ValueError):
        validate_complete_cost_report(report)


def test_missing_stages_domains_and_functional_qualification_are_not_zero_cost():
    scope = ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy"))
    row = observation(sha("target"), scope)
    row = replace(
        row,
        warm=(ComponentCostRegion("partial", ("device",), ("compute",), {"count": 0}),),
        functional_status="UNKNOWN",
    )
    totals, _ = complete_component_cost(row, calibration(row.target_sha256), scope=scope, qualified_domains=())
    assert not totals["warm"].resolved
    assert any("missing complete-cost stage allocation" in reason for reason in totals["warm"].missing)
    assert any("domain" in reason for reason in totals["warm"].missing)
    assert any("functional" in reason for reason in totals["warm"].missing)


def _interval_type_report(resolved):
    """Synthetic arithmetic only; no runtime or held qualification is issued."""
    scope = ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy"))
    row = observation(sha("target"), scope)
    if not resolved:
        row = replace(
            row,
            cold=tuple(replace(region, context={"count": None}) for region in row.cold),
            warm=tuple(replace(region, context={"count": None}) for region in row.warm),
        )
    _totals, report = complete_component_cost(
        row, calibration(row.target_sha256), scope=scope, qualified_domains=(sha("domain"),)
    )
    return report


@pytest.mark.parametrize("resolved", [False, True])
def test_complete_cost_preserves_exact_boolean_interval_states(resolved):
    report = _interval_type_report(resolved)
    assert validate_complete_cost_report(report) == report
    for regime in report["regimes"].values():
        intervals = [regime["total"], *(row["cycles"] for row in regime["regions"])]
        assert all(interval["resolved"] is resolved for interval in intervals)
    assert report["promotion"] == "SCREENING_ONLY"


@pytest.mark.parametrize("regime", ["cold", "warm"])
@pytest.mark.parametrize("location", ["total", "region"])
@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("numeric", [int, float], ids=["int", "float"])
def test_complete_cost_refuses_numeric_interval_states(regime, location, resolved, numeric):
    report = _interval_type_report(resolved)
    selected = report["regimes"][regime]
    interval = selected["total"] if location == "total" else selected["regions"][0]["cycles"]
    interval["resolved"] = numeric(resolved)
    with pytest.raises(ValueError, match="interval schema"):
        validate_complete_cost_report(report)


def held_rows():
    return [
        Observation(
            sha([group, n]), sha(group), group, sha("domain"), {"/count": n}, 10 * n, (sha([group, n, "measurement"]),)
        )
        for group in ("a", "b", "c")
        for n in range(1, 11)
    ]


class Exact:
    def predict(self, features, *, domain_sha256):
        center = 10 * features["/count"]
        return CycleInterval(center - 0.1, center + 0.1, provenance=("held independent mechanism",))


def test_default_screening_requires_held_ranking_error_and_interval_coverage():
    train_rosters = []

    def fit(train):
        train_rosters.append({row.group for row in train})
        return Exact()

    report = qualify_component_screen(held_rows(), fit, calibration_sha256=sha("calibration"))
    assert report["exposable"] and report["interval_coverage"]["rate"] == 1
    assert report["ranking"]["overall"]["decided"] == 135
    assert all(len(roster) == 2 for roster in train_rosters)
    assert report["policy"] == ComponentScreenPolicy().to_dict()
    assert validate_component_screen_report(report) == report
    from copy import deepcopy

    forged = deepcopy(report)
    forged["interval_coverage"]["n"] += 1
    with pytest.raises(ValueError, match="coverage/count"):
        validate_component_screen_report(forged)
    leaked = deepcopy(report)
    leaked["predictions"][0]["training_programs"].append(leaked["predictions"][0]["program"])
    with pytest.raises(ValueError, match="leaks"):
        validate_component_screen_report(leaked)

    class NarrowBiased:
        def predict(self, features, *, domain_sha256):
            return CycleInterval.point(10 * features["/count"] + 0.1, "biased fit")

    failed = qualify_component_screen(held_rows(), lambda train: NarrowBiased(), calibration_sha256=sha("calibration"))
    assert not failed["exposable"] and failed["ranking"]["exposable"]
    assert failed["interval_coverage"]["rate"] == 0


def test_independent_priority_retains_diversity_overlap_and_negative_results():
    def point(n):
        return CycleInterval.point(n, "controlled complete cost")

    rows = [
        ComponentOpportunity("a", "f1", "warm", point(100), point(10), 1, True, True),
        ComponentOpportunity("b", "f1", "warm", point(100), point(20), 1, True, True),
        ComponentOpportunity("c", "f2", "cold", point(100), point(90), 1, True, True),
        ComponentOpportunity("d", "f2", "warm", point(100), point(100), 1, True, True),
        ComponentOpportunity("e", "f2", "warm", point(100), point(120), 1, True, True),
    ]
    result = rank_component_opportunities(rows, work_shares={row.id: 0.2 for row in rows})
    assert result["order"] == ["a", "c", "b"]
    assert {row["status"] for row in result["evidence"]} >= {"TIE_OR_OVERLAP", "REGRESSION"}
    experiments = [
        {"id": "expensive", "evaluation_seconds": 12, "distinguishes": ["overlap"]},
        {"id": "irrelevant", "evaluation_seconds": 1, "distinguishes": ["another"]},
        {"id": "cheap", "evaluation_seconds": 2, "distinguishes": ["overlap"]},
    ]
    assert cheapest_distinguishing_experiment(experiments, unresolved=["overlap"])["id"] == "cheap"


@pytest.mark.parametrize("status", ["FAIL", "UNKNOWN", "REFUSAL"])
def test_unqualified_legality_keeps_complete_cost_unknown_with_scoped_diagnostics(status):
    scope = ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy"))
    row = replace(observation(sha("target"), scope), legality_status=status)
    totals, report = complete_component_cost(
        row, calibration(row.target_sha256), scope=scope, qualified_domains=(sha("domain"),)
    )
    assert all(not interval.resolved for interval in totals.values())
    assert "complete member legality qualification is " + status in totals["warm"].missing
    assert report["regimes"]["warm"]["regions"][0]["cycles"]["resolved"]


def test_opportunity_unknown_gates_remain_unresolved_and_failed_gates_refuse():
    from merlin.perf.component_screen import ComponentOpportunity, rank_component_opportunities

    baseline = CycleInterval.point(20, "complete baseline")
    candidate = CycleInterval.point(10, "complete candidate")
    rows = [
        ComponentOpportunity("unknown_legal", "a", "cold", baseline, candidate, 1, None, True),
        ComponentOpportunity("unknown_correct", "a", "cold", baseline, candidate, 1, True, None),
        ComponentOpportunity("failed_legal", "a", "cold", baseline, candidate, 1, False, None),
        ComponentOpportunity("failed_correct", "a", "cold", baseline, candidate, 1, None, False),
    ]
    evidence = rank_component_opportunities(rows, work_shares={row.id: 0.25 for row in rows})["evidence"]
    assert [row["status"] for row in evidence] == ["UNRESOLVED", "UNRESOLVED", "REFUSAL", "REFUSAL"]
    assert all(row["priority"] is None for row in evidence)

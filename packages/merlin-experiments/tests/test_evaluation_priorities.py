"""Diagnostic arithmetic only; synthetic costs establish no original execution."""

from copy import deepcopy

import pytest
from merlin_experiments.phase2 import evaluation_costs as COST
from merlin_experiments.phase2 import evaluation_priorities as PRIORITY
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_measurement_selection as RTL
from merlin_experiments.phase2.contracts import document_sha256

SOURCE, BASELINE, CORPUS, CERTIFICATE = (letter * 64 for letter in "abcd")
IDENTITY = ("compute", "large")
PRIORITIES = {
    "ranked": [
        {
            "family": IDENTITY[0],
            "capsule": IDENTITY[1],
            "potential_weighted_savings": {"lo": 100, "hi": 200},
            "basis": "derived lower bound; attainable improvement unproven",
        }
    ],
    "unknown": [{"family": "host", "capsule": "copy", "potential_weighted_savings": None, "basis": "no derived bound"}],
}


def observation(seconds=10, *, identity=IDENTITY, kind=RTL.KIND, job="job", cpu=2):
    row = {"family": identity[0], "capsule": identity[1], "measured": True, "comparable": True}
    if kind == FAST.SELECTED_KIND:
        row.update(baseline={"functional_status": "PASS"}, candidate={"functional_status": "PASS"})
    resources = {"cpu_ids": [cpu], "grant": {"cores": 1}, "enforcement": {"hard_aggregate_limits": "UNKNOWN"}}
    context = {
        "kind": kind,
        "candidate_sha256": SOURCE,
        "baseline_sha256": BASELINE,
        "corpus_sha256": CORPUS,
        "certificate_sha256": CERTIFICATE,
        "resource_context_sha256": document_sha256(resources),
        "resources": resources,
        "closure_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "selection_scope": "selected public members",
        "selected_members": [list(identity)],
        "member": {
            "family": identity[0],
            "capsule": identity[1],
            "source_sha256": "1" * 64,
            "performance_sha256": "2" * 64,
        },
    }
    return {
        "state": "completed",
        "kind": kind,
        "job_id": job,
        "source_sha256": SOURCE,
        "selected_members": [list(identity)],
        "result": {"cells": [row]},
        "evaluation_cost": {
            "schema": COST.SCHEMA,
            "status": "observed",
            "job_id": job,
            "outcome": "completed",
            "wall_seconds": seconds,
            "context": context,
            "boundary": COST.BOUNDARY,
            "cache_state": "UNKNOWN",
        },
    }


def project(*observations, priorities=PRIORITIES):
    return PRIORITY.project(
        priorities,
        observations,
        source_sha256=SOURCE,
        baseline_sha256=BASELINE,
        corpus_sha256=CORPUS,
        certificate_sha256=CERTIFICATE,
    )


def test_repeats_keep_conservative_pair_cost_and_savings_ranges():
    result = project(observation(10, job="first"), observation(20, job="second"))
    group = result["groups"][0]
    row = group["ranked"][0]
    assert row["evaluation_cost_seconds"] == {"lo": 10, "hi": 20}
    assert row["savings_per_evaluation_second"] == {"lo": 5, "hi": 20}
    assert row["jobs"] == ["first", "second"]
    assert group["missing_cost_members"] == [["host", "copy"]]
    assert "no future duration" in result["scope"]


def test_cost_routes_and_exact_resources_cannot_be_combined():
    result = project(
        observation(10), observation(1, kind=FAST.SELECTED_KIND, job="cheap"), observation(20, cpu=3, job="other-cpu")
    )
    assert len(result["groups"]) == 3
    assert all(len(group["ranked"][0]["jobs"]) == 1 for group in result["groups"])


def test_unknown_host_bound_keeps_cost_visible_without_inventing_savings():
    result = project(observation(1, identity=("host", "copy")))
    group = result["groups"][0]
    assert group["ranked"] == []
    assert group["unknown"][0]["evaluation_cost_seconds"] == {"lo": 1, "hi": 1}
    assert group["unknown"][0]["savings_per_evaluation_second"] is None


@pytest.mark.parametrize("field", ["candidate_sha256", "baseline_sha256", "corpus_sha256", "certificate_sha256"])
def test_other_source_pair_or_corpus_or_engine_cost_is_unpriced(field):
    row = observation()
    row["evaluation_cost"]["context"][field] = "9" * 64
    assert project(row)["groups"] == []


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf")])
def test_invalid_parent_wall_is_not_a_price(seconds):
    assert project(observation(seconds))["groups"] == []


@pytest.mark.parametrize(
    "mutation", ["batch", "complete-roster", "cancelled", "cleanup-unknown", "failed-arm", "warm", "job", "boundary"]
)
def test_partial_lifecycle_or_mismatched_action_never_prices_a_single_member(mutation):
    row = observation(kind=FAST.SELECTED_KIND)
    cost = row["evaluation_cost"]
    if mutation == "batch":
        row["selected_members"].append(["host", "copy"])
    elif mutation == "complete-roster":
        row["kind"] = FAST.KIND
    elif mutation == "cancelled":
        cost["outcome"] = "cancelled"
    elif mutation == "cleanup-unknown":
        cost["status"] = "UNKNOWN"
    elif mutation == "failed-arm":
        row["result"]["cells"][0]["candidate"]["functional_status"] = "UNKNOWN_OR_FAILURE"
    elif mutation == "warm":
        cost["cache_state"] = "warm"
    elif mutation == "job":
        cost["job_id"] = "another"
    else:
        cost["boundary"] = "worker-only"
    assert project(row)["groups"] == []


def test_changed_member_pins_within_same_context_cannot_add_a_repeat():
    first, second = observation(10), observation(1, job="changed")
    second["evaluation_cost"]["context"]["member"]["source_sha256"] = "9" * 64
    result = project(first, second)
    assert result["groups"][0]["ranked"][0]["jobs"] == ["job"]
    assert result["excluded"][0]["job_id"] == "changed"


def test_projection_cannot_change_input_priorities_or_promote_a_compiler():
    before = deepcopy(PRIORITIES)
    result = project(observation())
    assert PRIORITIES == before
    assert "diagnostic" in result["authority"] and "promotion" in result["authority"]

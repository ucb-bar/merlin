"""Pure advisory arithmetic; no fixture establishes model or target qualification."""

import copy

import pytest
from merlin_experiments.phase2 import exploratory_development_priority as P
from merlin_experiments.phase2.contracts import StageGateError


def feedback(cycles=(98, 100)):
    cells = []
    for index, candidate in enumerate(cycles):
        arms = {}
        for arm, value, digest in (("baseline", 100, "b"), ("candidate", candidate, "c")):
            estimate = dict(
                schema=P.ESTIMATE_SCHEMA, status="estimated", estimated_cycles=value,
                model_sha256="a" * 64, comparison_domain_sha256="d" * 64,
                scope_sha256="e" * 64, report_sha256=digest * 64,
                qualified_for_promotion=False, event_accounting_complete=True,
                domain_deviations=[{"feature": "target.class.flush", "observed": 0, "TRAIN_min": 1}],
                calibration_residuals={"p95_absolute_relative_error": .292, "qualification": "REFUSED_raw33"},
            )
            arms[arm] = dict(functional_status="PASS", compiled={"status": "observed", "elf_sha256": digest * 64},
                             native_exploration={"status": "UNKNOWN", "exploratory_estimate": estimate})
        cells.append(dict(family="public", capsule=str(index), **arms))
    return dict(selected_members=[["public", str(i)] for i in range(len(cycles))], cells=cells)


def required(document):
    return tuple(tuple(key) for key in document["selected_members"])


@pytest.mark.parametrize("cycles,expected", [((98, 100), "candidate"), ((99, 100), "candidate"),
                                           ((102, 100), "baseline"), ((99.1, 100), None),
                                           ((98, 102), None), ((100, 100), None)])
def test_weight_free_dominance_keeps_unchanged_controls_and_abstains_on_tradeoffs(cycles, expected):
    data = feedback(cycles)
    before = copy.deepcopy(data)
    result = P.decide(data, required_members=required(data))
    assert result["next_exploration_parent"] == expected
    assert result["estimate_coverage_complete"] is True
    assert [r["candidate_to_baseline_ratio"] for r in result["members"]] == [v / 100 for v in cycles]
    assert data == before
    assert result["qualified_for_promotion"] is False and result["promotion_authority"] == "NONE"
    assert result["measured_cycles"] is None and result["aggregate_objective"] is None
    assert result["plateau_evidence"] is False and result["status"] == "EXPLORATORY"
    assert result["members"][0]["provenance"]["candidate"]["domain_deviations"]
    result["members"][0]["provenance"]["candidate"]["domain_deviations"].append("another warning")
    assert data == before


def test_original_frozen_threshold_is_an_explicit_input():
    data = feedback((98,))
    assert P.decide(data, required_members=required(data), minimum_improvement=.03)["next_exploration_parent"] is None
    result = P.decide(data, required_members=required(data), minimum_improvement=.02)
    assert result["next_exploration_parent"] == "candidate"


@pytest.mark.parametrize("threshold", [True, 0, -1, 1, float("nan"), float("inf"), "0.01", 10**1000])
def test_bad_threshold_refuses(threshold):
    data = feedback()
    with pytest.raises(StageGateError):
        P.decide(data, required_members=required(data), minimum_improvement=threshold)


@pytest.mark.parametrize("fault", ["missing", "failed", "compiled", "bad_elf", "unavailable", "zero", "negative",
                                   "boolean", "model", "domain", "scope", "same_elf", "pair", "qualified"])
def test_partial_incorrect_or_incompatible_estimates_never_supply_parent_guidance(fault):
    data = feedback()
    arm = data["cells"][0]["candidate"]
    estimate = arm["native_exploration"]["exploratory_estimate"]
    if fault == "missing":
        arm["native_exploration"].pop("exploratory_estimate")
    elif fault == "failed":
        arm["functional_status"] = "FAIL"
    elif fault == "compiled":
        arm["compiled"]["status"] = "UNKNOWN"
    elif fault == "bad_elf":
        arm["compiled"]["elf_sha256"] = "not-an-original-product-pin"
    elif fault == "unavailable":
        estimate.update(status="unavailable", estimated_cycles=None)
    elif fault in {"zero", "negative", "boolean"}:
        estimate["estimated_cycles"] = {"zero": 0, "negative": -1, "boolean": True}[fault]
    elif fault in {"model", "domain", "scope"}:
        field = {"model": "model_sha256", "domain": "comparison_domain_sha256", "scope": "scope_sha256"}[fault]
        estimate[field] = "f" * 64
    elif fault == "pair":
        data["cells"][0]["diagnostic_pair_check"] = {"status": "UNKNOWN"}
    elif fault == "qualified":
        estimate["qualified_for_promotion"] = True
    else:
        arm["compiled"]["elf_sha256"] = data["cells"][0]["baseline"]["compiled"]["elf_sha256"]
    result = P.decide(data, required_members=required(data))
    assert result["next_exploration_parent"] is None
    assert result["latency_recommendation_coverage_complete"] is False
    assert len(result["members"]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_estimates_refuse_instead_of_ranking(value):
    data = feedback()
    data["cells"][0]["candidate"]["native_exploration"]["exploratory_estimate"]["estimated_cycles"] = value
    result = P.decide(data, required_members=required(data))
    assert result["next_exploration_parent"] is None
    assert result["estimate_coverage_complete"] is False
    assert result["members"][0]["candidate_to_baseline_ratio"] is None


@pytest.mark.parametrize("flag", [False, None, "missing"])
def test_incomplete_event_accounting_keeps_numbers_visible_but_abstains(flag):
    data = feedback()
    estimate = data["cells"][0]["candidate"]["native_exploration"]["exploratory_estimate"]
    if flag == "missing":
        estimate.pop("event_accounting_complete")
    else:
        estimate["event_accounting_complete"] = flag
    result = P.decide(data, required_members=required(data))
    assert result["next_exploration_parent"] is None
    if flag is not None:
        assert result["members"][0]["candidate_to_baseline_ratio"] == .98
    assert result["latency_recommendation_coverage_complete"] is False


def test_models_must_also_match_across_the_complete_selected_roster():
    data = feedback()
    for arm in ("baseline", "candidate"):
        data["cells"][1][arm]["native_exploration"]["exploratory_estimate"]["model_sha256"] = "f" * 64
    result = P.decide(data, required_members=required(data))
    assert all(row["candidate_to_baseline_ratio"] is not None for row in result["members"])
    assert result["estimate_coverage_complete"] is False and result["next_exploration_parent"] is None


@pytest.mark.parametrize("fault", ["dropped_selected", "changed_selected", "duplicate_cell", "dropped_cell"])
def test_original_declared_members_cannot_be_replaced_or_dropped(fault):
    data = feedback()
    members = required(data)
    if fault == "dropped_selected":
        data["selected_members"].pop()
    elif fault == "changed_selected":
        data["selected_members"][0][1] = "another"
    elif fault == "duplicate_cell":
        data["cells"].append(copy.deepcopy(data["cells"][0]))
    else:
        data["cells"].pop()
    with pytest.raises(StageGateError):
        P.decide(data, required_members=members)


def test_unselected_frozen_members_do_not_change_the_declared_advisory_roster():
    data = feedback((98,))
    data["cells"].append({"family": "pending", "capsule": "other", "baseline": None, "candidate": None})
    result = P.decide(data, required_members=required(data))
    assert len(result["members"]) == 1 and result["next_exploration_parent"] == "candidate"

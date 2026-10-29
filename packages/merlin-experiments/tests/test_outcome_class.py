"""Outcome/failure-category fields on score rows are record-only and carry the TRUE category."""
from __future__ import annotations

from merlin.targetgen import outcome_class as OC


def _tiers(**kw):
    return {t: v for t, v in kw.items()}


NA = {"status": "skipped", "not_applicable": True, "reason": "whole program"}


def test_pass_row_has_no_failure_fields():
    f = OC.outcome_fields({"status": "pass", "tiers": {"L0": NA, "L2": {"status": "pass"}}}, ["L0", "L2"])
    assert f["outcome"] == "pass" and "failure_category" not in f
    assert f["tier_outcomes"] == {"L0": "not_applicable", "L2": "pass"}


def test_host_compute_veto_is_a_post_tier_fail_not_a_skip():
    r = {"status": "fail", "tiers": {"L0": NA, "L1": NA, "L2": {"status": "pass"}},
         "failure": {"plane": "host_compute", "category": "host_compute_in_accelerator_group", "detail": "x"}}
    f = OC.outcome_fields(r, ["L0", "L1", "L2"])
    assert (f["outcome"], f["failure_category"], f["failure_tier"], f["failure_stage"]) == (
        "fail", "host_compute_in_accelerator_group", None, "post_tier")
    assert f["tier_outcomes"]["L0"] == "not_applicable"


def test_must_refuse_violated_category_and_unavailable_tier():
    r = {"status": "fail", "tiers": {"L2": {"status": "pass"}, "L3": {"status": "unavailable"}},
         "failure": {"plane": "expected_refusal", "category": "MUST_REFUSE_VIOLATED"}}
    f = OC.outcome_fields(r, ["L2", "L3"])
    assert f["failure_category"] == "MUST_REFUSE_VIOLATED" and f["failure_stage"] == "post_tier"
    assert f["tier_outcomes"]["L3"] == "unavailable"


def test_mismatch_names_the_failing_tier():
    r = {"status": "fail", "tiers": {"L0": {"status": "pass"}, "L2": {"status": "fail"}},
         "failure": {"plane": "numeric", "category": "FUNCTIONAL_MISMATCH"}}
    f = OC.outcome_fields(r, ["L0", "L2"])
    assert (f["failure_category"], f["failure_tier"], f["failure_stage"]) == ("FUNCTIONAL_MISMATCH", "L2", "tier")


def test_explicit_failure_tier_wins_and_missing_category_is_explicit():
    f = OC.outcome_fields({"status": "fail", "tiers": {"L2": {"status": "fail"}}, "failure": {"tier": "L3"}}, ["L2"])
    assert f["failure_tier"] == "L3" and f["failure_category"] == OC.UNCATEGORIZED
    g = OC.outcome_fields({"status": "fail", "tiers": {}}, [])
    assert g["failure_stage"] == "pre_tier" and g["failure_tier"] is None


def test_skipped_and_unavailable_statuses_are_distinct_from_fail():
    for status, kind in (("gated", "skipped"), ("not_graded", "skipped"), ("screened_only", "skipped"),
                         ("infrastructure_fault", "unavailable"), ("budget_exhausted", "unavailable"),
                         ("cert_not_measured", "unavailable"), ("fail", "fail"), ("pass", "pass")):
        assert OC.outcome_fields({"status": status, "tiers": {}}, [])["outcome"] == kind


def test_tier_outcome_kinds():
    assert OC.tier_outcome({"status": "skipped", "budget_deferred": True}) == "skipped"
    assert OC.tier_outcome({"status": "skipped"}) == "skipped"
    assert OC.tier_outcome("fail") == "fail"
    assert OC.tier_outcome(None) == "unavailable"
    assert OC.tier_outcome(NA) == "not_applicable"


def test_summarize_by_category_and_tier_and_never_touches_status():
    rows = [
        {"status": "pass", **OC.outcome_fields({"status": "pass", "tiers": {"L0": NA}}, ["L0"])},
        *[{"status": "fail", **OC.outcome_fields(
            {"status": "fail", "tiers": {"L2": {"status": "pass"}},
             "failure": {"plane": "host_compute", "category": "host_compute_in_accelerator_group"}}, ["L2"])}] * 2,
        {"status": "gated", **OC.outcome_fields({"status": "gated", "tiers": {}}, [])},
    ]
    before = [dict(r) for r in rows]
    s = OC.summarize(rows)
    assert s["pass"] == 1 and s["fail"] == {"host_compute_in_accelerator_group@post_tier": 2}
    assert s["skipped"] == {"gated": 1} and s["not_applicable_tiers"] == {"L0": 1}
    assert rows == before


def test_final_coverage_obligations_carry_category_without_changing_status():
    from merlin_experiments.phase1 import joint_final_coverage as F
    group = F._capsule_group([{"name": "a"}])
    row = {"capsule": "a", "status": "fail", "tiers": {"L0": "skipped"}, "tier_reasons": {"L0": "n/a"},
           **OC.outcome_fields({"status": "fail", "tiers": {"L2": {"status": "pass"}},
                                "failure": {"plane": "expected_refusal", "category": "MUST_REFUSE_VIOLATED"}}, ["L2"])}
    F._apply_score(group, {"per_capsule": [row]})
    ob = group["obligations"]["a"]
    assert ob["status"] == "fail" and ob["failure_category"] == "MUST_REFUSE_VIOLATED" and ob["outcome"] == "fail"

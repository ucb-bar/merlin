"""The grader must judge a lane report on the SHAPE the producer actually emits.

`lane_report` reports the evidence rung PER LANE, because one label for the whole report had to lie
about at least one lane. The grader compared that mapping against the bare string
"dynamic_dispatch_ledger", so the condition held for every report ever produced: every lane-declaring
whole-model capsule failed with `lane_report_missing_or_malformed`, naming the submission for a report
the harness builds itself. These tests pin both directions -- a real report must pass, and a report
resting only on the router's INTENT must still fail.
"""
from __future__ import annotations

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_grade import model_execution_check


def _capsule(require: list[str]) -> dict:
    return {"lanes": {"require": require}}


def _ledger(lane: str = "on_mesh") -> dict:
    """The ordered dispatch ledger `observed_lanes` is derived from."""
    return {"dispatch_ledger": [{"ordinal": 0, "symbol": "mm0", "lane": lane, "status": "pass"}]}


def _result(evidence: dict, *, observed: list[str], unexercised: list[str],
            lane: str = "on_mesh") -> dict:
    """A result whose lane report is the only thing under test."""
    return {
        "mesh_execution": _ledger(lane),
        "lane_report": {"required": sorted(evidence), "observed": observed,
                        "unexercised": unexercised, "evidence": evidence},
    }


def _lane_violations(caps: dict, res: dict) -> list[str]:
    return [v for v in model_execution_check(res, caps).get("violations", [])
            if "lane" in v]


def test_producer_emits_a_mapping_not_a_string():
    """The contract this test defends: `evidence` is per-lane, so a string compare can never hold."""
    rep = CR.lane_report(_capsule(["on_mesh"]), {"on_mesh": {"matmul": 3}}, {"matmul": 3})
    assert isinstance(rep["evidence"], dict)
    assert rep["evidence"] != "dynamic_dispatch_ledger"


def test_real_ledger_evidence_is_not_malformed():
    """A well-formed report must NOT be reported as malformed -- the defect being fixed."""
    res = _result({"on_mesh": "dynamic_dispatch_ledger"}, observed=["on_mesh"], unexercised=[])
    assert "lane_report_missing_or_malformed" not in _lane_violations(_capsule(["on_mesh"]), res)


def test_plan_only_evidence_still_fails():
    """MUTATION: intent is not execution. A router that ASSIGNED the lane proves nothing ran there."""
    res = _result({"on_mesh": "routing_plan"}, observed=["on_mesh"], unexercised=[])
    assert "required_lane_evidenced_by_plan_only" in _lane_violations(_capsule(["on_mesh"]), res)


def test_aggregate_execution_evidence_is_accepted():
    res = _result({"on_mesh": "execution"}, observed=["on_mesh"], unexercised=[])
    assert _lane_violations(_capsule(["on_mesh"]), res) == []


def test_non_mapping_evidence_is_malformed():
    """The shape check must still be able to fail: a bare string is not a lane report."""
    res = {"mesh_execution": _ledger(),
           "lane_report": {"required": ["on_mesh"], "observed": ["on_mesh"],
                           "unexercised": [], "evidence": "dynamic_dispatch_ledger"}}
    assert "lane_report_missing_or_malformed" in _lane_violations(_capsule(["on_mesh"]), res)


def test_vocabulary_is_shared_with_the_producer():
    """One exported vocabulary, so the two ends cannot drift apart again."""
    assert "routing_plan" not in CR.EXECUTED_LANE_EVIDENCE
    assert set(CR.EXECUTED_LANE_EVIDENCE) == {
        "dynamic_dispatch_ledger", "execution", CR.WHOLE_PROGRAM_COMPLETION_EVIDENCE}


# --- the scope the report is ABOUT ------------------------------------------------------------------
# Second instance of the same defect class: the producer judges only the lanes a capsule DECLARED, the
# grader derives its set from the dispatch ledger, which sees every lane that ran. Compared whole, any
# capsule dispatching on more lanes than it declared disagreed on every grade. Measured on
# SY_micro_model: `require: [on_mesh]`, a ledger carrying on_mesh + scalar_rvv_lane, and
# `lane_report_disagrees_with_dispatch_ledger` in all seven rounds that reached a grade.


def _mixed_ledger() -> dict:
    """One matmul on the mesh, one host call -- a model running on more lanes than it declares."""
    return {"dispatch_ledger": [
        {"ordinal": 0, "symbol": "mm0", "lane": "on_mesh", "status": "pass"},
        {"ordinal": 1, "symbol": "norm0", "lane": "scalar_rvv_lane", "status": "pass"},
    ]}


def test_producer_never_observes_a_lane_the_capsule_did_not_declare():
    """The property the grader is entitled to rely on, stated so it cannot quietly change."""
    rep = CR.lane_report(_capsule(["on_mesh"]), None, _mixed_ledger())
    assert rep["observed"] == ["on_mesh"]
    assert set(rep["observed"]) <= set(rep["scope"]), "observed must lie inside the declared scope"
    assert rep["scope"] == ["on_mesh"], "scope is what the report is about, not what ran"


def test_declaring_fewer_lanes_than_you_dispatch_on_is_not_a_disagreement():
    """SY_micro_model's shape: the extra lane is outside the report's scope, so it contradicts nothing."""
    caps = _capsule(["on_mesh"])
    res = {"mesh_execution": _mixed_ledger(),
           "lane_report": CR.lane_report(caps, None, _mixed_ledger())}
    assert "lane_report_disagrees_with_dispatch_ledger" not in _lane_violations(caps, res)


def test_a_report_still_fails_when_it_disagrees_INSIDE_its_own_scope():
    """The mutation: keep the scope honest and claim a declared lane the ledger never carried."""
    caps = _capsule(["on_mesh", "scalar_rvv_lane"])
    rep = CR.lane_report(caps, None, _ledger("on_mesh"))
    rep["observed"] = ["on_mesh", "scalar_rvv_lane"]  # scalar never ran
    res = {"mesh_execution": _ledger("on_mesh"), "lane_report": rep}
    assert "lane_report_disagrees_with_dispatch_ledger" in _lane_violations(caps, res)


def test_a_report_without_a_scope_is_compared_whole():
    """Back-compat: a report predating the key is judged the old way, which is right exactly when the
    capsule declares every lane it dispatches on."""
    caps = _capsule(["on_mesh"])
    rep = CR.lane_report(caps, None, _ledger("on_mesh"))
    rep.pop("scope")
    res = {"mesh_execution": _ledger("on_mesh"), "lane_report": rep}
    assert "lane_report_disagrees_with_dispatch_ledger" not in _lane_violations(caps, res)

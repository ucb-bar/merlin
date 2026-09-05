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
    assert set(CR.EXECUTED_LANE_EVIDENCE) == {"dynamic_dispatch_ledger", "execution"}

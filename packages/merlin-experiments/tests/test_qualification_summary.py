"""An execution score cannot silently stand in for required source/trace evidence."""

import copy

import pytest
from merlin_experiments.phase1.feedback.qualification_summary import summarize


def candidate():
    return {
        "status": "pass",
        "required_tiers": ["L0", "L1", "L2", "L3"],
        "tiers": dict.fromkeys(["L0", "L1", "L2", "L3"], "pass"),
        "source_coverage": {"status": "verified"},
        "components": {
            "emitted_host_compute": "clean",
            "source_placement": "clean",
            "completed_dispatch": "verified",
            "candidate_required_tiers": "pass",
        },
    }


def test_execution_score_exposes_missing_trace_evidence_and_separates_refusals():
    rows = [{"status": "pass", "trace_status": "pass", "trace_requirement_status": "pass"}] * 85
    rows += [{"status": "pass", "trace_status": "pass", "trace_requirement_status": "unavailable_protocol"}] * 47
    rows += [{"status": "pass", "expected_refusal": {"status": "met"}}]
    rows += [{"status": "incomplete", "candidate_native_verification": {"status": "incomplete"}}] * 6
    rows += [{"status": "fail", "trace_status": "pass", "trace_requirement_status": "pass"}]
    counts = summarize(rows)
    assert counts["n_completed"] == 140 and counts["result_passed"] == 133
    assert counts["execution_passed"] == 132 and counts["expected_refusals_met"] == 1
    assert counts["execution_passes_with_trace_evidence"] == 85
    assert counts["execution_passes_without_trace_evidence"] == counts["trace_evidence_unavailable"] == 47
    assert counts["model_programs_unverified"] == 6


@pytest.mark.parametrize("missing", ["required_tiers", "source_coverage", "components", "L3", "dispatch"])
def test_pass_label_without_complete_closed_model_evidence_is_unverified(missing):
    native = candidate()
    if missing == "L3":
        native["tiers"]["L3"] = "unverified"
    elif missing == "dispatch":
        native["components"]["completed_dispatch"] = "unknown"
    else:
        native.pop(missing)
    counts = summarize([{"status": "pass", "candidate_native_verification": native}])
    assert counts["model_programs_verified"] == 0 and counts["model_programs_unverified"] == 1


def test_complete_model_counts_remain_distinct_from_ordinary_trace_and_final_gates():
    row = {"status": "pass", "candidate_native_verification": candidate()}
    before = copy.deepcopy(row)
    counts = summarize([row])
    assert counts["execution_passed"] == counts["model_programs_verified"] == 1
    assert counts["execution_passes_with_trace_evidence"] == 0
    assert "all_pass" not in counts and "certified_complete" not in counts
    assert before == row

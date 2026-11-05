"""A deferred test case says which gate holds it and how it can run; nothing about its status changes."""

from __future__ import annotations

import json

from merlin_experiments.phase1.feedback import certification_gates as G
from merlin_experiments.phase1.feedback import joint_test_scope
from merlin_experiments.phase1.feedback.qa import _redact_detail


def test_an_operator_gate_deferral_states_observed_and_required_fractions():
    # The grade summary entry the runner's blocked() row produces (capsule_grade: gate_reason).
    row = {"capsule": "SY_source_coverage_mlp", "kind": "model", "status": "gated",
           "gate_reason": "whole-model capsule deferred: op pass fraction 0.62 < gate 0.8"}
    fields = G.gated_row_fields(row, _redact_detail)
    assert fields["failure_plane"] == "gate" and fields["failure_category"] == "GATED"
    assert fields["failure_detail"] and "0.62" not in fields["failure_detail"]  # the ordinary scrub applies
    gate = fields["certification_gate"]
    assert gate["observed_operator_pass_fraction"] == 0.62 and gate["required_operator_pass_fraction"] == 0.8
    assert "agent_selfcheck.py --capsules" in gate["how_to_run_it"] and gate["advisory_only"] is True


def test_an_admission_deferral_says_the_grade_closed_before_the_gate_opened():
    # capsule_scheduling.deferred_results(): admission closed before this test case started.
    row = {"capsule": "SY_micro_model", "kind": "model", "status": "gated",
           "background_deferred": "snapshot admission deadline reached"}
    fields = G.gated_row_fields(row, _redact_detail)
    assert fields["failure_category"] == "ADMISSION_DEFERRED"
    assert "stopped admitting" in fields["certification_gate"]["deferred_because"]
    assert "observed_operator_pass_fraction" not in fields["certification_gate"]


def test_only_deferred_rows_with_a_reason_are_annotated():
    assert G.gated_row_fields({"status": "pass", "gate_reason": "x"}, _redact_detail) == {}
    assert G.gated_row_fields({"status": "gated", "kind": "model"}, _redact_detail) == {}
    operator = G.gated_row_fields({"status": "gated", "kind": "layer",
                                   "background_deferred": "snapshot admission deadline reached"}, _redact_detail)
    assert operator["failure_category"] == "ADMISSION_DEFERRED" and "certification_gate" not in operator


def test_calibration_access_reports_this_grade_and_the_initial_diagnostics(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control

    ledger = tmp_path / "joint_gsim_budget.json"
    ledger.write_text(json.dumps({"reserved_s": 0, "active": [], "admissions": [{"capsule": "PC01_k128"}]}))
    monkeypatch.setattr(joint_control, "selected", lambda: {"policy": {"operator_measurement_policy": {}}})
    monkeypatch.setattr(joint_control, "gsim_ledger", lambda: ledger)
    rows = [{"capsule": "PC01_k128", "tiers": {"L2": "pass"}}, {"capsule": "PK00_k16", "tiers": {"L2": "fail"}}]
    witnesses = [{"capsule": "PC01_k128"}, {"capsule": "PK00_k16"}, {"capsule": "PL00_k16"}]
    access = G.calibration_access(rows, witnesses)
    assert access["spike_l2_this_grade"] == {"PC01_k128": "pass", "PK00_k16": "fail",
                                             "PL00_k16": "not_run_in_this_grade"}
    assert access["spike_l2_passes_this_grade"] == 1
    assert access["initial_diagnostics"] == {"allowed": 2, "used": 1, "remaining": 1}
    assert "--sim gsim --capsules" in access["request"] and access["advisory_only"] is True


def test_calibration_access_without_a_campaign_omits_the_budget(monkeypatch):
    from merlin_experiments.phase1 import joint_control

    monkeypatch.setattr(joint_control, "selected", lambda: None)
    access = G.calibration_access([], [{"capsule": "PC00_k64"}])
    assert "initial_diagnostics" not in access
    assert access["spike_l2_this_grade"] == {"PC00_k64": "not_run_in_this_grade"}


def test_the_verdict_brief_carries_gsim_access_for_selected_witnesses(monkeypatch):
    from merlin_experiments.phase1 import joint_control

    monkeypatch.setattr(joint_control, "selected", lambda: None)
    witness = {"capsule": "PC01_k128", "correctness": "pending_matching_spike_pass", "question": "q",
               "physical_status": "unknown", "dispatch_authorized": False}
    report = {"original_development_required": 1, "original_unresolved": 1, "next_batch": ["PC01_k128"],
              "physical_rule": "r", "parallel_large_spike": "p", "correctness_scope": "s",
              "exact_correctness_reused": [], "witness_prerequisites": [witness]}
    selection = {"selected": ["PC01_k128"], "pending": [], "required_development": 1, "scope": "joint",
                 "curriculum": report, "performance_cohorts": {}}
    verdict = {"per_capsule": [{"capsule": "PC01_k128", "status": "pass", "tiers": {"L2": "pass"}}], "n_passed": 1}
    out = joint_test_scope.annotate(verdict, selection)
    prerequisites = out["feedback_brief"]["correctness_curriculum"]["calibration_prerequisites"]
    assert prerequisites["matched_spike_passes"] == 0  # the selection-time count is unchanged
    assert prerequisites["gsim_access"]["spike_l2_this_grade"] == {"PC01_k128": "pass"}

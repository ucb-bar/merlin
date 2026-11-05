"""A model run that stops before its receipt reports why it stopped, not that the receipt is absent.

Measured on one run: every SY_micro_model / SY_source_* attempt read "incomplete: owned host model
receipt or original frozen capsule is absent" (NOT_RUN_IS_NOT_PASS, counted as tooling), while the
result's own candidate_emission_failure said the candidate's emitted code lacked attributed
contraction work for an eligible source operation -- a compiler failure.
"""

from __future__ import annotations

import pytest
from merlin_experiments.phase1.convergence_report import classify_result

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import owned_host_model_execution as OH
from merlin.targetgen import owned_host_model_selection

REFUSAL = (
    "eligible source operation 9 lacks unavoidable attributed contraction work; "
    "fused transforms need independent evidence"
)


def _stopped(exc, stage="owned_host_execution"):
    try:
        raise exc
    except Exception as raised:  # noqa: BLE001 -- the runner records any stop this way
        return CR._candidate_exception_diagnostic(raised, stage=stage)


def _model_result(stopped):
    return {"capsule": "SY_micro_model", "kind": "model", "status": "incomplete", "candidate_emission_failure": stopped}


def test_the_dispatch_attribution_rule_raises_the_candidate_refusal(monkeypatch):
    from merlin.targetgen import native_dispatch_accounting

    monkeypatch.setattr(native_dispatch_accounting, "_mandatory_command_blocks", lambda function, matched: False)
    obligation = {"semantic_family": "contraction", "region_id": "r9", "source_op_index": 9}
    with pytest.raises(OH.CandidateDispatchRefusal, match="lacks unavoidable attributed contraction work"):
        OH._mandatory_source_commands(object(), {"contraction": set()}, [obligation])
    assert issubclass(OH.CandidateDispatchRefusal, ValueError)  # every existing ValueError handler still applies


def test_a_candidate_refusal_is_the_reported_compiler_failure():
    check = OH.audit_owned_host_model(_model_result(_stopped(OH.CandidateDispatchRefusal(REFUSAL))), None, target="t")
    assert check["status"] == "fail"
    assert check["failure"] == {
        "plane": "candidate_compilation",
        "category": "CANDIDATE_EMISSION_FAILED",
        "detail": REFUSAL,
        "stage": "owned_host_execution",
    }
    assert "receipt" not in check["detail"]


def test_any_other_stop_stays_incomplete_but_names_its_cause_without_host_paths():
    stop = _stopped(FileNotFoundError("/srv/private/goldens/x.npz is missing"), stage="prepare_runtime_bundle")
    check = OH.audit_owned_host_model(_model_result(stop), None, target="t")
    assert check["status"] == "incomplete"
    assert check["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    detail = check["failure"]["detail"]
    assert detail.startswith("the model run stopped at prepare_runtime_bundle with FileNotFoundError")
    assert "/srv/private" not in detail and "<host-path>" in detail
    # A refusal class raised outside the owned-host execution stage is not trusted as one.
    misplaced = _stopped(OH.CandidateDispatchRefusal(REFUSAL), stage="verify_runtime_bundle")
    assert OH.audit_owned_host_model(_model_result(misplaced), None, target="t")["status"] == "incomplete"


def test_without_any_recorded_stop_the_absent_receipt_is_still_incomplete():
    check = OH.audit_owned_host_model({"capsule": "M", "kind": "model"}, None, target="t")
    assert check["status"] == "incomplete" and "receipt" in check["detail"] and "failure" not in check


def test_parent_budget_stop_survives_missing_receipt_audit(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("a stopped child returned no receipt to audit")

    monkeypatch.setattr(CG, "model_execution_check", unexpected)
    failure = {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS", "detail": "stopped after 600s"}
    row = {"status": "budget_exhausted", "failure": failure, "model_budget_s": 600, "elapsed_s": 614.7}
    assert CG.enforce_model_execution_check(row, {}, target="t") is row
    assert row["status"] == "budget_exhausted" and row["failure"] == failure
    assert row["numeric"] == {} and row["tiers"] == {}
    assert row["model_execution_check"]["detail"] == failure["detail"]


def test_budget_label_cannot_bypass_an_existing_native_receipt(monkeypatch):
    seen = []

    def audit(*args, **kwargs):
        seen.append(True)
        return {"status": "incomplete", "kind": "owned_host_model_execution"}

    monkeypatch.setattr(CG, "model_execution_check", audit)
    row = {"status": "budget_exhausted", "owned_host_native_execution": {"status": "verified"}}
    CG.enforce_model_execution_check(row, {}, target="t")
    assert seen == [True] and row["status"] == "incomplete"


def test_the_graded_row_carries_the_root_cause_and_counts_as_a_genuine_failure(monkeypatch):
    monkeypatch.setattr(owned_host_model_selection, "owned_model_required", lambda capsule, *, target: True)
    result = _model_result(_stopped(OH.CandidateDispatchRefusal(REFUSAL)))
    CG.enforce_model_execution_check(result, {"name": "SY_micro_model", "kind": "model"}, target="t")
    assert result["status"] == "fail"
    assert result["failure"]["category"] == "CANDIDATE_EMISSION_FAILED"
    assert result["failure"]["detail"] == REFUSAL
    assert result["model_execution_check"]["violations"] == ["candidate_dispatch_attribution_refused"]
    assert classify_result(result["status"], result["failure"])[0] == "genuine_fail"
    # NOT_RUN_IS_NOT_PASS is unchanged for a stop that is not the candidate's refusal.
    other = _model_result(_stopped(RuntimeError("simulator slot lost")))
    CG.enforce_model_execution_check(other, {"name": "SY_micro_model", "kind": "model"}, target="t")
    assert other["status"] == "incomplete" and other["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert "simulator slot lost" in other["failure"]["detail"]
    assert classify_result(other["status"], other["failure"])[0] == "tooling_incomplete"

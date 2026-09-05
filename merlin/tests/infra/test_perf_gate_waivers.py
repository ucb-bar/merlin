"""A launch may proceed over a NAMED completeness gap, and never over an integrity one.

The gate had exactly one setting: refuse. So the only way to launch against a baseline with a known,
understood gap was to edit the gate — which is how a gate stops meaning anything. These tests pin the
override's three properties, which are what make it a record rather than a bypass:

* only COMPLETENESS predicates are waivable; the integrity ones (sandbox, answer mask, answer-access
  audit, cohort-admission accounting, public/hidden identity separation) refuse the waiver itself,
  because waiving one does not weaken a result, it makes it unattributable;
* the waiver is a set of NAMES, never a boolean — a blanket force would waive whatever failed next,
  including something nobody had looked at — and a name that did not actually fail is an error, so a
  stale waiver cannot pre-authorise a future failure;
* every accepted waiver travels with the result as a :class:`Deviation`, and `gate_clean` goes false.

The fixture is the one the gate's own tests use, so "clean" here means the same thing it does there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from test_perf_campaign_gate import PC, _functional_run


def _break_convergence(run: Path) -> None:
    summary = yaml.safe_load((run / "qa_loop_summary.yaml").read_text())
    summary["converged"] = False
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump(summary))


def _skip_finalize(run: Path) -> None:
    """Exactly what a submission-identity pin produces: finalize SKIPPED, so no post-finalize regrade."""
    summary = yaml.safe_load((run / "qa_loop_summary.yaml").read_text())
    summary["finalize"] = {"skipped": True, "reason": "submission identity is pinned"}
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump(summary))


def _drop_a_tier(run: Path) -> None:
    path = run / "grading_public" / "score_capsule.json"
    score = json.loads(path.read_text())
    score["per_capsule"][0]["tiers"] = {"L3": "pass"}      # a model row: no L2 key at all
    path.write_text(json.dumps(score))


def test_the_gate_still_refuses_by_default(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    _break_convergence(run)
    with pytest.raises(PC.CampaignGateError, match="qa_loop_not_converged"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_every_failure_is_named_at_once(tmp_path: Path) -> None:
    """A caller cannot acknowledge a gap it was never shown, so one run reports the whole picture."""
    run, digest = _functional_run(tmp_path)
    _break_convergence(run)
    _skip_finalize(run)
    _drop_a_tier(run)
    with pytest.raises(PC.CampaignGateError) as excinfo:
        PC.inspect_functional_run(tmp_path, run.name, digest)
    message = str(excinfo.value)
    for predicate in ("qa_loop_not_converged", "finalize_regrade_not_pass",
                      "capsule_tier_not_earned"):
        assert predicate in message, message
    assert "refused by 3 gate predicate(s)" in message


def test_a_named_waiver_launches_and_is_recorded(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    _break_convergence(run)
    _skip_finalize(run)
    record = PC.inspect_functional_run(
        tmp_path, run.name, digest,
        waive={"qa_loop_not_converged", "finalize_regrade_not_pass"})
    assert record.gate_clean is False
    assert {d.predicate for d in record.deviations} == {"qa_loop_not_converged",
                                                        "finalize_regrade_not_pass"}
    for deviation in record.deviations:
        assert deviation.detail, "a waived predicate must record what was actually observed"
        assert deviation.to_dict()["predicate"] == deviation.predicate


def test_a_clean_run_reports_gate_clean(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    record = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert record.gate_clean is True and record.deviations == ()


def test_a_partial_waiver_still_refuses_the_rest(tmp_path: Path) -> None:
    """Waiving one gap does not open the others — the refusal names only what is left."""
    run, digest = _functional_run(tmp_path)
    _break_convergence(run)
    _drop_a_tier(run)
    with pytest.raises(PC.CampaignGateError) as excinfo:
        PC.inspect_functional_run(tmp_path, run.name, digest, waive={"qa_loop_not_converged"})
    message = str(excinfo.value)
    assert "capsule_tier_not_earned" in message
    assert "refused by 1 gate predicate(s)" in message


@pytest.mark.parametrize("predicate", sorted(PC.UNWAIVABLE))
def test_no_integrity_predicate_can_be_waived(tmp_path: Path, predicate: str) -> None:
    """Refused as a REQUEST, before any evidence is read: this is not a permission that can be given."""
    run, digest = _functional_run(tmp_path)
    with pytest.raises(PC.CampaignGateError, match="UNWAIVABLE"):
        PC.inspect_functional_run(tmp_path, run.name, digest, waive={predicate})


def test_the_integrity_set_covers_the_things_that_make_numbers_mean_anything(tmp_path: Path) -> None:
    for predicate in ("sandbox_not_bwrap", "answer_mask_vacuous", "round_answer_access_unclean",
                      "finalize_answer_access_unclean", "isolation_audit_unclean",
                      "capsule_identity_reused", "score_integrity_failed",
                      "bundle_input_snapshot_incomplete", "cohort_admission_missing",
                      "cohort_admission_does_not_close", "excluded_name_set_unpinned"):
        assert predicate in PC.UNWAIVABLE, predicate
    assert not (PC.UNWAIVABLE & PC._WAIVABLE_PREDICATES), "a predicate cannot be both"


def test_an_unknown_predicate_name_is_an_error(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    with pytest.raises(PC.CampaignGateError, match="unknown gate predicate"):
        PC.inspect_functional_run(tmp_path, run.name, digest, waive={"converged_maybe"})


def test_a_waiver_for_something_that_did_not_fail_is_an_error(tmp_path: Path) -> None:
    """A stale waiver would silently pre-authorise a failure nobody has seen."""
    run, digest = _functional_run(tmp_path)
    _break_convergence(run)
    with pytest.raises(PC.CampaignGateError, match="did not fail"):
        PC.inspect_functional_run(tmp_path, run.name, digest,
                                  waive={"qa_loop_not_converged", "capsule_tier_not_earned"})


def test_a_capability_exclusion_is_reconciled_not_refused(tmp_path: Path) -> None:
    """The hidden holdout's sealed count exceeds its graded count by its recorded exclusions.

    Measured on gemmini: 11 sealed held-out capsules, 1 excluded by the frozen hardware-capability
    predicate (a bf16 capsule this target cannot execute, the same predicate that excluded its 11
    public bf16 siblings), 10 graded 10/10. Comparing the sealed total against the graded total
    refused that honest run; reconciling through the admission record accepts it.
    """
    run, digest = _functional_run(tmp_path)
    environment = yaml.safe_load((run / "environment.yaml").read_text())
    environment["task_scope"]["held_out_capsules"] = 2        # one more than is gradeable
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    path = run / "grading_hidden" / "score_capsule.json"
    score = json.loads(path.read_text())
    score["cohort_admission"].update({"policy": "frozen_target_capability_operand_dtype",
                                      "n_source_capsules": 2, "n_capability_excluded": 1,
                                      "excluded_name_set_sha256": "c" * 64})
    path.write_text(json.dumps(score))
    record = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert record.gate_clean is True, [d.to_dict() for d in record.deviations]
    assert record.hidden_capsules == 1


def test_an_unaccounted_missing_capsule_is_still_refused(tmp_path: Path) -> None:
    """The reconciliation is not a hole: an exclusion nobody recorded still fails."""
    run, digest = _functional_run(tmp_path)
    environment = yaml.safe_load((run / "environment.yaml").read_text())
    environment["task_scope"]["held_out_capsules"] = 2
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(PC.CampaignGateError, match="task_scope_hidden_mismatch"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_an_exclusion_without_a_pinned_name_set_is_refused(tmp_path: Path) -> None:
    """An exclusion is evidence only if the excluded names are pinned; otherwise it is a gap."""
    run, digest = _functional_run(tmp_path)
    path = run / "grading_hidden" / "score_capsule.json"
    score = json.loads(path.read_text())
    score["cohort_admission"].update({"n_source_capsules": 2, "n_capability_excluded": 1})
    path.write_text(json.dumps(score))
    with pytest.raises(PC.CampaignGateError, match="excluded_name_set_unpinned"):
        PC.inspect_functional_run(tmp_path, run.name, digest)

"""Synthetic completed public rows cannot waive unavailable trace obligations."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import formal, qa, selfcheck

from merlin.targetgen import capsule_grade, trace_check


@pytest.mark.parametrize("requirement", [None, "pass", "unavailable_protocol", "unavailable_fields", "failed"])
def test_observed_pass_requires_available_selected_trace_evidence(tmp_path, monkeypatch, requirement):
    cap_root = tmp_path / "corpus" / "case"
    cap_root.mkdir(parents=True)
    capsule = {"name": "case", "kind": "isa", "label": "public", "required_oracle_tiers": ["L2", "L3"]}
    (cap_root / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    row = {
        "capsule": "case", "kind": "isa", "label": "public", "status": "pass",
        "numeric": {"status": "pass", "mismatch_count": 0},
        "tiers": {"L2": {"status": "pass"}, "L3": {"status": "pass", "derived_from_rtl": True}},
        "trace_check": {"status": "pass"},
    }
    if requirement is not None:
        row["trace_check"]["trace_requirement_status"] = requirement
    monkeypatch.setattr(capsule_grade, "source_experiment_env", lambda target: {})
    monkeypatch.setattr(capsule_grade, "load_package", lambda *a, **k: SimpleNamespace(integrity_exempt=False))
    monkeypatch.setattr(capsule_grade, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(capsule_grade, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(capsule_grade.CR, "discover_capsules", lambda *a, **k: [capsule])
    monkeypatch.setattr(capsule_grade.CR, "suite_for", lambda target: "fixture-suite")
    monkeypatch.setattr(capsule_grade.CR, "_rtl_tiers_of", lambda target: {"L3"})
    monkeypatch.setattr(capsule_grade.CR, "run_suite", lambda *a, **k: [deepcopy(row)])
    score = capsule_grade.grade(
        tmp_path / "submission", capsules_root=cap_root.parent, runs_root=tmp_path / "grade",
        target="fixture", max_workers=1, oracle_adapters={},
    )
    expected = requirement in (None, "pass")
    assert score["n_passed"] == score["n_capsules"] == 1
    assert score["numeric_all_exact"] is True and score["tier_reached"]["L3"] == 1
    assert score["pass_evidence"]["rtl_backed"] == 1
    assert score["trace_all_pass"] is expected
    assert score["structural_evidence_all_pass"] is expected
    assert score["structural_evidence_scope"]["n_instruction_trace_capsules"] == 1
    certified, screened, refused, _ = selfcheck._operator_feedback_state(
        row, capsule, target="fixture", barrier_status="pass",
    )
    assert (certified, screened, refused) == (expected, not expected, False)
    complete, failures = formal.phase_completion(score)
    assert complete is expected, failures
    if not expected:
        assert "trace_conformance_not_complete" in failures
        assert "structural_evidence_not_complete" in failures

    # Pass the actual grade score into the public QA projection as well.
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda context: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *a, **k: {"L2": object(), "L3": object()})
    monkeypatch.setattr(qa.CG, "grade", lambda *a, **k: score)
    context = InvocationContext(tmp_path, tmp_path / "target.yaml", tmp_path, "fixture",
                                tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles", ())
    verdict = qa.run("submission", str(cap_root.parent), tmp_path / "qa", {"public"}, False, 1, context=context)
    assert verdict["all_pass"] is expected
    assert verdict["n_passed"] == 1 and verdict["per_capsule"][0]["numeric_status"] == "pass"
    assert row["numeric"]["status"] == row["tiers"]["L3"]["status"] == "pass"


@pytest.mark.parametrize("observation", [None, {}, [], {"status": "fail"}, {"status": "skipped"},
                                        {"status": "pass", "trace_requirement_status": "invalid_binding"}])
def test_missing_or_nonpassing_trace_never_completes(observation):
    assert trace_check.requirements_complete(observation) is False

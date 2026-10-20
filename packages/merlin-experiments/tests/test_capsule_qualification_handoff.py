"""Reopen actual synthetic old observations; a saved pass is never sufficient."""

import json
import shutil

import pytest
import yaml
from merlin_experiments.phase1.feedback import capsule_qualification_handoff as Q
from test_capsule_certification import grade_and_freeze
from test_capsule_certification import selected as selected


@pytest.fixture
def qualified(selected, tmp_path, monkeypatch):
    s = selected
    grade_and_freeze(s)
    selection, verdict = tmp_path / "selection.json", tmp_path / "resources.json"
    selection.write_text("{}")
    verdict.write_text("{}")

    def resource_verifier(run, *, selection_path, verdict_path):
        assert run == s.run
        return {
            "selection": Q.C.pin(selection_path),
            "verdict": Q.C.pin(verdict_path),
            "result": s.resources,
            "products": [],
            "invocations": [],
        }

    monkeypatch.setattr(Q.G, "verify_retained_resource_observation", resource_verifier, raising=False)
    out = tmp_path / "new-owner"
    record = Q.publish(s.run, out, descriptor=s.descriptor, resource_selection=selection, resource_verdict=verdict)
    return s, out, record


def test_separate_qualification_reopens_old_grades_without_publication_claim(qualified):
    s, out, record = qualified
    assert record["capsule_complete"] is record["external_validation_ready"] is True
    assert record["authoring_converged"] is record["phase2_iteration_ready"] is False
    assert record["formal_grade_complete"] is record["external_full_models_complete"] is False
    assert not (s.run / "capsule_certification.json").exists()
    assert record["numerical_producer_epoch"]["public"]["score"]["path"].startswith(str(s.run))
    assert record["handoff"]["compiler"]["path"] == str(out / "compiler")
    assert Q.verify(out / Q.RECEIPT, descriptor=s.descriptor) == record


@pytest.mark.parametrize("fault", ["score", "candidate", "resource", "export", "observation", "claim"])
def test_qualification_rejects_changed_original_or_supplemental_evidence(qualified, fault):
    s, out, record = qualified
    path = {
        "score": s.run / "grading_hidden/score_capsule.json",
        "candidate": s.run / "submission/compiler.py",
        "resource": record["resource_producer_epoch"]["verdict"]["path"],
        "export": out / "compiler/compiler.py",
        "observation": record["verification"]["path"],
        "claim": out / Q.RECEIPT,
    }[fault]
    from pathlib import Path

    path = Path(path)
    path.chmod(0o600)
    if fault == "claim":
        value = json.loads(path.read_bytes())
        value["authoring_converged"] = True
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, KeyError)):
        Q.verify(out / Q.RECEIPT, descriptor=s.descriptor)


def external_fixture(qualified, monkeypatch, tmp_path):
    s, out, record = qualified
    run = tmp_path / "external"
    run.mkdir()
    shutil.copytree(s.run / "submission", run / "submission")
    (run / "input_bundle_manifest.yaml").write_text("{}")
    (run / "environment.yaml").write_text(yaml.safe_dump({"qualification_only": True, "validation_scope": "full"}))
    monkeypatch.setenv(Q.ENV, json.dumps(Q.C.pin(out / Q.RECEIPT)))
    return s, run, record


def test_external_reuse_requires_same_selected_source_rosters_facts_and_contract(qualified, monkeypatch, tmp_path):
    s, run, record = external_fixture(qualified, monkeypatch, tmp_path)
    args = dict(
        context=s.context,
        candidate=run / "submission",
        capsules=s.public,
        hidden_capsules=s.hidden,
        contract=s.contract,
        facts=None,
    )
    assert Q.selected(run, **args) == record
    wrong = tmp_path / "different"
    wrong.mkdir()
    (wrong / "input.txt").write_text("different roster")
    with pytest.raises(ValueError, match="selection differs"):
        Q.selected(run, **{**args, "hidden_capsules": wrong})
    (run / "environment.yaml").write_text(yaml.safe_dump({"qualification_only": False, "validation_scope": "full"}))
    with pytest.raises(ValueError, match="unpaid"):
        Q.selected(run, **args)


def test_absent_selection_preserves_existing_formal_path(monkeypatch, tmp_path):
    monkeypatch.delenv(Q.ENV, raising=False)
    assert (
        Q.selected(
            tmp_path, context=None, candidate=None, capsules=None, hidden_capsules=None, contract=None, facts=None
        )
        is None
    )


def test_actual_formal_route_reuses_capsules_but_keeps_private_gate_incomplete(qualified, monkeypatch, tmp_path):
    from merlin_experiments.phase1 import oot_history
    from merlin_experiments.phase1.feedback import formal

    s, run, record = external_fixture(qualified, monkeypatch, tmp_path)
    oot_history.start(run)
    monkeypatch.setattr(formal, "resolve_context", lambda *a: s.context)
    monkeypatch.setattr(formal, "_verify_formal_readback_policy", lambda *a: None)
    monkeypatch.setattr(formal, "_install_formal_model_simulator", lambda *a: ({"engine": "fixture"}, None))
    monkeypatch.setattr(formal.PFM, "requirements_for", lambda *a: ())
    monkeypatch.setattr(formal.PFM, "program_requirements_for", lambda *a: {})
    monkeypatch.setattr(formal.PFM, "loader_env_requirements_for", lambda *a: {})
    monkeypatch.setattr(formal.PFX, "gate_for", lambda *a, **kw: None)
    monkeypatch.setattr(formal.CR, "oracle_adapters", lambda *a: {})
    monkeypatch.setattr(formal, "_score", lambda *a, **kw: pytest.fail("new numerical grade started"))
    monkeypatch.setattr(Q.G, "run_gate", lambda **kw: pytest.fail("resource gate rerun"))
    result = formal._main(
        [
            "--run-dir",
            str(run),
            "--arm",
            "baseline",
            "--capsules",
            str(s.public),
            "--hidden-capsules",
            str(s.hidden),
            "--contract",
            str(s.contract),
        ],
        context=s.context,
    )
    assert result == 1  # Capsule reuse never supplies missing external-model evidence.
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_bytes())
    assert manifest["capsule_observation_reuse"]["qualification"] == record["receipt"]
    assert manifest["capsule_observation_reuse"]["new_capsule_execution"] is False
    assert manifest["public_dev"]["formal_complete"] is manifest["hidden"]["formal_complete"] is True
    assert manifest["completion"]["formal_grade_complete"] is False
    assert not (run / "grading_public/score_capsule.json").exists()
    assert not (run / "grading_hidden/score_capsule.json").exists()

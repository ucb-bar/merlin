"""A failed resource prerequisite stops before any expensive grading work."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase1.feedback import formal
from merlin_experiments.phase1.feedback import public_scalability_gate as G
from test_capsule_certification import selected as selected_fixture
from test_public_codegen_resource_policy import policy

from merlin.targetgen import target_experiment

selected = selected_fixture


@pytest.mark.parametrize("result", [
    {"required": True, "resource_complete": False, "status": "incomplete"},
    {"required": True, "resource_complete": False, "status": "unavailable"},
    {"required": True, "resource_complete": True, "status": "unavailable"},
    {},
])
def test_required_resource_failure_precedes_simulator_host_projection_and_grade(selected, monkeypatch, result):
    monkeypatch.setattr(G, "run_gate", lambda **kwargs: result)

    def forbidden(*args, **kwargs):
        pytest.fail("failed resource prerequisite started expensive grading preparation")

    monkeypatch.setattr(formal, "_install_formal_model_simulator", forbidden)
    monkeypatch.setattr(formal, "_score", forbidden)
    with pytest.raises(G.PublicResourceGateError, match="numerical grading was not started"):
        formal._main([
            "--run-dir", str(selected.run), "--arm", "baseline",
            "--capsules", str(selected.public), "--hidden-capsules", str(selected.hidden),
            "--contract", str(selected.contract), "--validation-scope", "capsules",
        ], context=selected.context)
    assert not (selected.run / "grading_public").exists()
    assert not (selected.run / "freeze.json").exists()


@pytest.mark.parametrize("result", [
    {"required": False, "resource_complete": False, "status": "not_selected"},
    {"required": False, "resource_complete": False, "status": "incomplete"},
    {"required": True, "resource_complete": True, "status": "accepted"},
])
def test_unselected_optional_and_completed_resources_keep_selected_semantics(result):
    G.require_complete(result)


def test_preparation_exception_is_retained_privately_with_safe_early_feedback(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    detail = f"private owner mismatch at {tmp_path}/private-golden-secret"

    def broken(_run):
        raise ValueError(detail)

    monkeypatch.setattr(R, "functional_service", broken)
    result = G.run_gate(run_dir=run, candidate=tmp_path / "candidate", context=None,
                        capsules_root=None, contract=None, facts_path=None)
    saved = json.loads((run / "public_codegen_resources.json").read_bytes())
    assert saved == result and saved["error"]["message"] == detail
    assert "ValueError" in saved["error"]["traceback"]
    assert (run / "public_codegen_resources.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(G.PublicResourceGateError) as caught:
        G.require_complete(result)
    assert str(tmp_path) not in str(caught.value) and "private-golden" not in str(caught.value)


def test_emission_exception_remains_exact_in_protected_result_and_public_projection_is_safe(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compile_only

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "tool.py").write_text("source")
    owner = CommonTransformService.__new__(CommonTransformService)
    owner.profile = {"public_codegen_resource_policy": policy(), "sha256": "a" * 64}
    owner._verify = lambda: None
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: None)
    detail = f"exact nested owner failure: {tmp_path}/operator-only"

    def broken(**_kwargs):
        raise ValueError(detail)

    monkeypatch.setattr(compile_only, "run", broken)
    args = dict(owner=owner, candidate=candidate, context=SimpleNamespace(descriptor=tmp_path),
                capsules_root=tmp_path, contract=tmp_path, facts_path=tmp_path / "facts")
    protected, actual = G._observe(**args, work_root=tmp_path / "protected")
    assert protected["error"]["message"] == detail and actual is None
    public = G.observe_authoring(**args, work_root=tmp_path / "public")
    assert public["resource_complete"] is False and public["samples"] == []
    assert "original public scale observation unavailable" == public["reason"]
    assert str(tmp_path) not in json.dumps(public) and "operator-only" not in json.dumps(public)

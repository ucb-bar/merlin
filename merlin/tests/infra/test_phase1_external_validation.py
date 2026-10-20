"""Independent model intake must not accept another compiler or an unready handoff."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import capsule_certification as C

from merlin.common.digest import sha256_file


@pytest.fixture
def modules():
    root = next(
        parent for parent in Path(__file__).resolve().parents if (parent / "examples/gemmini/phase1/launch").is_dir()
    )
    launch = root / "examples/gemmini/phase1/launch"
    sys.path.insert(0, str(launch))
    result = []
    try:
        for name in ("capsule_handoff_intake", "external_validation"):
            spec = importlib.util.spec_from_file_location(name, launch / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result.append(module)
        yield result
    finally:
        sys.path.remove(str(launch))


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def fixture_handoff(tmp_path, monkeypatch):
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: example\n")
    proof = tmp_path / "capsule_certification.json"
    proof.write_text(json.dumps({"descriptor": pin(descriptor)}))
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    digest = "a" * 64
    record = {
        "capsule_complete": True,
        "formal_grade_complete": False,
        "external_full_models_complete": False,
        "submission_sha256": digest,
        "handoff": {"compiler": {"path": str(compiler), "hash_tree_sha256": digest}},
    }
    record["descriptor"] = pin(descriptor)
    proof.write_text(json.dumps(record))
    published = tmp_path / "capsule_publication.json"
    published.write_text("{}")
    monkeypatch.setattr(C, "verify", lambda *a, **kw: record)
    monkeypatch.setattr(
        C,
        "verify_phase2_handoff",
        lambda *a, **kw: {
            "phase2_iteration_ready": True,
            "capsule_certification": pin(proof),
            "receipt": pin(published),
        },
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "new_author_allowed": False,
                "authorized_by_user_text": "Validate the frozen compiler separately.",
            }
        )
    )
    started = time.time()
    budget = tmp_path / "budget.json"
    data = {
        "schema": "merlin.operator.external_validation_budget.v1",
        "scope": "external_full_model_validation_after_capsule_closure",
        "capsule_receipt": pin(proof),
        "submission_sha256": digest,
        "authorized_at_unix": started,
        "wall_budget_s": 600,
        "absolute_deadline_unix": started + 600,
        "new_author_allowed": False,
        "authorization": pin(authorization),
    }
    budget.write_text(json.dumps(data))
    F = SimpleNamespace(package_identity=lambda p: {"path": str(p), "hash_tree_sha256": digest})
    return F, proof, compiler, budget, data, record


def test_admits_only_exact_ready_artifact_with_independent_budget(modules, tmp_path, monkeypatch):
    intake, _ = modules
    F, proof, compiler, budget, data, _ = fixture_handoff(tmp_path, monkeypatch)
    package, admission = intake.admit(
        F,
        proof,
        sha256_file(proof),
        compiler,
        data["absolute_deadline_unix"],
        budget_path=budget,
        budget_sha256=sha256_file(budget),
    )
    assert package["path"] == str(compiler)
    assert admission["phase1_clock_preserved"] is True
    assert admission["new_author_allowed"] is False


@pytest.mark.parametrize("fault", ["other_compiler", "unready", "extended_deadline", "changed_authorization"])
def test_rejects_artifact_readiness_and_budget_drift(modules, tmp_path, monkeypatch, fault):
    intake, _ = modules
    F, proof, compiler, budget, data, _record = fixture_handoff(tmp_path, monkeypatch)
    if fault == "other_compiler":
        F.package_identity = lambda p: {"path": str(p), "hash_tree_sha256": "b" * 64}
    elif fault == "unready":
        monkeypatch.setattr(C, "verify_phase2_handoff", lambda *a, **kw: {"phase2_iteration_ready": False})
    elif fault == "extended_deadline":
        data["absolute_deadline_unix"] += 1
        budget.write_text(json.dumps(data))
    else:
        Path(data["authorization"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        intake.admit(
            F,
            proof,
            sha256_file(proof),
            compiler,
            data["absolute_deadline_unix"],
            budget_path=budget,
            budget_sha256=sha256_file(budget),
        )


def test_rebinding_changes_only_external_paths_and_deadline(modules, tmp_path):
    _, external = modules
    original = {
        "schema": "example",
        "target": "example",
        "models": [{"id": "network", "capture": "same"}],
        "numerics_bundle": {"path": "same"},
        "numerical_qualification": {
            "config": "old",
            "static_root": "old",
            "deadline_unix": 1,
            "kernel_exact": True,
            "output_scope": "all_declared",
        },
    }
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(original))
    selected = tmp_path / "capsule.yaml"
    selected.write_text(yaml.safe_dump({k: v for k, v in original.items() if k != "numerical_qualification"}))
    rebound = external.rebound_spec(
        template,
        selected,
        config=tmp_path / "new.yaml",
        static_root=tmp_path / "native",
        deadline=100,
    )
    assert rebound["models"] == original["models"]
    assert rebound["numerics_bundle"] == original["numerics_bundle"]
    assert rebound["numerical_qualification"]["kernel_exact"] is True
    assert rebound["numerical_qualification"]["output_scope"] == "all_declared"
    original["models"][0]["capture"] = "changed"
    selected.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match="models"):
        external.rebound_spec(
            template,
            selected,
            config=tmp_path / "new.yaml",
            static_root=tmp_path / "native",
            deadline=100,
        )


def test_explicit_qualification_handoff_needs_no_authored_convergence(modules, tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import capsule_qualification_handoff as Q

    intake, _ = modules
    F, old, compiler, budget, data, record = fixture_handoff(tmp_path, monkeypatch)
    proof = tmp_path / Q.RECEIPT
    record.update(
        schema=Q.SCHEMA,
        qualification_only=True,
        authoring_converged=False,
        external_validation_ready=True,
        phase2_iteration_ready=False,
    )
    proof.write_text(json.dumps(record))
    verified = {**record, "receipt": pin(proof)}
    monkeypatch.setattr(Q, "verify", lambda *a, **kw: verified)
    monkeypatch.setattr(C, "verify_phase2_handoff", lambda *a, **kw: pytest.fail("authored route selected"))
    data["capsule_receipt"] = pin(proof)
    budget.write_text(json.dumps(data))
    package, result = intake.admit(
        F,
        proof,
        sha256_file(proof),
        compiler,
        data["absolute_deadline_unix"],
        budget_path=budget,
        budget_sha256=sha256_file(budget),
    )
    assert result["handoff_kind"] == "capsule_qualification"
    assert result["phase2_publication"] == pin(proof)
    assert package["hash_tree_sha256"] == record["submission_sha256"]
    verified["authoring_converged"] = True
    with pytest.raises(ValueError, match="independently ready"):
        intake.admit(
            F,
            proof,
            sha256_file(proof),
            compiler,
            data["absolute_deadline_unix"],
            budget_path=budget,
            budget_sha256=sha256_file(budget),
        )


@pytest.mark.parametrize("fault", [None, "preparation_expired", "too_long", "unselected_legacy"])
def test_qualification_execution_clock_starts_after_bounded_preparation(modules, tmp_path, monkeypatch, fault):
    from merlin_experiments.phase1.feedback import capsule_qualification_handoff as Q

    intake, _ = modules
    F, old, compiler, budget, data, record = fixture_handoff(tmp_path, monkeypatch)
    proof = tmp_path / Q.RECEIPT
    record.update(
        schema=Q.SCHEMA,
        qualification_only=True,
        authoring_converged=False,
        external_validation_ready=True,
        phase2_iteration_ready=False,
    )
    proof.write_text(json.dumps(record))
    monkeypatch.setattr(Q, "verify", lambda *a, **kw: {**record, "receipt": pin(proof)})
    data.update(
        schema="merlin.operator.external_validation_budget.v2",
        capsule_receipt=pin(proof),
        preparation_budget_s=30,
        absolute_deadline_unix=data["authorized_at_unix"] + 630,
    )
    if fault == "preparation_expired":
        monkeypatch.setattr(intake.time, "time", lambda: data["authorized_at_unix"] + 31)
    elif fault == "too_long":
        data["wall_budget_s"] = 7201
        data["absolute_deadline_unix"] = data["authorized_at_unix"] + 7231
    elif fault == "unselected_legacy":
        proof = old
        data["capsule_receipt"] = pin(proof)
    budget.write_text(json.dumps(data))

    def call():
        return intake.admit(
            F,
            proof,
            sha256_file(proof),
            compiler,
            data["absolute_deadline_unix"],
            budget_path=budget,
            budget_sha256=sha256_file(budget),
        )

    if fault:
        with pytest.raises(ValueError, match="execution budget"):
            call()
    else:
        _, grant = call()
        assert grant["execution_budget_s"] == 600
        assert grant["preparation_deadline_unix"] == data["authorized_at_unix"] + 30
        assert grant["authorized_ceiling_unix"] == data["absolute_deadline_unix"]

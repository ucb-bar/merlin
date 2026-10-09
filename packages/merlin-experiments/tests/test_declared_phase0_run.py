"""Closed declarations refuse substitution before fresh live issuance."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase0 import declared_run as D


@pytest.fixture
def declared(tmp_path):
    def pin(name):
        path = tmp_path / name
        path.write_text(name + "\n")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = tmp_path / "public"
    source.mkdir()
    request = {
        "schema": D.SCHEMA,
        "target": "independent_fixture",
        "inputs": {name: pin(name) for name in D._INPUTS},
        "operator_schemas": {
            "schema": D.S.SELECTION_SCHEMA,
            "status": "reviewed",
            "namespace": "aten",
            "python": pin("python"),
            "canonical_source": {"checkout": str(source), "commit": "a" * 40, "declarations": pin("native.yaml")},
        },
        "circt_opt": pin("circt-opt"),
        "forbidden_roots": [str(tmp_path / "excluded")],
        "automatic": {
            "schema": D.A.ORIGINAL_POLICY_SCHEMA,
            "status": "reviewed",
            "budget": {"max_members": 3, "max_interaction_cells": 2},
            "execution_budget": {
                "schema": "merlin.component_execution_budget.v1",
                "max_materialized_elements": 20,
                "max_reference_work": 20,
                "max_tensor_payload_bytes": 80,
                "max_scalar_bits": 64,
                "max_total_materialized_elements": 60,
                "max_total_reference_work": 60,
                "max_total_tensor_payload_bytes": 240,
            },
            "original_source_budget": {
                "schema": "merlin.original_call_source_budget.v1",
                "max_sources": 3,
                "max_tensor_elements": 20,
                "max_scalar_products": 20,
                "max_source_bytes": 1000,
                "max_total_tensor_elements": 60,
                "max_total_scalar_products": 60,
                "max_total_source_bytes": 3000,
            },
        },
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return request, path


@pytest.mark.parametrize(
    "change", ["saved_authority", "missing_source_budget", "missing_execution_budget", "extra_command"]
)
def test_request_cannot_import_authority_or_remove_construction_limits(declared, change):
    request, _ = declared
    request = copy.deepcopy(request)
    if change == "saved_authority":
        request["operator_schemas"]["software_intake_sha256"] = "0" * 64
    elif change == "extra_command":
        request["command"] = ["arbitrary-executable"]
    else:
        del request["automatic"]["original_source_budget" if change == "missing_source_budget" else "execution_budget"]
    with pytest.raises(ValueError):
        D.validate(request)


def test_changed_declared_bytes_refuse_before_issuer(declared, tmp_path, monkeypatch):
    request, path = declared
    Path(request["inputs"]["software_source"]["path"]).write_text("replacement")
    monkeypatch.setattr(
        D, "issue_independent_hardware_intake", lambda **kwargs: pytest.fail("stale source reached issuer")
    )
    with pytest.raises(ValueError, match="bytes changed"):
        D.run(path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_forbidden_inputs_refuse_before_issuer(declared, tmp_path, monkeypatch):
    request, path = declared
    request["forbidden_roots"].append(request["inputs"]["descriptor"]["path"])
    path.write_text(json.dumps(request))
    monkeypatch.setattr(
        D, "issue_independent_hardware_intake", lambda **kwargs: pytest.fail("excluded source reached issuer")
    )
    with pytest.raises(ValueError):
        D.run(path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_actual_issuer_failure_is_private_and_does_not_advance_phases(declared, tmp_path, monkeypatch):
    _, path = declared

    def fail(**kwargs):
        raise subprocess.CalledProcessError(1, ["explicit-native-tool"], stderr=b"original parser failure\n")

    monkeypatch.setattr(D, "issue_independent_hardware_intake", fail)
    monkeypatch.setattr(
        D, "issue_independent_software_intake", lambda **kwargs: pytest.fail("failed hardware issued software")
    )
    owner = tmp_path / "run"
    with pytest.raises(subprocess.CalledProcessError):
        D.run(path, output=owner)
    report = json.loads((owner / "report.json").read_bytes())
    assert report["status"] == "diagnostic_failed"
    assert report["steps"] == [{"name": "fresh_public_rtl_issuance", "status": "failed"}]
    assert report["error"]["stderr"] == "original parser failure\n"
    assert report["phases"]["1"]["status"] == report["phases"]["2"]["status"] == "blocked"
    assert owner.stat().st_mode & 0o777 == 0o700
    assert (owner / "report.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="fresh ordinary run owner"):
        D.run(path, output=owner)


@pytest.mark.parametrize("change", ["other_failure", "complete", "missing_manifest", "no_unknown"])
def test_arbitrary_generator_error_cannot_be_a_completed_diagnostic(tmp_path, change):
    root = tmp_path / "generated"
    evidence = root / "_evidence" / "coverage"
    evidence.mkdir(parents=True)
    (root / "MANIFEST.yaml").write_text("generated: []\n")
    receipt = {
        "schema": "merlin.phase0_generation.v1",
        "mode": "diagnostic",
        "qualification": "not_established",
        "failures": [
            {
                "capsule": "component coverage",
                "reason": "mandatory independent obligations unavailable; inspect private coverage report",
            }
        ],
        "corpus_manifest": str(root / "MANIFEST.yaml"),
    }
    coverage = {"status": "source_prepared_incomplete", "obligations": [{"mandatory": True, "state": "unavailable"}]}
    if change == "other_failure":
        receipt["failures"].append({"capsule": "source", "reason": "capture failed"})
    elif change == "complete":
        coverage["status"] = "complete"
    elif change == "no_unknown":
        coverage["obligations"][0]["state"] = "source_generated"
    else:
        (root / "MANIFEST.yaml").unlink()
    (evidence / "generation.json").write_text(json.dumps(receipt))
    (evidence / "component-coverage.json").write_text(json.dumps(coverage))

    def fail():
        raise RuntimeError("unrelated native failure")

    with pytest.raises(RuntimeError, match="unrelated native failure"):
        D._diagnostic_generation(fail, root)

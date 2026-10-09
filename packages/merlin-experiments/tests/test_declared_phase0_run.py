"""Closed declarations refuse substitution before fresh live issuance."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("change", [None, "source_budget", "execution_budget", "saved_status"])
def test_pointwise_policy_request_keeps_both_limits_and_cannot_import_admission(declared, change):
    request, _ = declared
    request = copy.deepcopy(request)
    request["automatic"]["schema"] = D.A.POINTWISE_POLICY_SCHEMA
    if change == "source_budget":
        del request["automatic"]["original_source_budget"]
    elif change == "execution_budget":
        del request["automatic"]["execution_budget"]
    elif change == "saved_status":
        request["automatic"]["original_operator_admission"] = {"status": "passed"}
    if change is None:
        assert D.validate(request) is request
    else:
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


def _bridge_request(declared, *, zero=False):
    request, path = declared
    request = copy.deepcopy(request)
    request["schema"] = D.BRIDGE_SCHEMA
    operator = request["operator_schemas"]
    operator["schema"] = D.S.ZERO_SELECTION_SCHEMA if zero else D.S.TENSOR_SELECTION_SCHEMA
    compiler = path.parent / "selected-compiler"
    compiler.write_bytes(b"explicit diagnostic compiler selection\n")
    operator["tensor_arguments"] = {
        "compiler": {"path": str(compiler), "sha256": hashlib.sha256(compiler.read_bytes()).hexdigest()}
    }
    if zero:
        operator["zero_returns"] = copy.deepcopy(operator["tensor_arguments"])
    path.write_text(json.dumps(request))
    return request, path, compiler


@pytest.mark.parametrize("zero", [False, True])
def test_bridge_request_preserves_closed_version_and_same_compiler(declared, zero):
    request, _, _ = _bridge_request(declared, zero=zero)
    assert D.validate(request) is request
    request["schema"] = D.SCHEMA
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize(
    "change",
    ["bare_path", "extra_authority", "other_namespace", "different_zero_compiler", "missing_zero", "extra_facet"],
)
def test_bridge_declaration_cannot_import_authority_or_change_sdk_selection(declared, change):
    request, _, compiler = _bridge_request(declared, zero=True)
    operator = request["operator_schemas"]
    if change == "bare_path":
        operator["tensor_arguments"]["compiler"] = str(compiler)
    elif change == "extra_authority":
        operator["tensor_arguments"]["getter_receipt"] = "saved-authority.json"
    elif change == "other_namespace":
        operator["namespace"] = "other"
    elif change == "different_zero_compiler":
        operator["zero_returns"]["compiler"]["sha256"] = "0" * 64
    elif change == "missing_zero":
        del operator["zero_returns"]
    else:
        operator["native_effects"] = {"status": "accepted"}
    with pytest.raises(ValueError):
        D.validate(request)


@pytest.mark.parametrize("change", ["changed", "forbidden", "alias", "parent_component"])
def test_bridge_compiler_membership_refuses_before_live_issuance(declared, tmp_path, monkeypatch, change):
    request, path, compiler = _bridge_request(declared)
    if change == "changed":
        compiler.write_bytes(b"changed compiler bytes\n")
    elif change == "forbidden":
        request["forbidden_roots"].append(str(compiler))
    elif change == "alias":
        alias = tmp_path / "compiler-alias"
        alias.symlink_to(compiler)
        request["operator_schemas"]["tensor_arguments"]["compiler"]["path"] = str(alias)
    else:
        request["operator_schemas"]["tensor_arguments"]["compiler"]["path"] = str(
            tmp_path / "missing" / ".." / compiler.name
        )
    path.write_text(json.dumps(request))
    monkeypatch.setattr(
        D, "issue_independent_hardware_intake", lambda **kwargs: pytest.fail("bad compiler reached issuer")
    )
    with pytest.raises(ValueError):
        D.run(path, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def _source_stage_diagnostics(monkeypatch, request):
    # Diagnostic stop fixtures exercise declaration forwarding only. They do
    # not issue a software/hardware/runtime capability or complete Phase 0.
    hardware = object()
    software = SimpleNamespace(
        sha256="1" * 64,
        receipt_json=json.dumps({"semantic_basis_sha256": request["inputs"]["semantic_basis"]["sha256"]}),
    )
    monkeypatch.setattr(D, "issue_independent_hardware_intake", lambda **kwargs: hardware)
    monkeypatch.setattr(D, "issue_independent_software_intake", lambda **kwargs: software)
    return software


@pytest.mark.parametrize("zero", [False, True])
def test_normal_bridge_route_forwards_selection_to_existing_native_issuer(declared, tmp_path, monkeypatch, zero):
    request, path, compiler = _bridge_request(declared, zero=zero)
    software = _source_stage_diagnostics(monkeypatch, request)
    seen = []

    def native_stop(**kwargs):
        assert kwargs["software"] is software
        selected = json.loads(kwargs["selection"].read_bytes())
        assert D.S._selection(kwargs["selection"].read_bytes()) == selected
        assert selected["tensor_arguments"] == {"compiler": str(compiler)}
        assert selected["schema"] == request["operator_schemas"]["schema"]
        if zero:
            assert selected["zero_returns"] == selected["tensor_arguments"]
        else:
            assert "zero_returns" not in selected
        seen.append(selected)
        raise RuntimeError("explicit diagnostic stop before native authority issuance")

    monkeypatch.setattr(D.S, "issue_independent_operator_schema_intake", native_stop)
    with pytest.raises(RuntimeError, match="diagnostic stop"):
        D.run(path, output=tmp_path / "run")
    assert len(seen) == 1
    report = json.loads((tmp_path / "run/report.json").read_bytes())
    assert report["status"] == "diagnostic_failed"
    assert report["steps"][-1] == {"name": "fresh_original_public_native_schemas", "status": "failed"}
    assert report["phases"]["1"]["status"] == report["phases"]["2"]["status"] == "blocked"


def test_changed_compiler_after_native_issuer_cannot_advance_to_rtl_observers(declared, tmp_path, monkeypatch):
    request, path, compiler = _bridge_request(declared, zero=True)
    _source_stage_diagnostics(monkeypatch, request)

    def changed(**kwargs):
        compiler.write_bytes(b"replacement during selected native preparation\n")
        return object()

    monkeypatch.setattr(D.S, "issue_independent_operator_schema_intake", changed)
    monkeypatch.setattr(
        D, "issue_independent_arithmetic_intake", lambda **kwargs: pytest.fail("changed compiler advanced")
    )
    with pytest.raises(ValueError, match="bytes changed"):
        D.run(path, output=tmp_path / "run")
    assert json.loads((tmp_path / "run/report.json").read_bytes())["status"] == "diagnostic_failed"


@pytest.mark.parametrize("purpose", ["source_diagnostic", "source_preparation", "performance_campaign"])
def test_requirement_request_is_explicit_and_does_not_change_legacy_fields(declared, purpose):
    request, _ = declared
    request = copy.deepcopy(request)
    request.update(schema=D.REQUIREMENT_SCHEMA, release_purpose=purpose)
    assert D.validate(request) == request
    for old in (D.SCHEMA, D.BRIDGE_SCHEMA):
        request["schema"] = old
        with pytest.raises(ValueError, match="closed explicit"):
            D.validate(request)


@pytest.mark.parametrize("change", ["absent", "unknown", "saved_ready"])
def test_requirement_request_refuses_absent_purpose_or_saved_readiness(declared, change):
    request, _ = declared
    request = copy.deepcopy(request)
    request.update(schema=D.REQUIREMENT_SCHEMA, release_purpose="source_preparation")
    if change == "absent":
        del request["release_purpose"]
    elif change == "unknown":
        request["release_purpose"] = "ready"
    else:
        request["source_requirement_ledger"] = {"status": "ready"}
    with pytest.raises(ValueError):
        D.validate(request)

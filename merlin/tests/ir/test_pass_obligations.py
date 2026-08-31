"""Production pass obligations are explicit, measurable, and capsule-attributed.

The production catalog, frontend normalizations, synthetic research pipeline, and target-family
edges are deliberately separate. Only a real model capsule may certify the production catalog.
"""
from __future__ import annotations

import importlib
import importlib.util
import json

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.xdsl_dialects.lowering import passes as PS

GATE = repo_root() / "build_tools" / "scripts" / "check_pass_obligations.py"
RATCHET = repo_root() / "build_tools" / "scripts" / "pass_obligations_ratchet.txt"
MODEL_CAPSULE = repo_root() / "merlin" / "contract" / "capsules" / "model" / \
    "M2_microvit_gemmini"


def _gate():
    spec = importlib.util.spec_from_file_location("_check_pass_obligations", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_ambient_log(monkeypatch):
    """An outer audit must not change what these tests measure."""
    monkeypatch.delenv(PS.PASS_LOG_ENV, raising=False)
    monkeypatch.delenv(PS.PASS_LOG_CAPSULE_ENV, raising=False)
    monkeypatch.delenv(PS.PASS_LOG_REQUIREMENTS_ENV, raising=False)


def test_production_catalog_is_fully_obligated_and_required():
    assert PS.catalog()
    for item in PS.catalog():
        assert item.discharges(), item.name
        assert item.is_required(), item.name
        assert item.input_dialect != PS.UNKNOWN, item.name
        assert item.output_dialect != PS.UNKNOWN, item.name


def test_all_authored_entry_points_resolve():
    """Every registry is executable even though only production entries are capsule-gated."""
    for item in PS.all_catalogs():
        module_name, _, function_name = item.entry.rpartition(".")
        assert callable(getattr(importlib.import_module(module_name), function_name)), item.name


def test_normalizations_are_not_laundered_into_target_obligations():
    assert PS.normalization_catalog()
    assert all(not item.discharges() and not item.is_required()
               for item in PS.normalization_catalog())
    production_names = {item.name for item in PS.catalog()}
    assert production_names.isdisjoint(item.name for item in PS.normalization_catalog())


def test_production_catalog_has_no_ratchet_debt():
    gate = _gate()
    findings = gate.findings(gate.audit([]), gate._load_ratchet(RATCHET))
    assert findings["undischarged"] == []
    assert findings["unrequired"] == []
    assert findings["unknown_dialect"] == []


def _row(item: PS.PassInfo, *, exercise: str = "unmeasured") -> dict:
    return {
        "name": item.name,
        "discharges": item.discharges(),
        "is_required": item.is_required(),
        "input_dialect": item.input_dialect,
        "output_dialect": item.output_dialect,
        "exercise": exercise,
        "capsules": [],
        "required_by": list(item.required_by),
        "required_hits": [],
        "install": [],
    }


def test_pass_with_no_obligation_and_no_requirement_is_rejected():
    gate = _gate()
    orphan = PS.PassInfo("merlin-invented", "normalize", "score went up", "merlin.nowhere.fn",
                         input_dialect="linalg", output_dialect="linalg")
    findings = gate.findings({"passes": [_row(orphan)], "measured": False}, set())
    assert [item["pass"] for item in findings["undischarged"]] == [orphan.name]
    assert [item["pass"] for item in findings["unrequired"]] == [orphan.name]


def _write_log(path, *, installed, invoked):
    lines = [json.dumps({"kind": "install",
                         "passes": {name: "instrumented" for name in installed}})]
    lines += [json.dumps({"kind": "invoke", "pass": name, "capsule": capsule,
                          "requirements": requirements})
              for name, capsule, requirements in invoked]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_declared_but_dead_pass_is_reported(tmp_path):
    log = tmp_path / "pass.jsonl"
    catalog = list(PS.catalog())
    names = [item.name for item in catalog]
    requirement = catalog[0].required_by[0]
    _write_log(log, installed=names, invoked=[(names[0], "CAP_A", [requirement])])
    report = PS.exercise_report(logs=[log])
    assert report["per_pass"][names[0]]["status"] == "exercised"
    assert [name for name in names[1:] if report["per_pass"][name]["status"] == "dead"] == names[1:]

    gate = _gate()
    findings = gate.findings(gate.audit([log]), set())
    assert sorted(item["pass"] for item in findings["dead"]) == sorted(names[1:])


def test_wrong_requirement_class_does_not_certify_a_pass(tmp_path):
    log = tmp_path / "pass.jsonl"
    item = PS.catalog()[0]
    _write_log(log, installed=[item.name],
               invoked=[(item.name, "UNRELATED_CAPSULE", ["different-capstone"])])
    report = PS.exercise_report([item], logs=[log])
    assert report["per_pass"][item.name]["status"] == "exercised_wrong_capsule"
    assert report["per_pass"][item.name]["required_hits"] == []


def test_dead_or_wrong_capsule_fails_only_when_requested(tmp_path):
    log = tmp_path / "pass.jsonl"
    names = [item.name for item in PS.catalog()]
    _write_log(log, installed=names, invoked=[])
    gate = _gate()
    assert gate.main(["--log", str(log)]) == 0
    assert gate.main(["--log", str(log), "--fail-on-dead"]) == 1


def test_no_log_is_unmeasured_and_fail_on_dead_cannot_pass(capsys):
    report = PS.exercise_report(logs=[])
    assert {value["status"] for value in report["per_pass"].values()} == {"unmeasured"}
    assert _gate().main(["--fail-on-dead"]) == 2
    assert "CANNOT DECIDE" in capsys.readouterr().err


def test_uninstrumented_is_not_dead(tmp_path):
    log = tmp_path / "pass.jsonl"
    name = PS.catalog()[0].name
    log.write_text(json.dumps({"kind": "install",
                               "passes": {name: "failed: ImportError: x"}}) + "\n",
                   encoding="utf-8")
    report = PS.exercise_report(logs=[log])
    assert report["per_pass"][name]["status"] == "not_instrumented"
    gate = _gate()
    findings = gate.findings(gate.audit([log]), set())
    assert findings["dead"] == []
    assert name in [item["pass"] for item in findings["not_instrumented"]]


def test_ratchet_entries_do_not_leak_across_pass_or_axis():
    gate = _gate()
    first = PS.PassInfo("first", "stage", "", "x.y", input_dialect="linalg",
                        output_dialect="linalg")
    second = PS.PassInfo("second", "stage", "", "x.y", input_dialect="linalg",
                         output_dialect="linalg")
    ratchet = {gate._debt(first.name, "undeclared", "obligation")}
    findings = gate.findings({"passes": [_row(first), _row(second)], "measured": False}, ratchet)
    assert findings["undischarged"][0]["ratcheted"] is True
    assert findings["unrequired"][0]["ratcheted"] is False
    assert findings["undischarged"][1]["ratcheted"] is False


def test_recorder_carries_concrete_capsule_and_requirement(tmp_path, monkeypatch):
    log = tmp_path / "pass.jsonl"
    item = PS.catalog()[0]
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log))
    PS.install_pass_recorder()
    with PS.pass_run_context("CAP_UNDER_TEST", item.required_by):
        PS.record_invocation(item.name)
    report = PS.exercise_report(logs=[log])
    row = report["per_pass"][item.name]
    assert row["status"] == "exercised"
    assert row["capsules"] == ["CAP_UNDER_TEST"]
    assert row["required_hits"] == list(item.required_by)


def test_invocation_without_capsule_context_is_not_coverage(tmp_path, monkeypatch):
    log = tmp_path / "pass.jsonl"
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log))
    PS.install_pass_recorder()
    PS.record_invocation(PS.catalog()[0].name)
    report = PS.exercise_report(logs=[log])
    assert report["per_pass"][PS.catalog()[0].name]["status"] == "exercised_unattributed"


@pytest.mark.skipif(not PS.HAS_XDSL, reason="xDSL not installed")
def test_prototype_pipeline_is_exercised_only_in_its_own_registry(tmp_path, monkeypatch):
    log = tmp_path / "prototype.jsonl"
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log))
    PS.install_pass_recorder(PS.prototype_catalog())
    from merlin.xdsl_dialects.lowering.pipeline import lower_repeated_rhs_matmul
    with PS.pass_run_context("SYNTH_REPEATED_RHS"):
        lower_repeated_rhs_matmul()
    report = PS.exercise_report(PS.prototype_catalog(), logs=[log])
    assert {name for name, value in report["per_pass"].items()
            if value["status"] == "exercised"} == {
                item.name for item in PS.prototype_catalog()}


@pytest.mark.skipif(not PS.HAS_XDSL, reason="xDSL not installed")
def test_real_model_capstone_exercises_every_production_pass(tmp_path, monkeypatch):
    """The positive proof is the real model boundary path, not a synthetic lookalike."""
    capsule = yaml.safe_load((MODEL_CAPSULE / "capsule.yaml").read_text(encoding="utf-8"))
    assert PS.MODEL_BOUNDARY_CAPSTONE in capsule["pass_requirements"]
    log = tmp_path / "model.jsonl"
    monkeypatch.setenv(PS.PASS_LOG_ENV, str(log))
    PS.install_pass_recorder()
    from merlin.frontends.linalg_mlir import parse_mlir_file
    with PS.pass_run_context(capsule["name"], capsule["pass_requirements"]):
        result = PS.run_dialect_plane(parse_mlir_file(MODEL_CAPSULE / capsule["linalg_mlir"]))
    assert result.stats["kernels"] > 0
    assert result.stats["c_interface_funcs"] > 0
    report = PS.exercise_report(logs=[log])
    assert {value["status"] for value in report["per_pass"].values()} == {"exercised"}
    assert all(value["capsules"] == [capsule["name"]]
               for value in report["per_pass"].values())
    assert all(value["required_hits"] == [PS.MODEL_BOUNDARY_CAPSTONE]
               for value in report["per_pass"].values())

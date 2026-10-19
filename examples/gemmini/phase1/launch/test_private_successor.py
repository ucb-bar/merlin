"""Operator successor custody and scope, without captures, compilers or targets."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import evidence
    from merlin_experiments.phase1 import launch_config
    from merlin_experiments.phase1.feedback import private_full_models

    from merlin.targetgen import target_experiment

    source = Path(__file__).with_name("prepare_private_successor.py")
    spec = importlib.util.spec_from_file_location("test_successor_operator", source)
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    plan = {
        "schema": "merlin.operator.private_successor_plan.v1",
        "task_root": str(tmp_path),
        "rtl_source_relocations": {},
    }
    for name in (
        "original_private_spec",
        "descriptor",
        "host_capabilities",
        "hardware_spec",
        "conformance_spec",
        "application_inventory",
        "rtl_audit",
        "public_corpus_seal",
    ):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        plan[name] = operator.pin(path)
    row = {
        "id": "first",
        "host_capabilities": "original-host",
        "host_capabilities_sha256": "old",
        "recipe_derivation_root": "original-evidence",
        "recipe_derivation_manifest_sha256": "old",
    }
    for field in ("software_spec", "capability_contract", "rtl_facts"):
        path = tmp_path / field
        path.write_text("{}")
        row[field], row[field + "_sha256"] = str(path), operator.pin(path)["sha256"]
    capture = tmp_path / "selection.json"
    capture.write_text('{"plan":{}}')
    row["capture_selection"] = str(capture)
    row["capture_selection_sha256"] = operator.pin(capture)["sha256"]
    original = {"target": "fixture", "models": [row, {**row, "id": "second"}]}
    Path(plan["original_private_spec"]["path"]).write_text(yaml.safe_dump(original))
    plan["original_private_spec"] = operator.pin(plan["original_private_spec"]["path"])
    selection = SimpleNamespace(
        status="verified",
        diagnostics=[],
        host_capabilities={"host": {"capability_spec_sha256": plan["host_capabilities"]["sha256"]}},
    )
    monkeypatch.setattr(launch_config, "load", lambda _: {})
    monkeypatch.setattr(
        launch_config,
        "build",
        lambda *a, **k: SimpleNamespace(environment={}, launch_environment={}, path=lambda value: tmp_path),
    )
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a, **k: object())
    seen = []

    def select(target, **kwargs):
        seen.append(kwargs)
        return selection

    def export(selected, root):
        root.mkdir()
        (root / "evidence-manifest.json").write_text("{}")

    monkeypatch.setattr(evidence, "select_evidence", select)
    monkeypatch.setattr(evidence, "export_evidence", export)
    monkeypatch.setattr(evidence, "load_exported_evidence", lambda _: selection)
    monkeypatch.setattr(private_full_models, "_recipe_derivation", lambda *a, **k: ({"scope": "recipe only"}, {}, {}))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    (tmp_path / "config").write_text("{}")
    return operator, plan_path, plan, original, selection, seen


def test_successor_preserves_original_models_and_public_seal(prepared, tmp_path):
    operator, path, plan, original, _, seen = prepared
    output = tmp_path / "successor"
    receipt = operator.prepare(plan_path=path, config=tmp_path / "config", prep=tmp_path, output=output)
    assert receipt["model_qualification"] is False and receipt["compiler_or_target_execution"] is False
    assert receipt["selected_inputs"]["public_corpus_seal"] == plan["public_corpus_seal"]
    after = yaml.safe_load((output / "private-models.yaml").read_bytes())
    for before, successor in zip(original["models"], after["models"], strict=True):
        assert {key for key in before if before[key] != successor[key]} == {
            "host_capabilities",
            "host_capabilities_sha256",
            "recipe_derivation_root",
            "recipe_derivation_manifest_sha256",
        }
    assert seen[0]["conformance_spec"] == Path(plan["conformance_spec"]["path"])
    assert seen[0]["facts_path"].read_bytes() == Path(original["models"][0]["rtl_facts"]).read_bytes()
    assert yaml.safe_load(Path(plan["original_private_spec"]["path"]).read_bytes()) == original


@pytest.mark.parametrize("failure", ["diagnostic", "different_host", "changed_input"])
def test_successor_refuses_incomplete_or_changed_selection(prepared, tmp_path, failure):
    operator, path, plan, _, selected, seen = prepared
    if failure == "diagnostic":
        selected.status, selected.diagnostics = "diagnostic", [{"component": "missing"}]
    elif failure == "different_host":
        selected.host_capabilities = copy.deepcopy(selected.host_capabilities)
        selected.host_capabilities["host"]["capability_spec_sha256"] = "0" * 64
    else:
        Path(plan["rtl_audit"]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        operator.prepare(plan_path=path, config=tmp_path / "config", prep=tmp_path, output=tmp_path / "successor")
    assert not (tmp_path / "successor" / "selection.json").exists()
    if failure == "changed_input":
        assert seen == []

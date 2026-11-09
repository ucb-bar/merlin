"""The portable baseline contract must be generated from the authored definitions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/gemmini/protocol/check.py"
SPEC = SCRIPT.with_name("experiment.yaml")


@pytest.fixture
def protocol():
    spec = importlib.util.spec_from_file_location("gemmini_protocol_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_reopens_the_shared_definitions(protocol):
    value = protocol.resolved_contract(SPEC)
    models_path = SPEC.parent / value["protocol"]["authority"]["numerical_contract"]
    assert value["numerical_policy"] == yaml.safe_load(models_path.read_bytes())
    calibration_name = value["protocol"]["authority"]["capsule_calibration_requirements"]
    assert value["capsule_calibration_requirements"] == yaml.safe_load((SPEC.parent / calibration_name).read_bytes())
    assert calibration_name in value["files"]
    assert value["execution_qualified"] is False
    selected = value["models_configuration"]
    payload = yaml.safe_dump(selected["configuration"], sort_keys=True, allow_unicode=True).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == selected["sha256"]
    for pin in selected["sources"].values():
        assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
    baseline = yaml.safe_load(models_path.read_bytes())
    assert baseline["models"]["tiny_llama"]["chain"]["prefill_tokens"] == 128
    assert selected["configuration"]["models"]["tiny_llama"]["chain"]["prefill_tokens"] == 32
    for name, model in selected["configuration"]["models"].items():
        assert model["chain"] == value["protocol"]["models"][name]["chain"]
        model["chain"] = baseline["models"][name]["chain"]
    assert selected["configuration"] == baseline  # No other authored policy was copied or changed.
    for name, pin in value["files"].items():
        data = (SPEC.parent / name).read_bytes()
        assert pin == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_export_tracks_a_changed_declared_policy(protocol, tmp_path):
    original = protocol.resolved_contract(SPEC)
    root = tmp_path / "protocol"
    for name in original["files"]:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((SPEC.parent / name).read_bytes())
    models_path = root / original["protocol"]["authority"]["numerical_contract"]
    launch_path = root / original["protocol"]["authority"]["phase1_launch"]
    models = yaml.safe_load(models_path.read_bytes())
    launch = yaml.safe_load(launch_path.read_bytes())
    models["numerics"]["functional"]["atol"] = 0.0001
    models_path.write_text(yaml.safe_dump(models))
    with pytest.raises(ValueError, match="selected launch atol"):
        protocol.resolved_contract(root / SPEC.name)
    launch["full_model_verification"]["atol"] = 0.0001
    launch_path.write_text(yaml.safe_dump(launch))
    changed = protocol.resolved_contract(root / SPEC.name)
    assert changed["numerical_policy"]["numerics"]["functional"]["atol"] == 0.0001
    assert changed["files"]["../numerics/models.yaml"] != original["files"]["../numerics/models.yaml"]


def test_export_cli_writes_only_the_requested_snapshot(protocol, tmp_path, monkeypatch):
    output = tmp_path / "shared-contract.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--export", str(output)])
    assert protocol.main() == 0
    exported = json.loads(output.read_text())
    assert exported["schema"] == "merlin.gemmini.resolved_protocol.v1"
    selected = exported["models_configuration"]
    path = Path(selected["path"])
    assert path == output.with_name(output.stem + ".models.yaml")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == selected["sha256"]
    assert yaml.safe_load(path.read_bytes()) == selected["configuration"]
    original = SPEC.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--export", str(SPEC)])
    assert protocol.main() == 1
    assert SPEC.read_bytes() == original


@pytest.mark.parametrize("changed", ["selection_key", "unknown_model", "missing_model", "extra_chain_key",
                                     "changed_kind", "zero", "boolean", "program_order", "program_steps"])
def test_geometry_selection_is_closed_and_cannot_rewrite_policy_or_programs(protocol, changed):
    from merlin.model_check.models import ModelsFileError, _protocol_models_bytes

    spec = yaml.safe_load(SPEC.read_bytes())
    baseline = (SPEC.parent / spec["authority"]["numerical_contract"]).read_bytes()
    if changed == "selection_key":
        spec["authority"]["models_configuration"]["unreviewed"] = True
    elif changed == "unknown_model":
        spec["models"]["unselected"] = spec["models"]["tiny_llama"]
    elif changed == "missing_model":
        del spec["models"]["tiny_llama"]
    elif changed in {"extra_chain_key", "changed_kind", "zero", "boolean"}:
        chain = spec["models"]["tiny_llama"]["chain"]
        if changed == "extra_chain_key":
            chain["host_math_policy"] = "portable_exp_trig"
        elif changed == "changed_kind":
            chain["kind"] = "single"
        else:
            chain["prefill_tokens"] = 0 if changed == "zero" else True
    elif changed == "program_order":
        spec["models"]["tiny_llama"]["stages"] = dict(reversed(list(spec["models"]["tiny_llama"]["stages"].items())))
    else:
        spec["models"]["tiny_llama"]["stages"]["decode"]["steps"] += 1
    with pytest.raises(ModelsFileError):
        _protocol_models_bytes(baseline, spec)


def test_geometry_is_not_implicitly_enabled_for_legacy_protocol(protocol):
    spec = yaml.safe_load(SPEC.read_bytes())
    del spec["authority"]["models_configuration"]
    baseline = (SPEC.parent / spec["authority"]["numerical_contract"]).read_bytes()
    from merlin.model_check.models import ModelsFileError, _protocol_models_bytes

    with pytest.raises(ModelsFileError, match="explicit"):
        _protocol_models_bytes(baseline, spec)

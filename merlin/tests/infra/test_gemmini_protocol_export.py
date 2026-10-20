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
    assert value["execution_qualified"] is False
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
    assert json.loads(output.read_text())["schema"] == "merlin.gemmini.resolved_protocol.v1"
    original = SPEC.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--export", str(SPEC)])
    assert protocol.main() == 1
    assert SPEC.read_bytes() == original

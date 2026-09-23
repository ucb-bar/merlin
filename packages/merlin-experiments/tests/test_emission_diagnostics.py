"""Explicit analysis resources; no checkout policy or native engine discovery."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import emission_diagnostics as DIAG
from merlin_experiments.phase2.contracts import StageGateError


def _schema(tmp_path):
    path = tmp_path / "external-schema.json"
    payload = json.dumps({"type": "object", "required": ["commands"]}).encode()
    path.write_bytes(payload)
    return path, {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def test_schema_record_uses_explicit_resource(tmp_path):
    path, record = _schema(tmp_path)
    assert DIAG.whole_program_schema_record(path) == record
    DIAG.validate_whole_program_schema({"commands": []}, record, arm="candidate")
    with pytest.raises(StageGateError, match="schema"):
        DIAG.validate_whole_program_schema({}, record, arm="candidate")


@pytest.mark.parametrize("change", ["bytes", "missing", "symlink"])
def test_schema_drift_and_linked_resources_refuse(tmp_path, change):
    path, record = _schema(tmp_path)
    if change == "bytes":
        path.write_text("{}")
    elif change == "missing":
        path.unlink()
    else:
        original = tmp_path / "original.json"
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(StageGateError):
        DIAG.validate_whole_program_schema({"commands": []}, record, arm="candidate")


def test_schema_validation_parses_the_same_bytes_it_hashes(tmp_path, monkeypatch):
    path, record = _schema(tmp_path)
    original_read = Path.read_bytes
    reads = []

    def read_once(actual):
        data = original_read(actual)
        if actual == path:
            reads.append(actual)
            assert len(reads) == 1
            actual.write_text('{"not": {}}')
        return data

    def no_text_reread(actual, *args, **kwargs):
        raise AssertionError("validation must parse the hashed bytes")

    monkeypatch.setattr(Path, "read_bytes", read_once)
    monkeypatch.setattr(Path, "read_text", no_text_reread)
    DIAG.validate_whole_program_schema({"commands": []}, record, arm="candidate")
    assert reads == [path]


def _no_ambient_resources(monkeypatch):
    from merlin.common import provenance
    from merlin.perf import gate_phase

    def forbidden(*args, **kwargs):
        raise AssertionError("ambient resource discovery is forbidden")

    monkeypatch.setattr(provenance, "load_artifacts", forbidden)
    monkeypatch.setattr(gate_phase, "configured_phase", forbidden)


def test_device_resolution_uses_only_supplied_registry(monkeypatch):
    _no_ambient_resources(monkeypatch)
    keys = {"hw_config": "queue", "hwdb_config_artifact_sha256": "a" * 64}
    artifacts = {
        "device": SimpleNamespace(
            role="firesim_bitstream", hw_configs=("queue",), hwdb_digest="a" * 64, config="synthetic-config"
        )
    }
    result = DIAG.resolved_device(keys, artifacts=artifacts)
    assert result["name"] == "device"
    assert result["confirmed_by"] == "hw_config and hwdb digest"
    assert DIAG.resolved_device(keys, artifacts={})["name"] is None


@pytest.mark.parametrize("invalid", [None, []])
def test_invalid_registry_never_triggers_ambient_discovery(monkeypatch, invalid):
    from merlin.common import provenance
    from merlin.liveness import facts

    calls = []
    monkeypatch.setattr(provenance, "load_artifacts", lambda: calls.append("ambient") or {})
    monkeypatch.setattr(facts, "silicon_facts", lambda target: SimpleNamespace(mesh_rows=2, mesh_cols=3))
    assert DIAG.resolved_device({"hw_config": "queue"}, artifacts=invalid)["name"] is None
    result = DIAG.iteration_cost_plane({}, target="synthetic", arms={}, phase="report", artifacts=invalid)
    assert result["status"] == "incomplete"
    assert calls == []


@pytest.mark.parametrize("phase", ["report", "fail"])
def test_cost_plane_keeps_declared_phase_and_missing_measurements_incomplete(monkeypatch, phase):
    from merlin.liveness import facts

    _no_ambient_resources(monkeypatch)
    monkeypatch.setattr(facts, "silicon_facts", lambda target: SimpleNamespace(mesh_rows=2, mesh_cols=3))
    arms = {arm: {"macs": 12, "exact": True} for arm in ("baseline", "candidate")}
    result = DIAG.iteration_cost_plane({}, target="synthetic", arms=arms, phase=phase, artifacts={})
    assert result["phase"] == phase
    assert result["status"] == "incomplete"
    assert result["admitted"] is False
    assert result["blocking"] is False
    assert result["device"]["name"] is None
    assert result["array"] == {"rows": 2, "cols": 3, "basis": "derived from this target's own RTL facts"}

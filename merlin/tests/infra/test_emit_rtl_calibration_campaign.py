"""Contract tests for the RTL-derived calibration-campaign planning edge."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from merlin.common.paths import repo_root


_SCRIPT = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/"
           "emit_rtl_calibration_campaign.py")
_SPEC = importlib.util.spec_from_file_location("_emit_rtl_calibration_campaign", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _rtl() -> dict:
    return {
        "generator": {"tool": "synthetic CIRCT extractor"},
        "facts": {"source": "synthetic elaboration", "arrays": [
            {"rows": 3, "cols": 5, "primary": True, "provenance": "array declaration"},
        ]},
    }


def _capabilities() -> dict:
    return {
        "dma": {
            "directions": {"value": ["read", "write", "copy"],
                           "derived_from_tool": True, "source": "compiler probe"},
            "measurement_protocols": {"value": ["fresh-process", "predecessor-run"],
                             "derived_from_tool": True, "source": "runner probe"},
            **{direction: {"sizes_bytes": {
                "value": [8, 16, 32, 64], "derived_from_tool": True,
                "source": f"{direction} compiler probe"}}
               for direction in ("read", "write", "copy")},
        },
        "compute": {
            "workload_emitter": {"value": True, "derived_from_tool": True,
                           "source": "compiler probe"},
            "tile_multiples": {"value": [1, 2, 4, 8], "derived_from_tool": True,
                               "source": "compiler probe"},
        },
        "measurement_auxiliary": {
            "empty_workload_emitter": {
                "value": True, "derived_from_tool": True,
                "source": "frozen compiler empty-workload probe"},
            "joint_occupancy_probe": {
                "value": True, "derived_from_tool": True,
                "source": "RTL joint-occupancy probe"},
        },
    }


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_ready_manifest_has_every_derived_coordinate_and_content_addressed_identity(tmp_path: Path) -> None:
    rtl_path = _write(tmp_path / "rtl.json", _rtl())
    capabilities_path = _write(tmp_path / "capabilities.json", _capabilities())

    artifact, status = _MODULE.build(rtl_path, capabilities_path)

    assert status == 0 and artifact["dispatchable"] is True
    requests = artifact["measurement_requests"]
    assert len(requests) == 28
    assert {row["identity"]["sweep_id"] for row in requests} == {
        "dma.read.fresh-process", "dma.read.predecessor-run",
        "dma.write.fresh-process", "dma.write.predecessor-run",
        "dma.copy.fresh-process", "dma.copy.predecessor-run", "compute.saturation",
    }
    assert all(row["request_sha256"] == hashlib.sha256(json.dumps(
        row["identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               for row in requests)
    assert all(row["facts"] for row in requests)
    assert {tuple(row["required_raw_receipts"]) for row in requests} == {
        ("rtl_cycle_measurement", "compiler_command_buffer"),
        ("rtl_cycle_measurement", "physical_counter"),
    }
    auxiliary = artifact["auxiliary_measurement_requests"]
    assert len(auxiliary) == 9
    assert sum(row["identity"]["kind"] == "empty_run" for row in auxiliary) == 8
    assert sum(row["identity"]["kind"] == "composition_probe" for row in auxiliary) == 1
    assert all(row["request_sha256"] == hashlib.sha256(json.dumps(
        row["identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               for row in auxiliary)
    compute_protocols = {row["identity"]["measurement_protocol"] for row in requests
                         if row["mechanism"] == "compute"}
    composition_protocols = {row["identity"]["measurement_protocol"] for row in auxiliary
                             if row["identity"]["kind"] == "composition_probe"}
    assert compute_protocols == composition_protocols == {"fresh-process"}


def test_missing_tool_probe_evidence_is_refused_and_cannot_dispatch_a_ready_subset(tmp_path: Path) -> None:
    capabilities = _capabilities()
    del capabilities["dma"]["copy"]["sizes_bytes"]

    artifact, status = _MODULE.build(
        _write(tmp_path / "rtl.json", _rtl()), _write(tmp_path / "capabilities.json", capabilities))

    assert status == 1
    assert artifact["status"] == "refused"
    assert artifact["dispatchable"] is False
    assert len(artifact["measurement_requests"]) == 20
    assert {row["sweep_id"] for row in artifact["refusals"]} == {
        "dma.copy.fresh-process", "dma.copy.predecessor-run"}
    assert artifact["execution_contract"]["partial_execution_is_admissible"] is False


def test_input_byte_changes_change_all_request_identities(tmp_path: Path) -> None:
    rtl_path = _write(tmp_path / "rtl.json", _rtl())
    capabilities_path = _write(tmp_path / "capabilities.json", _capabilities())
    first, _ = _MODULE.build(rtl_path, capabilities_path)
    changed = _capabilities()
    changed["compute"]["tile_multiples"]["value"] = [1, 2, 4, 16]
    second, _ = _MODULE.build(rtl_path, _write(tmp_path / "capabilities-2.json", changed))

    assert first["inputs"]["harness_capabilities"]["sha256"] != second["inputs"][
        "harness_capabilities"]["sha256"]
    assert {row["request_sha256"] for row in first["measurement_requests"]} != {
        row["request_sha256"] for row in second["measurement_requests"]}


def test_unreadable_input_is_a_recorded_refusal_not_an_exception(tmp_path: Path) -> None:
    artifact, status = _MODULE.build(tmp_path / "missing.json", _write(
        tmp_path / "capabilities.json", _capabilities()))

    assert status == 1
    assert artifact["measurement_requests"] == []
    assert artifact["auxiliary_measurement_requests"] == []
    assert artifact["inputs"]["rtl_facts"]["sha256"] == "UNKNOWN"
    assert "cannot read explicit path" in artifact["refusals"][0]


def test_missing_empty_or_composition_tool_path_makes_campaign_nondispatchable(
        tmp_path: Path) -> None:
    capabilities = _capabilities()
    capabilities["measurement_auxiliary"]["empty_workload_emitter"] = {
        "value": False, "derived_from_tool": True,
        "source": "probe found no frozen-compiler empty path"}

    artifact, status = _MODULE.build(
        _write(tmp_path / "rtl.json", _rtl()),
        _write(tmp_path / "capabilities.json", capabilities))

    assert status == 1 and artifact["dispatchable"] is False
    assert [row["identity"]["kind"]
            for row in artifact["auxiliary_measurement_requests"]] == ["composition_probe"]
    assert any(row["sweep_id"] == "measurement_auxiliary.empty_workload_emitter"
               for row in artifact["refusals"])

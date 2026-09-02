"""Adversarial tests for the runner-to-roofline receipt bridge."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.perf import receipt_bridge


_SCRIPT = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/"
           "collect_rtl_roofline_receipts.py")
_SPEC = importlib.util.spec_from_file_location("_collect_rtl_roofline_receipts", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _command(macs: int = 4) -> dict[str, Any]:
    return {
        "tensors": {"a": {"shape": [macs, 1]}, "b": {"shape": [1, 1]}},
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}],
    }


def _identity(command: dict[str, Any], rtl_sha256: str) -> dict[str, Any]:
    return {
        "program": {"kind": "compiler_command_buffer", "sha256": _digest(command)},
        "inputs": {"kind": "frozen_input", "sha256": "1" * 64},
        "toolchain": {"target": "fixture", "frozen_submission_sha256": "2" * 64,
                      "recorded_revisions": {"compiler": "fixture-r1"},
                      "rtl_facts_sha256": rtl_sha256},
    }


def _pass(command: dict[str, Any], rtl_sha256: str, *, cycles: int, protocol: str,
          counters: dict[str, Any], nonce: str) -> dict[str, Any]:
    identity = _identity(command, rtl_sha256)
    return {
        "measurement_identity": identity, "measurement_identity_refusals": [],
        "rtl_facts_sha256": rtl_sha256, "run_nonce": nonce,
        "per_sim": {"tool-selected-rtl": {
            "cycles": cycles, "correct": True,
            "rtl_facts_sha256": rtl_sha256,
            "provenance": {"derived_from_rtl": True, "cycle_accurate": True,
                           "evidence": "fixture RTL transcript"},
            "measurement_conditions": {"cache_protocol": protocol, "window": "fixture"},
            "counters": counters,
        }},
    }


def _profiled_measurement(command: dict[str, Any], rtl_sha256: str, *, cycles: int,
                          protocol: str, nonce: str) -> dict[str, Any]:
    facts = [
        {
            "fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha256,
            "counter_field": "physical-read", "direction": "read", "unit_bytes": 1,
            "derived_from_rtl": True, "provenance": "fixture structural RTL proof",
        },
        {
            "fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha256,
            "counter_field": "physical-write", "direction": "write", "unit_bytes": 1,
            "derived_from_rtl": True, "provenance": "fixture structural RTL proof",
        },
    ]
    occupancy_report = {
        "occupancy": {"by_combination": {"engine": "only-engine"}},
        "readings": {"only-engine": cycles - 1},
    }
    byte_report = {
        "readings": {"physical-read": 7, "physical-write": 0},
        "selected_counters": {"physical-read": 17, "physical-write": 18},
    }
    occupancy = _pass(command, rtl_sha256, cycles=cycles, protocol=protocol,
                      counters=occupancy_report, nonce=f"{nonce}-occupancy")
    physical = _pass(command, rtl_sha256, cycles=cycles, protocol=protocol,
                     counters=byte_report, nonce=f"{nonce}-physical")
    measurement = dict(occupancy)
    measurement["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical}
    measurement["linked_counter_evidence"] = {
        "status": "linked", "refusals": [],
        "measurement_identity": measurement["measurement_identity"],
        "occupancy": occupancy_report,
        "physical_byte_counters": {
            "selected_counters": byte_report["selected_counters"],
            "readings": byte_report["readings"], "counter_facts": facts,
        },
    }
    return measurement


def _row(command: dict[str, Any], measurement: dict[str, Any], *, workload: str) -> dict[str, Any]:
    return {
        "kernel": workload, "measurement": measurement,
        "command_buffer_artifact": {
            "command_buffer": command, "artifact_sha256": _digest(command),
            "compiler_provenance": "fixture compiler output"},
        "resource_bindings": {
            "compute": {"resource": "fixture-compute", "derived_from_tool": True,
                        "provenance": "compiler resource-trait binding"},
            "movement": {"resource": "fixture-movement", "derived_from_tool": True,
                         "provenance": "RTL counter resource-trait binding"},
        },
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    circt_sha256 = "a" * 64
    rtl_path = _write(tmp_path / "rtl.json", {
        "inputs": {"core_hw_sha256": circt_sha256}, "facts": {"source": "fixture"}})
    rtl_sha256 = hashlib.sha256(rtl_path.read_bytes()).hexdigest()
    capability_sha256 = "b" * 64
    identity = {
        "rtl_facts_sha256": rtl_sha256,
        "harness_capabilities_sha256": capability_sha256,
        "sweep_id": "fixture-compute-sweep", "ordinal": 0,
        "coordinates": {"fixture-coordinate": 1},
    }
    protocol = "fixture-protocol"
    auxiliary_identities = [{
        "rtl_facts_sha256": rtl_sha256,
        "harness_capabilities_sha256": capability_sha256,
        "kind": "empty_run", "measurement_protocol": protocol, "replicate": index,
    } for index in range(4)] + [{
        "rtl_facts_sha256": rtl_sha256,
        "harness_capabilities_sha256": capability_sha256,
        "kind": "composition_probe", "measurement_protocol": protocol,
    }]
    campaign = {
        "schema": "rtl_calibration_campaign_v1", "dispatchable": True,
        "inputs": {"rtl_facts": {"sha256": rtl_sha256},
                   "harness_capabilities": {"sha256": capability_sha256}},
        "execution_contract": {"partial_execution_is_admissible": False},
        "measurement_requests": [{
            "request_sha256": _digest(identity), "identity": identity,
            "mechanism": "compute",
            "required_raw_receipts": ["rtl_cycle_measurement", "compiler_command_buffer"],
        }],
        "auxiliary_measurement_requests": [{
            "request_sha256": _digest(auxiliary), "identity": auxiliary,
            "required_raw_receipts": (["rtl_cycle_measurement", "compiler_command_buffer"]
                                      if auxiliary["kind"] == "empty_run" else
                                      ["rtl_cycle_measurement", "joint_occupancy_partition"]),
        } for auxiliary in auxiliary_identities],
    }
    campaign_path = _write(tmp_path / "campaign.json", campaign)
    campaign_sha256 = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
    command = _command()
    measurement = _profiled_measurement(
        command, rtl_sha256, cycles=12, protocol=protocol, nonce="calibration")
    calibration_row = _row(command, measurement, workload="calibration")
    calibration_row.update({"request_sha256": _digest(identity), "identity": identity})
    empty_command = {"tensors": {}, "commands": []}
    empty_runs = []
    for index in range(4):
        empty = _pass(empty_command, rtl_sha256, cycles=1, protocol=protocol,
                      counters={}, nonce=f"empty-{index}")
        empty_row = _row(empty_command, empty, workload=f"empty-{index}")
        empty_row.update({"request_sha256": _digest(auxiliary_identities[index]),
                          "identity": auxiliary_identities[index]})
        empty_runs.append(empty_row)
    composition = _row(command, measurement, workload="composition")
    composition.update({"request_sha256": _digest(auxiliary_identities[-1]),
                        "identity": auxiliary_identities[-1]})
    composition["occupancy_binding"] = {
        "derived_from_rtl": True, "rtl_facts_sha256": rtl_sha256,
        "circt_hw_sha256": circt_sha256, "provenance": "fixture CIRCT proof",
        "counter_layout": {
            "prefix": "fixture", "engines": ["engine"],
            "kinds": {"engine": "compute"}, "by_combination": {"engine": "only-engine"}},
        "codes": {"only-engine": 1}, "module": "Fixture", "counter_module": "Meter",
    }
    calibration_path = _write(tmp_path / "calibration-results.json", {
        "campaign_manifest_sha256": campaign_sha256, "results": [calibration_row],
        "empty_runs": empty_runs, "composition_probe": composition,
    })
    workload_path = _write(tmp_path / "perf-results.json", [
        _row(command, _profiled_measurement(
            command, rtl_sha256, cycles=20, protocol=protocol, nonce="workload"),
             workload="case-0")])
    return rtl_path, campaign_path, calibration_path, workload_path


def test_bridge_emits_builder_receipts_only_from_exact_linked_raw_evidence(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    report, calibration, observations, status = _MODULE.build(*paths)

    assert status == 0 and report["status"] == "READY"
    assert report["refusals"] == [] and report["outputs_are_complete"] is True
    assert len(calibration["samples"]) == 1
    sample = calibration["samples"][0]
    assert sample["kind"] == "compute"
    assert sample["command_buffer_receipt"]["artifact"]["command_buffer"] == _command()
    assert len(calibration["empty_run_receipts"]) == 4
    assert observations["expected_workloads"] == ["case-0"]
    observation = observations["observations"][0]
    assert observation["cycle_receipt"]["artifact"]["cycles"] == 20
    assert observation["traffic_receipt"]["artifact"]["readings"] == {
        "physical-read": 7, "physical-write": 0}
    assert "moved_bytes" not in observation and "work" not in observation


def test_current_style_derived_work_total_cannot_replace_raw_command_buffer(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    del document[0]["command_buffer_artifact"]
    document[0]["work_volume"] = {"exact_macs": 4, "artifact_sha256": _digest(_command())}
    _write(workloads, document)

    report, calibration_out, observations, status = _MODULE.build(
        rtl, campaign, calibration, workloads)

    assert status == 1 and report["status"] == "NO_GO"
    assert "MISSING_RAW_COMMAND_BUFFER" in {issue["code"] for issue in report["refusals"]}
    assert calibration_out["samples"] == [] and observations["observations"] == []


def test_declared_command_buffer_digest_must_match_the_raw_artifact(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    document[0]["command_buffer_artifact"]["artifact_sha256"] = "0" * 64
    _write(workloads, document)

    report, calibration_out, observations, status = _MODULE.build(
        rtl, campaign, calibration, workloads)

    assert status == 1
    assert "COMMAND_BUFFER_RECEIPT_MISMATCH" in {
        issue["code"] for issue in report["refusals"]}
    assert calibration_out["samples"] == [] and observations["observations"] == []


def test_logical_traffic_or_unbound_unit_counters_cannot_become_physical_bytes(
        tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    physical = document[0]["measurement"]["linked_counter_evidence"]["physical_byte_counters"]
    del physical["counter_facts"]
    document[0]["logical_bytes"] = 7
    _write(workloads, document)

    report, _, observations, status = _MODULE.build(rtl, campaign, calibration, workloads)

    assert status == 1
    assert "UNPROVEN_PHYSICAL_BYTES" in {issue["code"] for issue in report["refusals"]}
    assert observations == {"expected_workloads": [], "observations": []}


def test_rtl_cycle_evidence_must_name_the_exact_rtl_facts_hash(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    document[0]["measurement"]["rtl_facts_sha256"] = "0" * 64
    _write(workloads, document)

    report, _, _, status = _MODULE.build(rtl, campaign, calibration, workloads)

    assert status == 1
    assert "UNLINKED_RTL_FACTS" in {issue["code"] for issue in report["refusals"]}


def test_each_counter_pass_independently_names_the_exact_rtl_facts_hash(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    physical = document[0]["measurement"]["counter_passes"]["physical_bytes"]
    physical["per_sim"]["tool-selected-rtl"]["rtl_facts_sha256"] = "0" * 64
    _write(workloads, document)

    report, _, _, status = _MODULE.build(rtl, campaign, calibration, workloads)

    assert status == 1
    assert "UNLINKED_RTL_FACTS" in {issue["code"] for issue in report["refusals"]}


def test_counter_pass_conditions_must_be_identical(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    physical = document[0]["measurement"]["counter_passes"]["physical_bytes"]
    physical["per_sim"]["tool-selected-rtl"]["measurement_conditions"]["window"] = "other"
    _write(workloads, document)

    report, _, _, status = _MODULE.build(rtl, campaign, calibration, workloads)

    assert status == 1
    assert "COUNTER_PASS_MISMATCH" in {issue["code"] for issue in report["refusals"]}


def test_byte_traffic_uses_its_own_cycle_window_when_instrumentation_perturbs_cycles(
        tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(workloads.read_text(encoding="utf-8"))
    physical = document[0]["measurement"]["counter_passes"]["physical_bytes"]
    physical["per_sim"]["tool-selected-rtl"]["cycles"] = 23
    _write(workloads, document)

    report, _, observations, status = _MODULE.build(rtl, campaign, calibration, workloads)

    assert status == 0 and report["status"] == "READY"
    receipt = observations["observations"][0]["cycle_receipt"]["artifact"]
    assert receipt["cycles"] == 23


def test_missing_empty_runs_and_composition_are_explicit_no_go_not_partial_output(
        tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(calibration.read_text(encoding="utf-8"))
    del document["empty_runs"]
    del document["composition_probe"]
    _write(calibration, document)

    report, calibration_out, observations, status = _MODULE.build(
        rtl, campaign, calibration, workloads)

    assert status == 1 and report["partial_output_is_admissible"] is False
    codes = {issue["code"] for issue in report["refusals"]}
    assert {"MISSING_EMPTY_BASELINES", "MISSING_COMPOSITION_PROBE"} <= codes
    assert calibration_out == {"samples": [], "empty_run_receipts": [],
                               "composition_receipt": {}}
    assert observations == {"expected_workloads": [], "observations": []}


def test_empty_baseline_requires_its_raw_compiler_command_buffer(tmp_path: Path) -> None:
    rtl, campaign, calibration, workloads = _inputs(tmp_path)
    document = json.loads(calibration.read_text(encoding="utf-8"))
    del document["empty_runs"][0]["command_buffer_artifact"]
    _write(calibration, document)

    report, calibration_out, observations, status = _MODULE.build(
        rtl, campaign, calibration, workloads)

    assert status == 1
    assert "MISSING_RAW_COMMAND_BUFFER" in {issue["code"] for issue in report["refusals"]}
    assert calibration_out["empty_run_receipts"] == []
    assert observations["observations"] == []


def test_movement_identity_accepts_exact_emitted_program_but_compute_still_requires_raw_cb() -> None:
    measurement = {
        "measurement_identity": {
            "program": {"kind": "compiler_emitted_program", "sha256": "a" * 64},
            "inputs": {"kind": "fixture", "sha256": "b" * 64},
            "toolchain": {"rtl_facts_sha256": "c" * 64},
        },
        "measurement_identity_refusals": [],
    }
    issues: list[dict[str, str]] = []

    movement = receipt_bridge._identity(
        measurement, "movement", issues, require_command_buffer=False,
        accepted_program_kinds=frozenset({"compiler_emitted_program"}))

    assert movement == measurement["measurement_identity"] and issues == []
    compute_issues: list[dict[str, str]] = []
    assert receipt_bridge._identity(
        measurement, "compute", compute_issues, require_command_buffer=True) == {}
    assert {issue["code"] for issue in compute_issues} == {"UNLINKED_COMMAND_BUFFER"}

    forged = copy.deepcopy(measurement)
    forged["measurement_identity"]["program"]["kind"] = "golden_answer"
    forged_issues: list[dict[str, str]] = []
    assert receipt_bridge._identity(
        forged, "movement", forged_issues, require_command_buffer=False,
        accepted_program_kinds=frozenset({"compiler_emitted_program"})) == {}
    assert {issue["code"] for issue in forged_issues} == {"UNLINKED_COMMAND_BUFFER"}

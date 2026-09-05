"""Execute an exact Gemmini RTL calibration campaign into bridge-ready raw records."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf.dma_volume import physical_volume_from_counters
from merlin.perf.work_volume import work_from_command_buffer

from . import gemmini_dma_calibration as dma
from . import gemmini_roofline_auxiliary as auxiliary
from .gemmini_codegen import CodegenError


SCHEMA = "gemmini_rtl_calibration_execution_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _exact_program(raw: Mapping[str, Any], command: Mapping[str, Any] | None) -> tuple[str, str]:
    if command is not None:
        return "compiler_command_buffer", _digest(command)
    emitter = raw.get("emitter")
    emitted = emitter.get("emitted_mlir_sha256") if isinstance(emitter, Mapping) else None
    if not _is_sha256(emitted):
        raise CodegenError("DMA result did not retain its exact compiler-emitted MLIR identity")
    return "compiler_emitted_program", emitted


def _measurement(raw: Mapping[str, Any], *, request_sha256: str, pass_name: str,
                 rtl_facts_sha256: str, capabilities_sha256: str,
                 circt_hw_sha256: str, command: Mapping[str, Any] | None = None) -> dict[str, Any]:
    program_kind, program_sha256 = _exact_program(raw, command)
    cycles = raw.get("cycles")
    conditions = raw.get("measurement_conditions")
    oracle = raw.get("oracle")
    if (raw.get("correct") is not True or isinstance(cycles, bool)
            or not isinstance(cycles, int) or cycles <= 0
            or not isinstance(conditions, Mapping) or not conditions
            or not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True):
        raise CodegenError("runner result lacks a correct positive cycle-accurate RTL measurement")
    identity = {
        "program": {"kind": program_kind, "sha256": program_sha256},
        "inputs": {"kind": "calibration_request", "sha256": request_sha256},
        "toolchain": {
            "target": "gemmini", "harness_capabilities_sha256": capabilities_sha256,
            "rtl_facts_sha256": rtl_facts_sha256,
            "circt_core_hw_sha256": circt_hw_sha256,
        },
    }
    emitter = raw.get("emitter")
    if isinstance(emitter, Mapping):
        identity["toolchain"]["compiler_program_receipt_sha256"] = _digest(emitter)
    rtl = {
        "cycles": cycles, "correct": True,
        "provenance": {**dict(oracle), "cycle_accurate": True,
                       "evidence": raw.get("elf_sha256")},
        "measurement_conditions": dict(conditions),
        "rtl_facts_sha256": rtl_facts_sha256,
    }
    counters = raw.get("counters")
    if isinstance(counters, Mapping):
        rtl["counters"] = dict(counters)
    return {
        "measurement_identity": identity, "measurement_identity_refusals": [],
        "rtl_facts_sha256": rtl_facts_sha256,
        "run_nonce": f"{request_sha256}:{pass_name}",
        "per_sim": {"rtl": rtl},
    }


def _admissible_counter_facts(binding: Mapping[str, Any] | None,
                              readings: Mapping[str, Any], rtl_facts_sha256: str) \
        -> tuple[list[Mapping[str, Any]], str]:
    if not isinstance(binding, Mapping):
        return [], "no counter-byte binding artifact was supplied"
    facts_raw = binding.get("counter_facts")
    facts = (list(facts_raw) if isinstance(facts_raw, Sequence)
             and not isinstance(facts_raw, (str, bytes)) else [])
    if (binding.get("status") not in {"exact", "proved", "resolved"}
            or binding.get("rtl_facts_sha256") != rtl_facts_sha256 or not facts):
        return [], str(binding.get("why") or "counter-byte binding is not proved")
    if (not all(isinstance(fact, Mapping)
                and fact.get("fact_kind") == "counter_byte_binding"
                and fact.get("artifact_sha256") == rtl_facts_sha256
                and fact.get("derived_from_rtl") is True and fact.get("provenance")
                for fact in facts)
            or {fact.get("counter_field") for fact in facts} != set(readings)):
        return [], "counter-byte facts do not bind every and only measured field"
    volume = physical_volume_from_counters(readings, counter_facts=facts)
    if volume.total_bytes is None or volume.total_bytes <= 0:
        return [], "; ".join(volume.unresolved) or "physical traffic is not positive and exact"
    return facts, "proved against exact RTL facts"


def _linked(occupancy_raw: Mapping[str, Any], physical_raw: Mapping[str, Any], *,
            request_sha256: str, rtl_facts_sha256: str, capabilities_sha256: str,
            circt_hw_sha256: str, counter_binding: Mapping[str, Any] | None,
            command: Mapping[str, Any] | None = None) -> dict[str, Any]:
    occupancy = _measurement(
        occupancy_raw, request_sha256=request_sha256, pass_name="occupancy",
        rtl_facts_sha256=rtl_facts_sha256, capabilities_sha256=capabilities_sha256,
        circt_hw_sha256=circt_hw_sha256, command=command)
    physical = _measurement(
        physical_raw, request_sha256=request_sha256, pass_name="physical_bytes",
        rtl_facts_sha256=rtl_facts_sha256, capabilities_sha256=capabilities_sha256,
        circt_hw_sha256=circt_hw_sha256, command=command)
    if occupancy["measurement_identity"] != physical["measurement_identity"]:
        raise CodegenError("counter passes differ in exact compiler-program identity")
    occupancy_rtl = occupancy["per_sim"]["rtl"]
    physical_rtl = physical["per_sim"]["rtl"]
    if occupancy_rtl["measurement_conditions"] != physical_rtl["measurement_conditions"]:
        raise CodegenError("counter passes differ in observed measurement conditions")
    occupancy_report = occupancy_rtl.get("counters")
    physical_report = physical_rtl.get("counters")
    if not isinstance(occupancy_report, Mapping) or not isinstance(physical_report, Mapping):
        raise CodegenError("counter passes did not retain both raw reports")
    occupancy_selection = occupancy_report.get("selection")
    physical_selection = physical_report.get("selection")
    if (not isinstance(occupancy_selection, Mapping)
            or occupancy_selection.get("kind") != "joint_occupancy"
            or occupancy_selection.get("unit") is not None
            or not isinstance(physical_selection, Mapping)
            or physical_selection.get("kind") != "unit"
            or not isinstance(physical_selection.get("unit"), str)
            or not physical_selection["unit"]):
        raise CodegenError("counter passes did not retain exact occupancy/unit selections")
    occupancy_layout = occupancy_report.get("occupancy")
    combinations = (occupancy_layout.get("by_combination")
                    if isinstance(occupancy_layout, Mapping) else None)
    occupancy_readings = occupancy_report.get("readings")
    selected_occupancy = (set(combinations.values()) if isinstance(combinations, Mapping)
                          and all(isinstance(name, str) and name
                                  for name in combinations.values()) else set())
    if (not selected_occupancy or not isinstance(occupancy_readings, Mapping)
            or set(occupancy_readings) != selected_occupancy
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in occupancy_readings.values())
            or sum(occupancy_readings.values()) > occupancy_rtl["cycles"]):
        raise CodegenError("occupancy readings do not fit their own exact cycle window")
    readings = physical_report.get("readings")
    selected = physical_report.get("selected_counters")
    if (not isinstance(readings, Mapping) or not readings or not isinstance(selected, Mapping)
            or set(readings) != set(selected)):
        raise CodegenError("physical pass did not retain one exact selected counter family")
    facts, standing = _admissible_counter_facts(
        counter_binding, readings, rtl_facts_sha256)
    physical_evidence: dict[str, Any] = {
        "unit_family": physical_report.get("selection", {}).get("unit"),
        "semantic_resolution": ("rtl_bound_physical_bytes" if facts
                                else "raw_named_readings_only"),
        "selected_counters": dict(selected), "readings": dict(readings),
        "binding_status": standing,
    }
    if facts:
        physical_evidence["counter_facts"] = [dict(fact) for fact in facts]
    result = dict(occupancy)
    result["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical}
    result["linked_counter_evidence"] = {
        "status": "linked", "refusals": [],
        "measurement_identity": occupancy["measurement_identity"],
        "rtl_facts_sha256": rtl_facts_sha256,
        "cycle_windows": {
            "occupancy": occupancy_rtl["cycles"], "physical_bytes": physical_rtl["cycles"],
            "instrumentation_delta": physical_rtl["cycles"] - occupancy_rtl["cycles"],
        },
        "occupancy": dict(occupancy_report),
        "physical_byte_counters": physical_evidence,
    }
    return result


def _command_evidence(command: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    work = work_from_command_buffer(command).to_dict()
    artifact = {
        "command_buffer": dict(command), "artifact_sha256": _digest(command),
        "compiler_provenance": "target-owned native calibration compiler",
    }
    bindings: dict[str, Any] = {}
    if work.get("exact_macs") is not None and work.get("artifact_sha256") == artifact["artifact_sha256"]:
        bindings["compute"] = {
            "resource": f"compute:{work['basis']}:{work['unit']}",
            "derived_from_tool": True,
            "provenance": "resource identity derived from the exact compiler command buffer",
        }
    return artifact, bindings


def _movement_binding(measurement: Mapping[str, Any]) -> dict[str, Any]:
    physical = measurement.get("linked_counter_evidence", {}).get("physical_byte_counters", {})
    if physical.get("semantic_resolution") != "rtl_bound_physical_bytes":
        return {}
    return {"movement": {
        "resource": "movement:physical_counters:bytes", "derived_from_tool": True,
        "provenance": "resource identity derived from exact RTL-bound physical counter facts",
    }}


def _protocol(request: Mapping[str, Any]) -> str:
    identity = request.get("identity")
    coordinates = identity.get("coordinates") if isinstance(identity, Mapping) else None
    value = (coordinates.get("measurement_protocol") if isinstance(coordinates, Mapping) else None)
    if value is None and isinstance(identity, Mapping):
        value = identity.get("measurement_protocol")
    if not isinstance(value, str) or not value:
        raise CodegenError("campaign request has no tool-derived execution protocol")
    return value


def _validate_request(request: Mapping[str, Any], *, rtl_facts_sha256: str,
                      capabilities_sha256: str) -> tuple[str, Mapping[str, Any]]:
    request_sha256, identity = request.get("request_sha256"), request.get("identity")
    if (not _is_sha256(request_sha256) or not isinstance(identity, Mapping)
            or _digest(identity) != request_sha256
            or identity.get("rtl_facts_sha256") != rtl_facts_sha256
            or identity.get("harness_capabilities_sha256") != capabilities_sha256):
        raise CodegenError("campaign request identity is not content-linked to exact inputs")
    return request_sha256, identity


def execute(manifest: Mapping[str, Any], rtl_facts: Mapping[str, Any], *,
            manifest_sha256: str, rtl_facts_sha256: str, capabilities_sha256: str,
            circt_hw_sha256: str, workdir: str | Path, timeout: int = 600,
            counter_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute every exact request. Any missing row leaves the whole artifact ``NO_GO``."""
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    contract = manifest.get("execution_contract")
    if (manifest.get("schema") != "rtl_calibration_campaign_v1"
            or manifest.get("dispatchable") is not True
            or not isinstance(contract, Mapping)
            or contract.get("partial_execution_is_admissible") is not False
            or not all(_is_sha256(value) for value in
            (manifest_sha256, rtl_facts_sha256, capabilities_sha256, circt_hw_sha256))):
        return {"schema": SCHEMA, "status": "NO_GO",
                "campaign_manifest_sha256": manifest_sha256, "results": [], "empty_runs": [],
                "composition_probe": None,
                "issues": ["campaign is not dispatchable with exact input identities"]}
    results: list[dict[str, Any]] = []
    empty_runs: list[dict[str, Any]] = []
    composition_row: dict[str, Any] | None = None
    issues: list[str] = []

    primary = manifest.get("measurement_requests")
    auxiliaries = manifest.get("auxiliary_measurement_requests")
    if (not isinstance(primary, Sequence) or isinstance(primary, (str, bytes))
            or not isinstance(auxiliaries, Sequence) or isinstance(auxiliaries, (str, bytes))):
        return {"schema": SCHEMA, "status": "NO_GO",
                "campaign_manifest_sha256": manifest_sha256, "results": [], "empty_runs": [],
                "composition_probe": None, "issues": ["campaign request arrays are absent"]}

    for index, raw_request in enumerate(primary):
        try:
            if not isinstance(raw_request, Mapping):
                raise CodegenError("request is not an object")
            request_sha256, identity = _validate_request(
                raw_request, rtl_facts_sha256=rtl_facts_sha256,
                capabilities_sha256=capabilities_sha256)
            coordinates = identity.get("coordinates")
            if not isinstance(coordinates, Mapping):
                raise CodegenError("request coordinates are absent")
            protocol = _protocol(raw_request)
            mechanism = raw_request.get("mechanism")
            if mechanism == "compute":
                multiple = coordinates.get("tile_multiple")
                raw = auxiliary.run_compute_probe(
                    rtl_facts, multiple, protocol, timeout=timeout,
                    workdir=root / f"primary-{index}", counter_unit=None)
                command = raw["command_buffer"]
                measurement = _measurement(
                    raw, request_sha256=request_sha256, pass_name="compute",
                    rtl_facts_sha256=rtl_facts_sha256,
                    capabilities_sha256=capabilities_sha256,
                    circt_hw_sha256=circt_hw_sha256, command=command)
                command_artifact, resource_bindings = _command_evidence(command)
                row = {"request_sha256": request_sha256, "identity": dict(identity),
                       "measurement": measurement,
                       "command_buffer_artifact": command_artifact,
                       "resource_bindings": resource_bindings}
            elif isinstance(mechanism, str) and mechanism.startswith("dma_"):
                direction = mechanism[len("dma_"):]
                payload = coordinates.get("transfer_bytes")
                occupancy_raw = dma.run_dma_calibration(
                    direction, payload, rtl_facts, protocol=protocol, timeout=timeout,
                    workdir=root / f"primary-{index}-occupancy", counter_unit=None)
                physical_raw = dma.run_dma_calibration(
                    direction, payload, rtl_facts, protocol=protocol, timeout=timeout,
                    workdir=root / f"primary-{index}-physical", counter_unit="BYTES")
                measurement = _linked(
                    occupancy_raw, physical_raw, request_sha256=request_sha256,
                    rtl_facts_sha256=rtl_facts_sha256,
                    capabilities_sha256=capabilities_sha256,
                    circt_hw_sha256=circt_hw_sha256, counter_binding=counter_binding)
                resource_bindings = _movement_binding(measurement)
                row = {"request_sha256": request_sha256, "identity": dict(identity),
                       "measurement": measurement,
                       "resource_bindings": resource_bindings}
                if not resource_bindings:
                    issues.append(
                        f"measurement_requests[{index}]: physical-byte semantics remain UNKNOWN; "
                        "raw named readings were retained but cannot calibrate movement")
            else:
                raise CodegenError(f"unsupported target mechanism {mechanism!r}")
            results.append(row)
        except Exception as exc:
            issues.append(f"measurement_requests[{index}]: {type(exc).__name__}: {exc}")

    for index, raw_request in enumerate(auxiliaries):
        try:
            if not isinstance(raw_request, Mapping):
                raise CodegenError("request is not an object")
            request_sha256, identity = _validate_request(
                raw_request, rtl_facts_sha256=rtl_facts_sha256,
                capabilities_sha256=capabilities_sha256)
            kind = identity.get("kind")
            if kind == "empty_run":
                protocol = _protocol(raw_request)
                raw = auxiliary.run_empty_workload(
                    protocol, timeout=timeout, workdir=root / f"empty-{index}")
                command = raw["command_buffer"]
                measurement = _measurement(
                    raw, request_sha256=request_sha256, pass_name="empty",
                    rtl_facts_sha256=rtl_facts_sha256,
                    capabilities_sha256=capabilities_sha256,
                    circt_hw_sha256=circt_hw_sha256, command=command)
                command_artifact, _bindings = _command_evidence(command)
                empty_runs.append({
                    "request_sha256": request_sha256, "identity": dict(identity),
                    "measurement": measurement,
                    "command_buffer_artifact": command_artifact,
                })
            elif kind == "composition_probe":
                protocol = _protocol(raw_request)
                movement_sizes = sorted({
                    request.get("identity", {}).get("coordinates", {}).get("transfer_bytes")
                    for request in primary if isinstance(request, Mapping)
                    and str(request.get("mechanism", "")).startswith("dma_")
                })
                compute_multiples = sorted({
                    request.get("identity", {}).get("coordinates", {}).get("tile_multiple")
                    for request in primary if isinstance(request, Mapping)
                    and request.get("mechanism") == "compute"
                })
                if (not movement_sizes or not compute_multiples
                        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                               for value in movement_sizes + compute_multiples)):
                    raise CodegenError("composition coordinates are not derived from primary requests")
                joint = auxiliary.run_joint_occupancy_probe(
                    rtl_facts, protocol=protocol, payload_bytes=movement_sizes[0],
                    compute_multiple=compute_multiples[0], rtl_facts_sha256=rtl_facts_sha256,
                    timeout=timeout, workdir=root / "composition-occupancy")
                if joint.get("status") != "measured":
                    raise CodegenError(str(joint.get("why", "resource roles remain unknown")))
                occupancy_raw = joint["composition_measurement"]
                physical_raw = auxiliary.run_compute_probe(
                    rtl_facts, compute_multiples[0], protocol, timeout=timeout,
                    workdir=root / "composition-physical", counter_unit="BYTES")
                command = occupancy_raw["command_buffer"]
                measurement = _linked(
                    occupancy_raw, physical_raw, request_sha256=request_sha256,
                    rtl_facts_sha256=rtl_facts_sha256,
                    capabilities_sha256=capabilities_sha256,
                    circt_hw_sha256=circt_hw_sha256, counter_binding=counter_binding,
                    command=command)
                role_binding = joint["resource_role_binding"]
                report = occupancy_raw["counters"]
                partition = report["partition"]
                layout = dict(role_binding["counter_layout"])
                layout["kinds"] = dict(role_binding["kinds"])
                command_artifact, bindings = _command_evidence(command)
                composition_row = {
                    "request_sha256": request_sha256, "identity": dict(identity),
                    "measurement": measurement,
                    "command_buffer_artifact": command_artifact,
                    "resource_bindings": bindings,
                    "occupancy_binding": {
                        "derived_from_rtl": True, "rtl_facts_sha256": rtl_facts_sha256,
                        "circt_hw_sha256": circt_hw_sha256,
                        "provenance": ("CIRCT boolean partition plus content-linked differential "
                                       f"probe {role_binding['artifact_sha256']}"),
                        "counter_layout": layout,
                        "codes": report["event_codes"], "module": partition["module"],
                        "counter_module": partition["counter_module"],
                    },
                }
            else:
                raise CodegenError(f"unsupported auxiliary request kind {kind!r}")
        except Exception as exc:
            issues.append(f"auxiliary_measurement_requests[{index}]: {type(exc).__name__}: {exc}")

    complete = (len(results) == len(primary)
                and len(empty_runs) == sum(isinstance(request, Mapping)
                    and request.get("identity", {}).get("kind") == "empty_run"
                    for request in auxiliaries)
                and composition_row is not None and not issues)
    return {
        "schema": SCHEMA, "status": "READY" if complete else "NO_GO",
        "campaign_manifest_sha256": manifest_sha256,
        "results": results, "empty_runs": empty_runs,
        "composition_probe": composition_row, "issues": issues,
        "partial_execution_is_admissible": False,
    }

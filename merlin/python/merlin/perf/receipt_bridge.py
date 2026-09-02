"""Convert runner evidence into the raw receipts consumed by ``build_rtl_roofline``.

This edge deliberately does not estimate anything.  It admits a measurement only when the runner
record contains a cycle-accurate RTL result, the exact compiler command buffer that was executed,
and (for movement) a second, identity-matched counter pass whose byte semantics are bound to the
exact RTL-facts artifact.  A missing proof makes the complete output ``NO_GO``; ready rows are not
published as a tempting partial data set.

The script understands the structural ``perf_results.json`` shape emitted by ``run_perf_bench`` but
does not name a target, simulator, approach, opcode, counter, geometry, or measurement protocol.
Those identities must be present in tool-produced evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf.dma_volume import physical_volume_from_counters


_SCHEMA = "rtl_roofline_receipt_bridge_v1"
_CAMPAIGN_SCHEMA = "rtl_calibration_campaign_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _issue(source: str, detail: str, code: str) -> dict[str, str]:
    return {"code": code, "source": source, "detail": detail}


def _load(path: Path, label: str) -> tuple[Any, dict[str, str], list[dict[str, str]]]:
    identity = {"path": str(path), "sha256": "UNKNOWN"}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return None, identity, [_issue(label, f"cannot read explicit path {path}: {exc}",
                                       "MISSING_INPUT")]
    identity["sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        return json.loads(payload), identity, []
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, identity, [_issue(label, f"input is not UTF-8 JSON: {exc}", "INVALID_INPUT")]


def _rows(value: Any, source: str, issues: list[dict[str, str]]) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    issues.append(_issue(source, "must be a JSON array", "INVALID_INPUT"))
    return ()


def _mapping(value: Any, source: str, issues: list[dict[str, str]]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    issues.append(_issue(source, "must be a JSON object", "INVALID_INPUT"))
    return {}


def _receipt(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(artifact)
    return {"artifact": value, "artifact_sha256": _digest(value)}


def _approach(row: Mapping[str, Any], source: str,
              issues: list[dict[str, str]]) -> Mapping[str, Any]:
    direct = row.get("measurement")
    if isinstance(direct, Mapping):
        return direct
    approaches = row.get("approaches")
    candidates = ([value for value in approaches.values() if isinstance(value, Mapping)]
                  if isinstance(approaches, Mapping) else [])
    if len(candidates) != 1:
        issues.append(_issue(
            source, "requires exactly one runner measurement approach", "AMBIGUOUS_MEASUREMENT"))
        return {}
    return candidates[0]


def _rtl_row(measurement: Mapping[str, Any], source: str,
             issues: list[dict[str, str]]) -> Mapping[str, Any]:
    per_sim = measurement.get("per_sim")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(per_sim, Mapping):
        for value in per_sim.values():
            provenance = value.get("provenance") if isinstance(value, Mapping) else None
            if (isinstance(value, Mapping) and isinstance(provenance, Mapping)
                    and provenance.get("derived_from_rtl") is True
                    and provenance.get("cycle_accurate") is True):
                candidates.append(value)
    if len(candidates) != 1:
        issues.append(_issue(
            source, "requires exactly one explicitly cycle-accurate RTL simulator result",
            "UNKNOWN_CYCLES"))
        return {}
    rtl = candidates[0]
    cycles = rtl.get("cycles")
    conditions = rtl.get("measurement_conditions")
    if (rtl.get("correct") is not True or isinstance(cycles, bool)
            or not isinstance(cycles, int) or cycles <= 0):
        issues.append(_issue(
            source, "RTL result must be correct with a positive integer cycle count",
            "UNKNOWN_CYCLES"))
    if not isinstance(conditions, Mapping) or not conditions:
        issues.append(_issue(
            source, "RTL result has no observed measurement conditions", "UNKNOWN_PROTOCOL"))
    return rtl


def _identity(measurement: Mapping[str, Any], source: str,
              issues: list[dict[str, str]], *,
              require_command_buffer: bool = True,
              accepted_program_kinds: frozenset[str] | None = None) -> Mapping[str, Any]:
    refusals = measurement.get("measurement_identity_refusals")
    identity = measurement.get("measurement_identity")
    if refusals != [] or not isinstance(identity, Mapping):
        issues.append(_issue(
            source, "runner did not establish one exact program/input/toolchain identity",
            "UNLINKED_MEASUREMENT"))
        return {}
    program = identity.get("program")
    expected_kinds = (frozenset({"compiler_command_buffer"})
                      if require_command_buffer else accepted_program_kinds)
    if (not isinstance(program, Mapping)
            or (expected_kinds is not None and program.get("kind") not in expected_kinds)
            or not isinstance(program.get("kind"), str) or not program["kind"].strip()
            or not _is_sha256(program.get("sha256"))):
        issues.append(_issue(
            source, ("measurement identity lacks the exact compiler command-buffer SHA-256"
                     if require_command_buffer else
                     "measurement identity lacks an accepted exact compiler-program artifact SHA-256"),
            "UNLINKED_COMMAND_BUFFER"))
        return {}
    return identity


def _command_buffer(row: Mapping[str, Any], measurement: Mapping[str, Any],
                    identity: Mapping[str, Any], source: str,
                    issues: list[dict[str, str]]) -> tuple[Mapping[str, Any], str]:
    evidence = row.get("command_buffer_artifact", measurement.get("command_buffer_artifact"))
    evidence = evidence if isinstance(evidence, Mapping) else {}
    command = evidence.get("command_buffer")
    provenance = evidence.get("compiler_provenance")
    program = identity.get("program") if isinstance(identity, Mapping) else None
    expected = program.get("sha256") if isinstance(program, Mapping) else None
    if not isinstance(command, Mapping):
        issues.append(_issue(
            source, "raw compiler command buffer is absent; a derived work total is insufficient",
            "MISSING_RAW_COMMAND_BUFFER"))
        return {}, ""
    actual = _digest(command)
    declared = evidence.get("artifact_sha256")
    if declared != actual:
        issues.append(_issue(
            source, f"raw command buffer declares {declared}, but its canonical SHA-256 is {actual}",
            "COMMAND_BUFFER_RECEIPT_MISMATCH"))
        return {}, ""
    if actual != expected:
        issues.append(_issue(
            source, f"raw command buffer SHA-256 {actual} does not match measured program {expected}",
            "COMMAND_BUFFER_MISMATCH"))
        return {}, ""
    if not isinstance(provenance, str) or not provenance.strip():
        issues.append(_issue(source, "raw command buffer lacks compiler provenance",
                             "UNPROVEN_COMMAND_BUFFER"))
        return {}, ""
    return command, provenance.strip()


def _resource(row: Mapping[str, Any], kind: str, source: str,
              issues: list[dict[str, str]]) -> str:
    bindings = row.get("resource_bindings")
    binding = bindings.get(kind) if isinstance(bindings, Mapping) else None
    if (not isinstance(binding, Mapping) or not isinstance(binding.get("resource"), str)
            or not binding["resource"].strip() or binding.get("derived_from_tool") is not True
            or not binding.get("provenance")):
        issues.append(_issue(
            source, f"{kind} resource identity is not present as tool-derived evidence",
            "UNKNOWN_RESOURCE"))
        return ""
    return binding["resource"].strip()


def _traffic(measurement: Mapping[str, Any], identity: Mapping[str, Any], rtl_sha256: str,
             source: str, issues: list[dict[str, str]]) \
        -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...],
                 Mapping[str, Any], Mapping[str, Any]]:
    linked = measurement.get("linked_counter_evidence")
    passes = measurement.get("counter_passes")
    if (not isinstance(linked, Mapping) or linked.get("status") != "linked"
            or linked.get("refusals") != [] or linked.get("measurement_identity") != identity):
        issues.append(_issue(source, "physical counters are not identity-linked to the cycle run",
                             "UNLINKED_COUNTER_PASSES"))
        return {}, (), {}, {}
    if not isinstance(passes, Mapping) or set(passes) != {"occupancy", "physical_bytes"}:
        issues.append(_issue(source, "requires the two raw counter-pass records",
                             "MISSING_COUNTER_PASSES"))
        return {}, (), {}, {}
    pass_rows: list[Mapping[str, Any]] = []
    pass_identities: list[Mapping[str, Any]] = []
    local_issues: list[dict[str, str]] = []
    for name in ("occupancy", "physical_bytes"):
        raw_pass = _mapping(passes[name], f"{source}.{name}", local_issues)
        pass_rtl = _rtl_row(raw_pass, f"{source}.{name}", local_issues)
        pass_rows.append(pass_rtl)
        pass_identities.append(_identity(
            raw_pass, f"{source}.{name}", local_issues, require_command_buffer=False))
        _rtl_binding(
            raw_pass, pass_rtl, rtl_sha256, f"{source}.{name}.rtl_binding", local_issues)
    if (local_issues or any(value != identity for value in pass_identities)
            or pass_rows[0].get("measurement_conditions")
            != pass_rows[1].get("measurement_conditions")):
        issues.extend(local_issues)
        issues.append(_issue(source, "counter passes differ in identity or conditions",
                             "COUNTER_PASS_MISMATCH"))
        return {}, (), {}, {}
    occupancy_report = pass_rows[0].get("counters")
    occupancy_layout = (occupancy_report.get("occupancy")
                        if isinstance(occupancy_report, Mapping) else None)
    combinations = (occupancy_layout.get("by_combination")
                    if isinstance(occupancy_layout, Mapping) else None)
    occupancy_readings = (occupancy_report.get("readings")
                          if isinstance(occupancy_report, Mapping) else None)
    selected_occupancy = (set(combinations.values()) if isinstance(combinations, Mapping)
                          and all(isinstance(value, str) and value
                                  for value in combinations.values()) else set())
    if (not selected_occupancy or not isinstance(occupancy_readings, Mapping)
            or set(occupancy_readings) != selected_occupancy
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in occupancy_readings.values())
            or sum(occupancy_readings.values()) > pass_rows[0]["cycles"]):
        issues.append(_issue(
            source, "occupancy counters do not fit their own measured cycle window",
            "OCCUPANCY_WINDOW_MISMATCH"))
        return {}, (), {}, {}
    byte_report = pass_rows[1].get("counters")
    physical = linked.get("physical_byte_counters")
    if not isinstance(byte_report, Mapping) or not isinstance(physical, Mapping):
        issues.append(_issue(source, "physical-byte counter report is absent",
                             "UNKNOWN_PHYSICAL_TRAFFIC"))
        return {}, (), {}, {}
    readings = physical.get("readings")
    selected = physical.get("selected_counters")
    if (not isinstance(readings, Mapping) or not readings
            or not isinstance(selected, Mapping) or set(readings) != set(selected)
            or readings != byte_report.get("readings")):
        issues.append(_issue(source, "raw physical readings do not exactly match the selected pass",
                             "COUNTER_READING_MISMATCH"))
        return {}, (), {}, {}
    facts_raw = physical.get("counter_facts")
    facts = (tuple(facts_raw) if isinstance(facts_raw, Sequence)
             and not isinstance(facts_raw, (str, bytes)) else ())
    if (not facts or not all(isinstance(fact, Mapping)
                             and fact.get("fact_kind") == "counter_byte_binding"
                             and fact.get("artifact_sha256") == rtl_sha256
                             and fact.get("derived_from_rtl") is True
                             and fact.get("provenance") for fact in facts)):
        issues.append(_issue(
            source, "raw unit counters lack byte semantics proved from the exact RTL-facts artifact",
            "UNPROVEN_PHYSICAL_BYTES"))
        return {}, (), {}, {}
    counter_fields = [fact.get("counter_field") for fact in facts]
    if (not all(isinstance(field, str) and field for field in counter_fields)
            or set(counter_fields) != set(readings)):
        issues.append(_issue(source, "byte-counter facts do not bind every and only measured field",
                             "COUNTER_BINDING_MISMATCH"))
        return {}, (), {}, {}
    physical_volume = physical_volume_from_counters(readings, counter_facts=facts)
    if physical_volume.total_bytes is None or physical_volume.total_bytes <= 0:
        issues.append(_issue(
            source, "RTL-bound counters do not establish one positive exact physical-byte total: "
            + "; ".join(physical_volume.unresolved), "UNKNOWN_PHYSICAL_TRAFFIC"))
        return {}, (), {}, {}
    return readings, facts, pass_rows[1], _mapping(passes["physical_bytes"], source, issues)


def _protocol(rtl: Mapping[str, Any], source: str,
              issues: list[dict[str, str]]) -> str:
    conditions = rtl.get("measurement_conditions")
    candidates = ([conditions.get(key) for key in ("measurement_protocol", "cache_protocol")
                   if isinstance(conditions.get(key), str) and conditions.get(key).strip()]
                  if isinstance(conditions, Mapping) else [])
    values = {value.strip() for value in candidates}
    if len(values) != 1:
        issues.append(_issue(source, "measurement conditions do not name the observed protocol",
                             "UNKNOWN_PROTOCOL"))
        return ""
    return next(iter(values))


def _rtl_binding(measurement: Mapping[str, Any], rtl: Mapping[str, Any], rtl_sha256: str,
                 source: str, issues: list[dict[str, str]]) -> None:
    identity = measurement.get("measurement_identity")
    toolchain = identity.get("toolchain") if isinstance(identity, Mapping) else None
    candidates = [measurement.get("rtl_facts_sha256"), rtl.get("rtl_facts_sha256"),
                  toolchain.get("rtl_facts_sha256") if isinstance(toolchain, Mapping) else None]
    if any(value != rtl_sha256 for value in candidates):
        issues.append(_issue(
            source, "every runner, RTL-result, and toolchain identity must carry the exact "
            "RTL-facts SHA-256",
            "UNLINKED_RTL_FACTS"))


def _raw_receipts(row: Mapping[str, Any], *, source: str, context: Mapping[str, Any],
                  rtl_sha256: str, need_traffic: bool, include_command_receipt: bool,
                  need_compute_resource: bool,
                  issues: list[dict[str, str]]) -> dict[str, Any]:
    measurement = _approach(row, source, issues)
    rtl = _rtl_row(measurement, f"{source}.rtl", issues)
    identity = _identity(
        measurement, f"{source}.identity", issues,
        require_command_buffer=include_command_receipt,
        accepted_program_kinds=(None if include_command_receipt
                                else frozenset({"compiler_emitted_program"})))
    command: Mapping[str, Any] = {}
    compiler_provenance = ""
    if include_command_receipt:
        command, compiler_provenance = _command_buffer(
            row, measurement, identity, f"{source}.command_buffer", issues)
    _rtl_binding(measurement, rtl, rtl_sha256, f"{source}.rtl_binding", issues)
    protocol = _protocol(rtl, f"{source}.protocol", issues)
    if context.get("measurement_protocol") != protocol:
        issues.append(_issue(source, "receipt context does not match observed measurement protocol",
                             "CONTEXT_MISMATCH"))
    readings: Mapping[str, Any] = {}
    facts: tuple[Mapping[str, Any], ...] = ()
    cycle_rtl = rtl
    cycle_measurement = measurement
    if need_traffic:
        readings, facts, byte_rtl, byte_measurement = _traffic(
            measurement, identity, rtl_sha256, f"{source}.traffic", issues)
        if byte_rtl:
            cycle_rtl = byte_rtl
            cycle_measurement = byte_measurement
    provenance = cycle_rtl.get("provenance") if isinstance(cycle_rtl, Mapping) else None
    cycle_provenance = json.dumps(provenance, sort_keys=True) if isinstance(provenance, Mapping) else ""
    if not cycle_provenance:
        issues.append(_issue(source, "RTL cycle result lacks structured provenance",
                             "UNPROVEN_CYCLES"))
    result = {
        "cycle_receipt": _receipt({
            "cycles": cycle_rtl.get("cycles"), "provenance": cycle_provenance,
            "runner_record_sha256": _digest(cycle_measurement), "context": dict(context),
        }),
    }
    if include_command_receipt:
        command_artifact = {
            "compiler_provenance": compiler_provenance,
            "command_buffer": dict(command), "context": dict(context),
        }
        if need_compute_resource:
            command_artifact["resource"] = _resource(
                row, "compute", f"{source}.compute_resource", issues)
        result["command_buffer_receipt"] = _receipt(command_artifact)
    if need_traffic:
        result["traffic_receipt"] = _receipt({
            "resource": _resource(row, "movement", f"{source}.movement_resource", issues),
            "rtl_facts_sha256": rtl_sha256, "readings": dict(readings),
            "counter_facts": [dict(fact) for fact in facts], "context": dict(context),
        })
    return result


def _collect_observations(document: Any, rtl_sha256: str,
                          issues: list[dict[str, str]]) -> dict[str, Any]:
    rows = _rows(document.get("results") if isinstance(document, Mapping) else document,
                 "workload_results", issues)
    names: list[str] = []
    observations: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        source = f"workload_results[{index}]"
        row = _mapping(raw, source, issues)
        workload = row.get("kernel", row.get("workload"))
        if not isinstance(workload, str) or not workload.strip():
            issues.append(_issue(source, "requires a nonempty workload identity", "UNKNOWN_WORKLOAD"))
            workload = ""
        measurement = _approach(row, source, issues)
        rtl = _rtl_row(measurement, f"{source}.rtl", issues)
        protocol = _protocol(rtl, f"{source}.protocol", issues)
        context = {"workload": workload, "measurement_protocol": protocol,
                   "rtl_facts_sha256": rtl_sha256}
        receipts = _raw_receipts(
            row, source=source, context=context, rtl_sha256=rtl_sha256,
            need_traffic=True, include_command_receipt=True, need_compute_resource=True,
            issues=issues)
        names.append(workload)
        observations.append({
            "workload": workload, "measurement_protocol": protocol,
            **receipts,
        })
    if (not names or any(not name for name in names)
            or any(count != 1 for count in Counter(names).values())):
        issues.append(_issue("workload_results", "workload identities must be nonempty and unique",
                             "INVALID_DENOMINATOR"))
    return {"expected_workloads": names, "observations": observations}


def _collect_calibration(document: Any, manifest: Mapping[str, Any], rtl_sha256: str,
                         circt_hw_sha256: str,
                         issues: list[dict[str, str]]) -> dict[str, Any]:
    execution = _mapping(document, "calibration_results", issues)
    expected_manifest_sha = execution.get("campaign_manifest_sha256")
    requests_raw = manifest.get("measurement_requests")
    requests = _rows(requests_raw, "campaign_manifest.measurement_requests", issues)
    by_hash: dict[str, Mapping[str, Any]] = {}
    for index, raw_request in enumerate(requests):
        request = _mapping(raw_request, f"campaign_manifest.measurement_requests[{index}]", issues)
        request_sha256 = request.get("request_sha256")
        if not _is_sha256(request_sha256) or request_sha256 in by_hash:
            issues.append(_issue(
                f"campaign_manifest.measurement_requests[{index}]",
                "request identity must have one unique lowercase SHA-256", "INVALID_REQUEST"))
            continue
        by_hash[request_sha256] = request
    auxiliary = _rows(manifest.get("auxiliary_measurement_requests"),
                      "campaign_manifest.auxiliary_measurement_requests", issues)
    manifest_inputs = manifest.get("inputs")
    capability_input = (manifest_inputs.get("harness_capabilities")
                        if isinstance(manifest_inputs, Mapping) else None)
    auxiliary_by_hash: dict[str, Mapping[str, Any]] = {}
    for index, raw_request in enumerate(auxiliary):
        request = _mapping(
            raw_request, f"campaign_manifest.auxiliary_measurement_requests[{index}]", issues)
        request_sha256 = request.get("request_sha256")
        identity = request.get("identity")
        if (not _is_sha256(request_sha256) or request_sha256 in auxiliary_by_hash
                or not isinstance(identity, Mapping) or _digest(identity) != request_sha256):
            issues.append(_issue(
                f"campaign_manifest.auxiliary_measurement_requests[{index}]",
                "auxiliary request must have one unique exact identity SHA-256", "INVALID_REQUEST"))
            continue
        if (identity.get("rtl_facts_sha256") != rtl_sha256
                or not isinstance(capability_input, Mapping)
                or identity.get("harness_capabilities_sha256") != capability_input.get("sha256")):
            issues.append(_issue(
                f"campaign_manifest.auxiliary_measurement_requests[{index}]",
                "auxiliary request is not linked to both exact plan inputs",
                "REQUEST_INPUT_MISMATCH"))
            continue
        auxiliary_by_hash[request_sha256] = request
    rows = _rows(execution.get("results"), "calibration_results.results", issues)
    observed = Counter(row.get("request_sha256") for row in rows
                       if isinstance(row, Mapping) and isinstance(row.get("request_sha256"), str))
    expected = Counter(by_hash.keys())
    if observed != expected:
        issues.append(_issue(
            "calibration_results.results",
            f"must cover every campaign request exactly once; missing={list((expected-observed).elements())}, "
            f"unexpected={list((observed-expected).elements())}", "PLAN_COVERAGE_MISMATCH"))
    samples: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        source = f"calibration_results.results[{index}]"
        row = _mapping(raw, source, issues)
        request = by_hash.get(row.get("request_sha256"))
        if not isinstance(request, Mapping):
            continue
        identity = request.get("identity")
        if (row.get("identity") != identity or not isinstance(identity, Mapping)
                or request.get("request_sha256") != _digest(identity)):
            issues.append(_issue(source, "execution identity does not match the exact campaign request",
                                 "REQUEST_IDENTITY_MISMATCH"))
            continue
        if (identity.get("rtl_facts_sha256") != rtl_sha256
                or not isinstance(capability_input, Mapping)
                or identity.get("harness_capabilities_sha256") != capability_input.get("sha256")):
            issues.append(_issue(source, "campaign request is not linked to both exact plan inputs",
                                 "REQUEST_INPUT_MISMATCH"))
            continue
        coordinates = identity.get("coordinates")
        sweep_id = identity.get("sweep_id")
        mechanism = request.get("mechanism")
        required_receipts = request.get("required_raw_receipts")
        measurement = _approach(row, source, issues)
        rtl = _rtl_row(measurement, f"{source}.rtl", issues)
        protocol = _protocol(rtl, f"{source}.protocol", issues)
        kind = ("compute" if isinstance(required_receipts, Sequence)
                and not isinstance(required_receipts, (str, bytes))
                and "compiler_command_buffer" in required_receipts
                and "physical_counter" not in required_receipts else
                "movement" if isinstance(required_receipts, Sequence)
                and not isinstance(required_receipts, (str, bytes))
                and "physical_counter" in required_receipts
                and "compiler_command_buffer" not in required_receipts else "")
        if (not isinstance(coordinates, Mapping) or not isinstance(sweep_id, str)
                or not isinstance(mechanism, str) or not mechanism or not kind):
            issues.append(_issue(source, "campaign request is structurally invalid", "INVALID_REQUEST"))
            continue
        coordinate_protocol = coordinates.get("measurement_protocol")
        identity_protocol = identity.get("measurement_protocol")
        declared_protocol = (coordinate_protocol if coordinate_protocol is not None
                             else identity_protocol)
        if (coordinate_protocol is not None and identity_protocol is not None
                and coordinate_protocol != identity_protocol):
            issues.append(_issue(source, "request identity contains conflicting protocols",
                                 "INVALID_REQUEST"))
        if declared_protocol is not None and declared_protocol != protocol:
            issues.append(_issue(source, "runner used a different protocol than the requested coordinate",
                                 "CONTEXT_MISMATCH"))
        context = {"sweep_id": sweep_id, "coordinates": dict(coordinates),
                   "measurement_protocol": protocol, "rtl_facts_sha256": rtl_sha256}
        receipts = _raw_receipts(
            row, source=source, context=context, rtl_sha256=rtl_sha256,
            need_traffic=kind == "movement",
            include_command_receipt=kind == "compute",
            need_compute_resource=kind == "compute", issues=issues)
        samples.append({
            "sweep_id": sweep_id, "coordinates": dict(coordinates),
            "measurement_protocol": protocol, "kind": kind, **receipts,
        })

    empty_receipts: list[dict[str, Any]] = []
    observed_auxiliary: list[str] = []
    if "empty_runs" not in execution:
        issues.append(_issue(
            "calibration_results.empty_runs",
            "runner output has no four-replicate empty baseline for each measured protocol",
            "MISSING_EMPTY_BASELINES"))
    for index, raw in enumerate(_rows(execution.get("empty_runs"),
                                      "calibration_results.empty_runs", issues)):
        source = f"calibration_results.empty_runs[{index}]"
        row = _mapping(raw, source, issues)
        request_sha256 = row.get("request_sha256")
        request = (auxiliary_by_hash.get(request_sha256)
                   if isinstance(request_sha256, str) else None)
        request_identity = request.get("identity") if isinstance(request, Mapping) else None
        if (not isinstance(request_identity, Mapping) or row.get("identity") != request_identity
                or request_identity.get("kind") != "empty_run"):
            issues.append(_issue(source, "empty run does not match one exact auxiliary request",
                                 "REQUEST_IDENTITY_MISMATCH"))
        else:
            observed_auxiliary.append(request_sha256)
        measurement = _approach(row, source, issues)
        rtl = _rtl_row(measurement, f"{source}.rtl", issues)
        protocol = _protocol(rtl, f"{source}.protocol", issues)
        if isinstance(request_identity, Mapping) and request_identity.get(
                "measurement_protocol") != protocol:
            issues.append(_issue(source, "empty run used a different protocol than requested",
                                 "CONTEXT_MISMATCH"))
        context = {"kind": "empty_run", "measurement_protocol": protocol,
                   "rtl_facts_sha256": rtl_sha256}
        receipts = _raw_receipts(
            row, source=source, context=context, rtl_sha256=rtl_sha256,
            need_traffic=False, include_command_receipt=True, need_compute_resource=False,
            issues=issues)
        command = receipts["command_buffer_receipt"]["artifact"].get("command_buffer")
        commands = command.get("commands") if isinstance(command, Mapping) else None
        if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)) or commands:
            issues.append(_issue(source, "empty-run compiler command sequence is not empty",
                                 "NONEMPTY_BASELINE"))
        empty_receipts.append({
            "measurement_protocol": protocol,
            "cycle_receipt": receipts["cycle_receipt"],
            "command_buffer_receipt": receipts["command_buffer_receipt"],
        })

    required_protocols = {sample["measurement_protocol"] for sample in samples
                          if sample.get("measurement_protocol")}
    observed_protocols = {item.get("measurement_protocol") for item in empty_receipts}
    if observed_protocols != required_protocols:
        issues.append(_issue(
            "calibration_results.empty_runs",
            "empty-run protocols must equal the calibration protocols exactly",
            "BASELINE_PROTOCOL_MISMATCH"))
    for protocol in sorted(required_protocols):
        matches = [item for item in empty_receipts
                   if item.get("measurement_protocol") == protocol]
        identities = [item["cycle_receipt"].get("artifact_sha256") for item in matches]
        cycle_values = {item["cycle_receipt"]["artifact"].get("cycles") for item in matches}
        if (len(matches) != 4 or len(set(identities)) != len(identities)
                or len(cycle_values) != 1):
            issues.append(_issue(
                "calibration_results.empty_runs",
                f"protocol {protocol!r} requires four distinct structurally-empty RTL runs",
                "INSUFFICIENT_EMPTY_BASELINE"))

    composition_row = execution.get("composition_probe")
    if "composition_probe" not in execution:
        issues.append(_issue(
            "calibration_results.composition_probe",
            "runner output has no joint-occupancy composition measurement",
            "MISSING_COMPOSITION_PROBE"))
    if isinstance(composition_row, Mapping):
        request_sha256 = composition_row.get("request_sha256")
        request = (auxiliary_by_hash.get(request_sha256)
                   if isinstance(request_sha256, str) else None)
        request_identity = request.get("identity") if isinstance(request, Mapping) else None
        if (not isinstance(request_identity, Mapping)
                or composition_row.get("identity") != request_identity
                or request_identity.get("kind") != "composition_probe"):
            issues.append(_issue(
                "calibration_results.composition_probe",
                "composition probe does not match one exact auxiliary request",
                "REQUEST_IDENTITY_MISMATCH"))
        else:
            observed_auxiliary.append(request_sha256)
    composition = _composition_receipt(
        composition_row, rtl_sha256, circt_hw_sha256, issues)
    expected_auxiliary = Counter(auxiliary_by_hash.keys())
    observed_auxiliary_counts = Counter(observed_auxiliary)
    if observed_auxiliary_counts != expected_auxiliary:
        issues.append(_issue(
            "calibration_results",
            "auxiliary execution does not exactly cover all empty-run and composition requests",
            "PLAN_COVERAGE_MISMATCH"))
    if expected_manifest_sha != manifest.get("_input_sha256"):
        issues.append(_issue("calibration_results.campaign_manifest_sha256",
                             "does not match the exact campaign manifest bytes",
                             "CAMPAIGN_MISMATCH"))
    return {"samples": samples, "empty_run_receipts": empty_receipts,
            "composition_receipt": composition}


def _composition_receipt(raw: Any, rtl_sha256: str, circt_hw_sha256: str,
                         issues: list[dict[str, str]]) -> dict[str, Any]:
    source = "calibration_results.composition_probe"
    row = _mapping(raw, source, issues)
    measurement = _approach(row, source, issues)
    rtl = _rtl_row(measurement, f"{source}.rtl", issues)
    identity = _identity(measurement, f"{source}.identity", issues)
    _rtl_binding(measurement, rtl, rtl_sha256, f"{source}.rtl_binding", issues)
    linked = measurement.get("linked_counter_evidence")
    occupancy = linked.get("occupancy") if isinstance(linked, Mapping) else None
    readings = occupancy.get("readings") if isinstance(occupancy, Mapping) else None
    binding = row.get("occupancy_binding")
    if (not isinstance(linked, Mapping) or linked.get("status") != "linked"
            or linked.get("refusals") != []
            or linked.get("measurement_identity") != identity
            or not isinstance(readings, Mapping) or not readings):
        issues.append(_issue(source, "composition probe lacks linked raw occupancy readings",
                             "UNKNOWN_COMPOSITION"))
        readings = {}
    passes = measurement.get("counter_passes")
    if isinstance(passes, Mapping) and set(passes) == {"occupancy", "physical_bytes"}:
        local_issues: list[dict[str, str]] = []
        occupancy_pass = _mapping(passes["occupancy"], f"{source}.occupancy", local_issues)
        physical_pass = _mapping(passes["physical_bytes"], f"{source}.physical_bytes", local_issues)
        occupancy_rtl = _rtl_row(occupancy_pass, f"{source}.occupancy", local_issues)
        physical_rtl = _rtl_row(physical_pass, f"{source}.physical_bytes", local_issues)
        _rtl_binding(occupancy_pass, occupancy_rtl, rtl_sha256,
                     f"{source}.occupancy.rtl_binding", local_issues)
        _rtl_binding(physical_pass, physical_rtl, rtl_sha256,
                     f"{source}.physical_bytes.rtl_binding", local_issues)
        occupancy_report = occupancy_rtl.get("counters")
        if (local_issues or _identity(occupancy_pass, f"{source}.occupancy", local_issues) != identity
                or _identity(physical_pass, f"{source}.physical_bytes", local_issues) != identity
                or occupancy_rtl.get("cycles") != rtl.get("cycles")
                or occupancy_rtl.get("measurement_conditions")
                != physical_rtl.get("measurement_conditions")
                or occupancy_rtl.get("measurement_conditions")
                != rtl.get("measurement_conditions")
                or not isinstance(occupancy_report, Mapping)
                or occupancy_report.get("readings") != readings):
            issues.extend(local_issues)
            issues.append(_issue(source, "composition occupancy does not match both raw counter passes",
                                 "COMPOSITION_PASS_MISMATCH"))
    else:
        issues.append(_issue(source, "composition probe does not retain both raw counter passes",
                             "MISSING_COUNTER_PASSES"))
    if (not isinstance(binding, Mapping) or binding.get("derived_from_rtl") is not True
            or binding.get("rtl_facts_sha256") != rtl_sha256
            or binding.get("circt_hw_sha256") != circt_hw_sha256
            or not binding.get("provenance")):
        issues.append(_issue(source, "occupancy layout is not proved from the exact RTL facts",
                             "UNPROVEN_OCCUPANCY_BINDING"))
        binding = {}
    layout = binding.get("counter_layout") if isinstance(binding, Mapping) else None
    combinations = layout.get("by_combination") if isinstance(layout, Mapping) else None
    codes = binding.get("codes") if isinstance(binding, Mapping) else None
    module = binding.get("module") if isinstance(binding, Mapping) else None
    counter_module = binding.get("counter_module") if isinstance(binding, Mapping) else None
    selected_names = (set(combinations.values()) if isinstance(combinations, Mapping)
                      and all(isinstance(value, str) and value for value in combinations.values())
                      else set())
    if (not selected_names or set(readings) != selected_names or not isinstance(codes, Mapping)
            or set(codes) != selected_names or not isinstance(module, str) or not module
            or not isinstance(counter_module, str) or not counter_module):
        issues.append(_issue(source, "occupancy proof does not bind every selected raw counter",
                             "INCOMPLETE_OCCUPANCY_BINDING"))
    artifact = {
        "rtl_facts_sha256": rtl_sha256,
        "source": binding.get("provenance", ""),
        "circt_hw_sha256": binding.get("circt_hw_sha256"),
        "counter_layout": layout,
        "readings": dict(readings), "cycles": rtl.get("cycles"),
        "codes": codes, "module": module, "counter_module": counter_module,
    }
    return _receipt(artifact)


def build(rtl_path: Path, manifest_path: Path, calibration_path: Path,
          workload_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    issues: list[dict[str, str]] = []
    rtl, rtl_input, load_issues = _load(rtl_path, "rtl_facts")
    issues.extend(load_issues)
    manifest_raw, manifest_input, load_issues = _load(manifest_path, "campaign_manifest")
    issues.extend(load_issues)
    calibration, calibration_input, load_issues = _load(calibration_path, "calibration_results")
    issues.extend(load_issues)
    workloads, workload_input, load_issues = _load(workload_path, "workload_results")
    issues.extend(load_issues)
    manifest = _mapping(manifest_raw, "campaign_manifest", issues)
    if (manifest.get("schema") != _CAMPAIGN_SCHEMA or manifest.get("dispatchable") is not True):
        issues.append(_issue("campaign_manifest", "campaign is not a complete dispatchable plan",
                             "NONDISPATCHABLE_CAMPAIGN"))
    execution_contract = manifest.get("execution_contract")
    if (not isinstance(execution_contract, Mapping)
            or execution_contract.get("partial_execution_is_admissible") is not False):
        issues.append(_issue(
            "campaign_manifest.execution_contract",
            "manifest does not prohibit partial execution", "UNSAFE_EXECUTION_CONTRACT"))
    inputs = manifest.get("inputs")
    manifest_rtl = inputs.get("rtl_facts") if isinstance(inputs, Mapping) else None
    if (not isinstance(rtl, Mapping) or not isinstance(manifest_rtl, Mapping)
            or manifest_rtl.get("sha256") != rtl_input["sha256"]):
        issues.append(_issue("campaign_manifest.inputs.rtl_facts",
                             "does not match the exact RTL-facts input bytes", "RTL_MISMATCH"))
    rtl_inputs = rtl.get("inputs") if isinstance(rtl, Mapping) else None
    circt_hw_sha256 = (rtl_inputs.get("core_hw_sha256")
                       if isinstance(rtl_inputs, Mapping) else None)
    if not _is_sha256(circt_hw_sha256):
        issues.append(_issue("rtl_facts.inputs.core_hw_sha256",
                             "exact elaborated CIRCT SHA-256 is absent", "UNKNOWN_CIRCT_INPUT"))
        circt_hw_sha256 = "UNKNOWN"
    manifest = dict(manifest)
    manifest["_input_sha256"] = manifest_input["sha256"]
    calibration_doc = _collect_calibration(
        calibration, manifest, rtl_input["sha256"], circt_hw_sha256, issues)
    observations_doc = _collect_observations(workloads, rtl_input["sha256"], issues)
    ready = not issues
    if not ready:
        calibration_doc = {"samples": [], "empty_run_receipts": [],
                           "composition_receipt": {}}
        observations_doc = {"expected_workloads": [], "observations": []}
    report = {
        "schema": _SCHEMA, "status": "READY" if ready else "NO_GO",
        "inputs": {"rtl_facts": rtl_input, "campaign_manifest": manifest_input,
                   "calibration_results": calibration_input, "workload_results": workload_input},
        "refusals": issues,
        "outputs_are_complete": ready,
        "partial_output_is_admissible": False,
        "output_content": {
            "calibration_canonical_sha256": _digest(calibration_doc),
            "observations_canonical_sha256": _digest(observations_doc),
        },
    }
    return report, calibration_doc, observations_doc, 0 if ready else 1

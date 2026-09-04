#!/usr/bin/env python3
"""Run the Gemmini performance corpus on one frozen, functionally complete Arm-4 compiler.

The runner deliberately has no "latest submission" discovery and no alternate learned/compiler arm.
The caller supplies the exact functional run ID and submission SHA-256.  The submission is copied into
this campaign, mounted read-only in a credential-free/networkless bwrap, and checked against its
functional fork before and after the corpus.  A campaign is GO only when every expected Arm-4
kernel/simulator cell is correct and reports a positive cycle count.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

import _pbcommon as PB
import perf_campaign as PC
from merlin.benchharness import hash_tree, runs_root as _runs_root
from merlin.perf.dma_volume import physical_volume_from_counters
from merlin.perf.work_volume import work_from_command_buffer
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.rtl import mlc_bridge
from merlin.targetgen.target_experiment import load_target_experiment


_FUNCTIONAL_RUNS = _runs_root(PB.TARGET, "capsule-bench")
_CONTRACT = str(PB.REPO / "merlin/contract")
_DESCRIPTOR = (PB.REPO / "merlin/experiments/capsule_bench/targets" / PB.TARGET
               / "target_experiment.yaml")
_FIXED_PROFILE_FAMILY = "fixed_profile"
_FIXED_PROFILE_REPLICATE = "r000"
_PHYSICAL_BYTE_UNIT = "BYTES"
_COUNTER_ENV = ("MERLIN_HW_COUNTERS", "MERLIN_HW_COUNTER_UNIT")


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_rtl_identity(path: Path, target: str) -> dict:
    """Bind this run to exact extractor JSON and the elaborated CIRCT it names."""
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PC.CampaignGateError(f"cannot read exact RTL facts {path}: {exc}") from exc
    inputs = document.get("inputs") if isinstance(document, Mapping) else None
    recorded = inputs.get("core_hw_sha256") if isinstance(inputs, Mapping) else None
    circt_path = mlc_bridge.core_hw_mlir(target)
    if not _is_sha256(recorded) or circt_path is None or not Path(circt_path).is_file():
        raise PC.CampaignGateError(
            "RTL facts do not identify one available elaborated CIRCT input by full SHA-256")
    circt_payload = Path(circt_path).read_bytes()
    actual = hashlib.sha256(circt_payload).hexdigest()
    if recorded != actual:
        raise PC.CampaignGateError(
            "RTL facts core_hw_sha256 does not match the active elaborated CIRCT bytes")
    return {
        "rtl_facts": {"path": str(path.resolve()),
                      "sha256": hashlib.sha256(payload).hexdigest()},
        "circt_core_hw": {"path": str(Path(circt_path).resolve()), "sha256": actual},
    }


@contextmanager
def _counter_environment(*, enabled: bool, unit: str | None = None):
    """Scope instrumentation to one pass and restore the caller's environment exactly."""
    previous = {name: os.environ.get(name) for name in _COUNTER_ENV}
    try:
        if enabled:
            os.environ["MERLIN_HW_COUNTERS"] = "1"
            if unit is None:
                os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
            else:
                os.environ["MERLIN_HW_COUNTER_UNIT"] = unit
        else:
            os.environ.pop("MERLIN_HW_COUNTERS", None)
            os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _measurement_identity(*, package_before: str, package_after: str,
                          inputs_before: str, inputs_after: str,
                          work_volume: Mapping, toolchain_shas: object,
                          target: str, expected_package_sha256: str | None,
                          rtl_facts_sha256: str | None = None) -> tuple[dict, list[str]]:
    """Build independently observed pass identity; return every reason it is not exact."""
    refusals: list[str] = []
    program_sha256 = work_volume.get("artifact_sha256")
    if not _is_sha256(program_sha256):
        refusals.append("graded command buffer has no exact SHA-256 identity")
    for label, value in (("submission before pass", package_before),
                         ("submission after pass", package_after),
                         ("capsule inputs before pass", inputs_before),
                         ("capsule inputs after pass", inputs_after)):
        if not _is_sha256(value):
            refusals.append(f"{label} has no exact SHA-256 identity")
    if package_before != package_after:
        refusals.append("frozen submission changed during the counter pass")
    if inputs_before != inputs_after:
        refusals.append("frozen capsule inputs changed during the counter pass")
    if expected_package_sha256 is not None and package_before != expected_package_sha256:
        refusals.append("counter pass did not execute the certified functional submission")
    if not _is_sha256(rtl_facts_sha256):
        refusals.append("counter pass is not bound to exact RTL-facts bytes")

    revisions: dict[str, str] = {}
    if not isinstance(toolchain_shas, Mapping) or not toolchain_shas:
        refusals.append("grade has no exact toolchain revision map")
    else:
        for raw_name, raw_revision in toolchain_shas.items():
            if not isinstance(raw_name, str) or not raw_name:
                refusals.append("toolchain revision map contains an invalid component name")
                continue
            if (not isinstance(raw_revision, str) or not raw_revision
                    or raw_revision.strip().upper() == "UNKNOWN"):
                refusals.append(f"toolchain component {raw_name!r} has no exact revision")
                continue
            revisions[raw_name] = raw_revision

    identity = {
        "program": {"kind": "compiler_command_buffer", "sha256": program_sha256},
        "inputs": {"kind": "frozen_capsule_tree", "sha256": inputs_before},
        "toolchain": {
            "target": target,
            "frozen_submission_sha256": package_before,
            "recorded_revisions": dict(sorted(revisions.items())),
            "rtl_facts_sha256": rtl_facts_sha256,
        },
    }
    return identity, refusals


def _rtl_counter_row(pass_result: Mapping) -> Mapping | None:
    per_sim = pass_result.get("per_sim")
    if not isinstance(per_sim, Mapping):
        return None
    rtl = per_sim.get("gsim")
    if not isinstance(rtl, Mapping):
        rtl = per_sim.get("verilator")
    return rtl if isinstance(rtl, Mapping) else None


def _counter_report(pass_result: Mapping) -> Mapping | None:
    rtl = _rtl_counter_row(pass_result)
    counters = rtl.get("counters") if isinstance(rtl, Mapping) else None
    return counters if isinstance(counters, Mapping) else None


def _selected_counter_names(report: Mapping, *, occupancy: bool) -> set[str] | None:
    if occupancy:
        description = report.get("occupancy")
        combinations = description.get("by_combination") if isinstance(description, Mapping) else None
        if not isinstance(combinations, Mapping) or not combinations:
            return None
        raw_names = list(combinations.values())
    else:
        selected = report.get("selected_counters")
        if not isinstance(selected, Mapping) or not selected:
            return None
        raw_names = list(selected)
    return (set(raw_names)
            if all(isinstance(name, str) and name for name in raw_names) else None)


def _copy_mapping(value: object) -> dict | None:
    return dict(value) if isinstance(value, Mapping) else None


def _admissible_counter_facts(binding: object, readings: object, *,
                              rtl_facts_sha256: str | None) -> tuple[list[dict], str]:
    """Return only a complete, proved byte binding; never promote structural candidates."""
    if not isinstance(binding, Mapping):
        return [], "no counter-byte binding probe was supplied"
    facts = binding.get("counter_facts")
    if (binding.get("status") not in ("exact", "proved", "resolved")
            or not isinstance(facts, list) or not facts):
        return [], str(binding.get("why") or "counter-byte semantics remain UNKNOWN")
    if not isinstance(readings, Mapping) or not _is_sha256(rtl_facts_sha256):
        return [], "counter readings or exact RTL-facts identity are absent"
    if (binding.get("rtl_facts_sha256") != rtl_facts_sha256
            or any(not isinstance(fact, Mapping)
                   or fact.get("fact_kind") != "counter_byte_binding"
                   or fact.get("artifact_sha256") != rtl_facts_sha256
                   or fact.get("derived_from_rtl") is not True
                   or not fact.get("provenance") for fact in facts)):
        return [], "counter-byte facts are not proved from the exact RTL-facts artifact"
    fields = [fact.get("counter_field") for fact in facts]
    if (not all(isinstance(field, str) and field for field in fields)
            or len(set(fields)) != len(fields) or set(fields) != set(readings)):
        return [], "counter-byte facts do not exhaustively bind the selected readings"
    physical = physical_volume_from_counters(readings, counter_facts=facts)
    if physical.total_bytes is None or physical.total_bytes <= 0:
        return [], "counter-byte facts do not establish a positive exact physical volume"
    return [dict(fact) for fact in facts], "exact RTL-derived byte semantics"


def _linked_identity_refusals(identity: object) -> list[str]:
    """Validate a pass-produced identity again at the trust boundary that joins passes."""
    if not isinstance(identity, Mapping):
        return ["measurement identity is not a mapping"]
    refusals: list[str] = []
    program = identity.get("program")
    if (not isinstance(program, Mapping) or program.get("kind") != "compiler_command_buffer"
            or not _is_sha256(program.get("sha256"))):
        refusals.append("program identity is not an exact compiler-command-buffer SHA-256")
    inputs = identity.get("inputs")
    if (not isinstance(inputs, Mapping) or inputs.get("kind") != "frozen_capsule_tree"
            or not _is_sha256(inputs.get("sha256"))):
        refusals.append("input identity is not an exact frozen-capsule-tree SHA-256")
    toolchain = identity.get("toolchain")
    if not isinstance(toolchain, Mapping):
        refusals.append("toolchain identity is not a mapping")
    else:
        if not isinstance(toolchain.get("target"), str) or not toolchain.get("target"):
            refusals.append("toolchain identity has no target")
        if not _is_sha256(toolchain.get("frozen_submission_sha256")):
            refusals.append("toolchain identity has no exact frozen-submission SHA-256")
        revisions = toolchain.get("recorded_revisions")
        if not isinstance(revisions, Mapping) or not revisions:
            refusals.append("toolchain identity has no recorded revision map")
        elif any(not isinstance(name, str) or not name
                 or not isinstance(value, str) or not value
                 or value.strip().upper() == "UNKNOWN"
                 for name, value in revisions.items()):
            refusals.append("toolchain identity contains an unknown or malformed revision")
    return refusals


def _link_counter_passes(occupancy_pass: Mapping, byte_pass: Mapping, *,
                         physical_unit: str, counter_binding: object = None,
                         rtl_facts_sha256: str | None = None) -> dict:
    """Link two independent RTL runs without assigning semantics to raw unit counters."""
    refusals: list[str] = []
    for label, result in (("occupancy", occupancy_pass), ("physical-byte", byte_pass)):
        pass_refusals = result.get("measurement_identity_refusals")
        if not isinstance(pass_refusals, list):
            refusals.append(f"{label} pass has no identity validation record")
        else:
            refusals.extend(f"{label} pass: {reason}" for reason in pass_refusals)
    occupancy_identity = occupancy_pass.get("measurement_identity")
    byte_identity = byte_pass.get("measurement_identity")
    for label, identity in (("occupancy", occupancy_identity),
                            ("physical-byte", byte_identity)):
        refusals.extend(f"{label} pass: {reason}"
                        for reason in _linked_identity_refusals(identity))
    if not isinstance(occupancy_identity, Mapping) or not isinstance(byte_identity, Mapping):
        refusals.append("both passes must carry exact program/input/toolchain identities")
    elif occupancy_identity != byte_identity:
        refusals.append("counter-pass program/input/toolchain identities differ")

    rtl_rows: list[Mapping] = []
    for label, result in (("occupancy", occupancy_pass), ("physical-byte", byte_pass)):
        rtl = _rtl_counter_row(result)
        if not isinstance(rtl, Mapping):
            refusals.append(f"{label} pass has no RTL simulator result")
            rtl_rows.append({})
            continue
        rtl_rows.append(rtl)
        cycles = rtl.get("cycles")
        if (rtl.get("correct") is not True or not isinstance(cycles, int)
                or isinstance(cycles, bool) or cycles <= 0):
            refusals.append(f"{label} pass is not a correct positive-cycle RTL measurement")
    if len(rtl_rows) == 2:
        conditions = [row.get("measurement_conditions") for row in rtl_rows]
        if not all(isinstance(item, Mapping) and item for item in conditions):
            refusals.append("both passes must state their measurement conditions")
        elif conditions[0] != conditions[1]:
            refusals.append("counter passes report different measurement conditions")

    reports = [_counter_report(occupancy_pass), _counter_report(byte_pass)]
    expected_selections = (("joint_occupancy", None), ("unit", physical_unit))
    for index, (label, report) in enumerate(zip(("occupancy", "physical-byte"), reports)):
        if report is None:
            refusals.append(f"{label} pass has no counter report")
            continue
        selection = report.get("selection")
        expected_kind, expected_unit = expected_selections[index]
        if (not isinstance(selection, Mapping) or selection.get("kind") != expected_kind
                or selection.get("unit") != expected_unit):
            refusals.append(f"{label} pass reports the wrong counter selection")
        names = _selected_counter_names(report, occupancy=index == 0)
        readings = report.get("readings")
        if not isinstance(readings, Mapping) or not readings:
            refusals.append(f"{label} pass has no raw named counter readings")
        elif names is None:
            refusals.append(f"{label} pass has no exact selected-counter set")
        elif set(readings) != names:
            refusals.append(f"{label} pass did not report every and only selected counter")
        elif any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                 for value in readings.values()):
            refusals.append(f"{label} pass has a non-integer or negative raw counter reading")

    if reports[0] is not None and rtl_rows:
        occupancy = reports[0].get("occupancy")
        combinations = occupancy.get("by_combination") if isinstance(occupancy, Mapping) else None
        readings = reports[0].get("readings")
        selected = (set(combinations.values()) if isinstance(combinations, Mapping)
                    and all(isinstance(value, str) and value
                            for value in combinations.values()) else set())
        if (not selected or not isinstance(readings, Mapping) or set(readings) != selected
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in readings.values())
                or sum(readings.values()) > rtl_rows[0].get("cycles", -1)):
            refusals.append("occupancy readings do not fit their own RTL cycle window")

    counter_schema_sha256 = None
    counter_capacity = None
    if all(report is not None for report in reports):
        schemas = [report.get("measured_header_sha256") for report in reports]
        discovered = [report.get("discovery") for report in reports]
        discovered_shas = [item.get("header_sha256") if isinstance(item, Mapping) else None
                           for item in discovered]
        if (not all(isinstance(item, Mapping) and item.get("status") == "derived"
                    for item in discovered)
                or not all(_is_sha256(value) for value in schemas + discovered_shas)
                or len(set(schemas + discovered_shas)) != 1):
            refusals.append("counter passes do not share one exact measured header identity")
        else:
            counter_schema_sha256 = schemas[0]
        capacities = [report.get("capacity") for report in reports]
        if (not all(isinstance(value, Mapping) and value.get("status") == "derived"
                    and isinstance(value.get("slots"), int)
                    and not isinstance(value.get("slots"), bool) and value.get("slots") > 0
                    and isinstance(value.get("provenance"), Mapping)
                    and _is_sha256(value["provenance"].get("sha256"))
                    for value in capacities) or capacities[0] != capacities[1]):
            refusals.append("counter passes do not share one exact derived capacity receipt")
        else:
            counter_capacity = dict(capacities[0])

    physical_evidence = {
        "unit_family": physical_unit,
        "semantic_resolution": "raw_named_readings_only",
        "selected_counters": (_copy_mapping(reports[1].get("selected_counters"))
                              if reports[1] is not None else None),
        "readings": (_copy_mapping(reports[1].get("readings"))
                     if reports[1] is not None else None),
    }
    facts, binding_status = _admissible_counter_facts(
        counter_binding, physical_evidence["readings"],
        rtl_facts_sha256=rtl_facts_sha256)
    if facts:
        physical_evidence["semantic_resolution"] = "rtl_bound_physical_bytes"
        physical_evidence["counter_facts"] = facts
    physical_evidence["binding_status"] = binding_status
    return {
        "status": "linked" if not refusals else "refused",
        "refusals": refusals,
        "measurement_identity": dict(occupancy_identity) if not refusals else None,
        "counter_instrument": {
            "measured_header_sha256": counter_schema_sha256,
            "capacity": counter_capacity,
        },
        "occupancy": dict(reports[0]) if reports[0] is not None else None,
        "rtl_facts_sha256": rtl_facts_sha256 if _is_sha256(rtl_facts_sha256) else None,
        "cycle_windows": {
            "occupancy": rtl_rows[0].get("cycles") if len(rtl_rows) == 2 else None,
            "physical_bytes": rtl_rows[1].get("cycles") if len(rtl_rows) == 2 else None,
            "instrumentation_delta": (
                rtl_rows[1].get("cycles") - rtl_rows[0].get("cycles")
                if len(rtl_rows) == 2
                and isinstance(rtl_rows[0].get("cycles"), int)
                and not isinstance(rtl_rows[0].get("cycles"), bool)
                and isinstance(rtl_rows[1].get("cycles"), int)
                and not isinstance(rtl_rows[1].get("cycles"), bool) else None),
        },
        "physical_byte_counters": physical_evidence,
    }


def _collect_linked_counter_passes(run_one: Callable[[str], dict], *,
                                   physical_unit: str, counter_binding: object = None,
                                   rtl_facts_sha256: str | None = None) -> dict:
    """Execute occupancy and byte-family passes under disjoint instrumentation environments."""
    with _counter_environment(enabled=True, unit=None):
        occupancy = run_one("occupancy")
    with _counter_environment(enabled=True, unit=physical_unit):
        physical_bytes = run_one("physical_bytes")
    linked = _link_counter_passes(
        occupancy, physical_bytes, physical_unit=physical_unit,
        counter_binding=counter_binding, rtl_facts_sha256=rtl_facts_sha256)
    result = dict(occupancy)
    result["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical_bytes}
    result["linked_counter_evidence"] = linked
    return result


def _resource_bindings(measurement: Mapping) -> dict[str, dict]:
    """Name only resource axes established by the artifacts actually carried by this run."""
    out: dict[str, dict] = {}
    work = measurement.get("work_volume")
    command_artifact = measurement.get("command_buffer_artifact")
    command = (command_artifact.get("command_buffer")
               if isinstance(command_artifact, Mapping) else None)
    derived_work = work_from_command_buffer(command) if isinstance(command, Mapping) else None
    if (isinstance(work, Mapping) and derived_work is not None
            and isinstance(work.get("exact_macs"), int)
            and not isinstance(work.get("exact_macs"), bool)
            and work.get("exact_macs") > 0
            and derived_work.exact_macs == work.get("exact_macs")
            and _canonical_sha256(command) == work.get("artifact_sha256")
            and command_artifact.get("artifact_sha256") == work.get("artifact_sha256")
            and isinstance(work.get("basis"), str) and work.get("basis")
            and isinstance(work.get("unit"), str) and work.get("unit")):
        out["compute"] = {
            "resource": f"compute:{work['basis']}:{work['unit']}",
            "derived_from_tool": True,
            "provenance": ("resource axis derived from the exact compiler command-buffer work "
                           f"receipt {work['artifact_sha256']}"),
        }
    linked = measurement.get("linked_counter_evidence")
    physical = linked.get("physical_byte_counters") if isinstance(linked, Mapping) else None
    facts = physical.get("counter_facts") if isinstance(physical, Mapping) else None
    if (isinstance(facts, list) and facts
            and physical.get("semantic_resolution") == "rtl_bound_physical_bytes"):
        out["movement"] = {
            "resource": "movement:physical_counters:bytes",
            "derived_from_tool": True,
            "provenance": "resource axis derived from exhaustive RTL-bound physical-byte counters",
        }
    return out


def _roofline_auxiliary_requirements(results: list[dict], rtl_identity: Mapping) -> dict:
    """Expose required baselines/probe and fail closed when the runner has no honest path."""
    protocols: set[str] = set()
    profiled: list[str] = []
    raw_composition_probe: dict | None = None
    rtl_facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    circt = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    for cell in results:
        approaches = cell.get("approaches")
        candidates = ([value for value in approaches.values() if isinstance(value, Mapping)]
                      if isinstance(approaches, Mapping) else [])
        for measurement in candidates:
            per_sim = measurement.get("per_sim")
            if not isinstance(per_sim, Mapping):
                continue
            for sim_result in per_sim.values():
                provenance = sim_result.get("provenance") if isinstance(sim_result, Mapping) else None
                conditions = (sim_result.get("measurement_conditions")
                              if isinstance(sim_result, Mapping) else None)
                if (not isinstance(provenance, Mapping)
                        or provenance.get("derived_from_rtl") is not True
                        or provenance.get("cycle_accurate") is not True
                        or not isinstance(conditions, Mapping)):
                    continue
                values = {conditions.get(key) for key in ("measurement_protocol", "cache_protocol")
                          if isinstance(conditions.get(key), str) and conditions.get(key)}
                if len(values) == 1:
                    protocols.update(values)
                if (isinstance(measurement.get("linked_counter_evidence"), Mapping)
                        and measurement["linked_counter_evidence"].get("status") == "linked"):
                    profiled.append(str(cell.get("kernel") or ""))
                    linked = measurement["linked_counter_evidence"]
                    occupancy = linked.get("occupancy")
                    overlap = occupancy.get("overlap") if isinstance(occupancy, Mapping) else None
                    proof = overlap.get("partition_proof") if isinstance(overlap, Mapping) else None
                    layout = occupancy.get("occupancy") if isinstance(occupancy, Mapping) else None
                    discovery = occupancy.get("discovery") if isinstance(occupancy, Mapping) else None
                    if (raw_composition_probe is None and isinstance(rtl_facts, Mapping)
                            and isinstance(circt, Mapping)
                            and linked.get("rtl_facts_sha256") == rtl_facts.get("sha256")
                            and isinstance(proof, Mapping) and proof.get("status") == "proved"
                            and proof.get("sha256") == circt.get("sha256")
                            and isinstance(layout, Mapping)
                            and isinstance(discovery, Mapping)
                            and isinstance(occupancy.get("readings"), Mapping)):
                        raw_composition_probe = {
                            "workload": str(cell.get("kernel") or ""),
                            "rtl_facts_sha256": rtl_facts.get("sha256"),
                            "circt_core_hw": dict(circt),
                            "cycles": sim_result.get("cycles"),
                            "measurement_conditions": dict(conditions),
                            "counter_layout": dict(layout),
                            "readings": dict(occupancy["readings"]),
                            "codes": dict(discovery.get("event_codes") or {}),
                            "partition_proof": dict(proof),
                        }
    baseline_rows = [{
        "measurement_protocol": protocol,
        "required_replicates": 4,
        "status": "UNKNOWN",
        "receipts": [],
        "why": ("the performance corpus has no structurally-empty workload emitted by the frozen "
                "compiler; running a hand-authored empty kernel would not measure the same compiler path"),
    } for protocol in sorted(protocols)]
    composition_why = (
        "joint occupancy was collected, but its engine ResourceKind mapping is not derived from "
        "RTL/tool evidence; a role inferred from an engine name is forbidden"
        if raw_composition_probe is not None else
        "no joint-occupancy reading was proved against the exact CIRCT and RTL-facts inputs")
    composition = {
        "status": "UNKNOWN",
        "candidate_profiled_workloads": sorted(set(profiled)),
        "circt_core_hw": dict(circt) if isinstance(circt, Mapping) else None,
        "raw_probe": raw_composition_probe,
        "why": composition_why,
    }
    refusals = []
    if not protocols:
        refusals.append("no cycle-accurate RTL measurement reported an actual protocol")
    refusals.extend(
        f"{row['measurement_protocol']}: four compiler-produced empty RTL baselines are absent"
        for row in baseline_rows)
    refusals.append(composition["why"])
    return {
        "schema": "rtl_roofline_auxiliary_requirements_v1",
        "status": "NO_GO",
        "rtl_identity": dict(rtl_identity),
        "empty_run_requirements": baseline_rows,
        "composition_probe": composition,
        "refusals": refusals,
        "partial_evidence_is_admissible": False,
    }


def _probe_counter_byte_bindings(rtl_identity: Mapping) -> dict:
    """Run the target-owned structural probe and retain UNKNOWN rather than completing semantics."""
    try:
        from merlin.runtime.backends.base import get_backend
        backend = get_backend(PB.TARGET)
        probe = importlib.import_module(f"{backend.__name__}.counter_byte_bindings")
        artifact = probe.probe_counter_byte_bindings()
    except Exception as exc:  # probe failure is evidence unavailability, not a campaign crash
        return {"status": "unknown", "counter_facts": [],
                "why": f"{type(exc).__name__}: {exc}"}
    circt = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    probe_inputs = artifact.get("inputs") if isinstance(artifact, Mapping) else None
    probe_circt = probe_inputs.get("circt_core_hw") if isinstance(probe_inputs, Mapping) else None
    if (not isinstance(circt, Mapping) or not isinstance(probe_circt, Mapping)
            or circt.get("sha256") != probe_circt.get("sha256")):
        return {"status": "unknown", "counter_facts": [],
                "why": "counter-byte probe did not inspect the exact active CIRCT bytes",
                "probe": artifact}
    return dict(artifact)


#: The loop tier runs every kernel; the cert tier is the expensive one this policy rations.
_LOOP_SIM, _CERT_SIM, _CERT_TIER = "spike", "verilator", "L3"


def _selected_corpus(selection: str, kernels_root: Path = PB.KERNELS) -> list[dict]:
    doc = yaml.safe_load((kernels_root / "kernel_corpus.yaml").read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise PC.CampaignGateError("performance kernel corpus is not a mapping")
    corpus = [row for section in ("golden_kernels", "model_kernels", "attention_kernels",
                                  "conv_kernels", "movement_kernels")
              for row in (doc.get(section) or [])]
    if selection != "all":
        wanted = {value.strip() for value in selection.split(",") if value.strip()}
        known = {str(row.get("id")) for row in corpus}
        missing = sorted(wanted - known)
        if missing:
            raise PC.CampaignGateError(f"unknown performance kernel id(s): {missing}")
        corpus = [row for row in corpus if str(row.get("id")) in wanted]
    if not corpus:
        raise PC.CampaignGateError("performance selection contains zero kernels")
    names = [str(row.get("id") or "") for row in corpus]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise PC.CampaignGateError("performance corpus has missing or duplicate kernel ids")
    return corpus


#: What one kernel's cert tier is expected to cost, and the reason when it cannot be priced.
_CERT_PLAN: "dict[str, tuple[bool, str]]" = {}


def _kernel_output_elements(kernel: Mapping) -> int | None:
    """The kernel's output element count, which is what the cert cost is measured against."""
    for key in ("output_elements", "out_elements"):
        value = kernel.get(key)
        if isinstance(value, int) and value > 0:
            return value
    m, n = kernel.get("m"), kernel.get("n")
    if isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0:
        return m * n
    return None


def plan_cert_tier(corpus: "list[dict]", *, target: str,
                   budget_s: float | None) -> "dict[str, tuple[bool, str]]":
    """Decide which kernels earn the CERT tier, priced from what certification actually cost.

    THE LOOP TIER IS NOT THE QUESTION -- every kernel runs there. The expensive cycle-accurate tier
    is, and it used to be chosen by a string a generator wrote down: ``sim_hint`` set from the
    constant ``"L2+L3" if macs <= 2_000_000 else "L2_only"``, with the paired bench defaulting an
    unlabelled kernel to ``"L2+L3"``. So an unpriced kernel took the MOST expensive path by default,
    which is the wrong direction to fail in when a single deep member can cost more than the rest of
    the corpus put together.

    The whole point of the analytical tooling is that most performance work does NOT need the cert
    tier: the loop tier plus a derived model orders candidates, and certification is spent on a
    representative few. So the tier is DERIVED here from :mod:`merlin.targetgen.cert_cost`, whose fit
    is measured on this target's own certified runs and which refuses rather than guessing.

    FAIL CHEAP, NOT EXPENSIVE. A kernel that cannot be priced -- no fit for this target, or no
    declared output shape -- is held at the loop tier and the reason is recorded. That is the
    opposite of the old default and the only safe direction: a wrongly-cheap plan under-certifies
    and says so, a wrongly-expensive one silently spends the budget the corpus needed.
    """
    from merlin.targetgen import cert_cost                              # noqa: PLC0415

    try:
        fit = cert_cost.fit_for(target)
    except Exception:  # noqa: BLE001 - an unreadable history prices nothing; it is not an error here
        fit = None

    priced: list[tuple[float, str]] = []
    plan: dict[str, tuple[bool, str]] = {}
    for kernel in corpus:
        kid = str(kernel.get("id") or "")
        ceiling = str(kernel.get("max_oracle_tier") or "").strip()
        if ceiling and ceiling != _CERT_TIER:
            plan[kid] = (False, f"the kernel caps its oracle tier at {ceiling!r}")
            continue
        elements = _kernel_output_elements(kernel)
        if fit is None:
            plan[kid] = (False, f"no certified run on {target!r} prices the cert tier yet")
            continue
        if elements is None:
            plan[kid] = (False, "the kernel declares no output shape, so its cert cost is UNKNOWN")
            continue
        seconds = cert_cost.predict_seconds(fit, elements)
        if seconds is None:
            plan[kid] = (False, f"the cost fit cannot price {elements} output element(s)")
            continue
        priced.append((float(seconds), kid))

    if budget_s is None:
        for _, kid in priced:
            plan[kid] = (True, "no cert budget declared, so every priced kernel is certified")
        return plan
    # Cheapest first: the most cert cover the declared budget buys.
    spent = 0.0
    for seconds, kid in sorted(priced):
        if spent + seconds <= float(budget_s):
            spent += seconds
            plan[kid] = (True, f"predicted {seconds:.0f}s fits the remaining cert budget")
        else:
            plan[kid] = (False, f"predicted {seconds:.0f}s exceeds the remaining cert budget "
                                f"({float(budget_s) - spent:.0f}s of {float(budget_s):.0f}s left)")
    return plan


def _sims_for(kernel: dict, requested: str) -> tuple[str, ...]:
    if requested == "auto":
        kid = str(kernel.get("id") or "")
        admitted, _why = _CERT_PLAN.get(kid, (False, "no cert plan was computed for this kernel"))
        return (_LOOP_SIM, _CERT_SIM) if admitted else (_LOOP_SIM,)
    sims = tuple(value.strip() for value in requested.split(",") if value.strip())
    if not sims or len(sims) != len(set(sims)) or any(s not in (_LOOP_SIM, _CERT_SIM) for s in sims):
        raise PC.CampaignGateError("--sims must be auto, spike, or a unique spike,verilator list")
    return sims


def _expected_cells(corpus: list[dict], requested: str) -> tuple[PC.PerfCell, ...]:
    """Expand the fixed profiling corpus into the exact identities its completion gate expects."""
    return tuple(
        PC.PerfCell(_FIXED_PROFILE_FAMILY, str(kernel["id"]), simulator,
                    _FIXED_PROFILE_REPLICATE)
        for kernel in corpus
        for simulator in _sims_for(kernel, requested)
    )


def _completion_rows(capsule: str, arm: dict, sims: tuple[str, ...]) -> list[dict]:
    """Project one legacy profiler record into exact, simulator-specific completion evidence."""
    rows: list[dict] = []
    per_sim = arm.get("per_sim") or {}
    for simulator in sims:
        result = per_sim.get(simulator)
        if not isinstance(result, dict):
            continue
        rows.append({
            "family": _FIXED_PROFILE_FAMILY,
            "capsule": capsule,
            "simulator": simulator,
            "replicate": _FIXED_PROFILE_REPLICATE,
            "correct": result.get("correct"),
            "cycles": None if simulator == "spike" else result.get("cycles"),
            "provenance": result.get("provenance"),
        })
    return rows


def run_arm4(package: Path, kernel: dict, kernel_dir: Path, sims: tuple[str, ...],
             capsule_runs: Path, timeout: int, target: str, *,
             measurement_pass: str | None = None,
             expected_package_sha256: str | None = None,
             rtl_identity: Mapping | None = None) -> dict:
    """Run one kernel through the frozen Arm-4 package; entrypoints are boxed by the caller."""
    result = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    package_before = hash_tree(package)["sha256"]
    inputs_before = hash_tree(kernel_dir)["sha256"]
    capsule = CR.load_capsule(kernel_dir, contract=_CONTRACT)
    capsule = dict(capsule)
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2"] + (
        ["L3"] if "verilator" in sims else [])
    adapters = CR.default_adapters()
    if "verilator" not in sims:
        adapters = {tier: adapter for tier, adapter in adapters.items() if tier != "L3"}
    try:
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(capsule_runs),
            run_id=(f"arm4_{kernel['id']}_{measurement_pass}" if measurement_pass
                    else f"arm4_{kernel['id']}"),
            contract=_CONTRACT,
            oracle_adapters=adapters,
            timeout=timeout,
            target=target,
            workers=1,
        )
    except Exception as exc:  # one failed cell is recorded; the global completion gate still refuses
        result.update({"ok_build": False, "status": "error",
                       "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                       "traceback": traceback.format_exc()[-1600:]})
        return result
    result["status"] = grade.get("status")
    numeric = grade.get("numeric")
    result["numeric"] = numeric.get("status") if isinstance(numeric, dict) else numeric
    work_volume = grade.get("work_volume") if isinstance(grade.get("work_volume"), dict) else {}
    result["work_volume"] = work_volume
    command_artifact = grade.get("command_buffer_artifact")
    if isinstance(command_artifact, Mapping):
        result["command_buffer_artifact"] = dict(command_artifact)
    rtl_facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_facts_sha256 = rtl_facts.get("sha256") if isinstance(rtl_facts, Mapping) else None
    circt_core = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    if _is_sha256(rtl_facts_sha256):
        result["rtl_facts_sha256"] = rtl_facts_sha256
    if isinstance(circt_core, Mapping) and _is_sha256(circt_core.get("sha256")):
        result["circt_core_hw"] = dict(circt_core)
    identity, identity_refusals = _measurement_identity(
        package_before=package_before,
        package_after=hash_tree(package)["sha256"],
        inputs_before=inputs_before,
        inputs_after=hash_tree(kernel_dir)["sha256"],
        work_volume=work_volume,
        toolchain_shas=grade.get("toolchain_shas"),
        target=target,
        expected_package_sha256=expected_package_sha256,
        rtl_facts_sha256=rtl_facts_sha256,
    )
    result["measurement_identity"] = identity
    result["measurement_identity_refusals"] = identity_refusals
    tiers = grade.get("tiers") or {}
    for sim, tier in (("spike", "L2"), ("verilator", "L3")):
        if sim not in sims:
            continue
        tier_result = tiers.get(tier) or {}
        status = tier_result.get("status") if isinstance(tier_result, dict) else tier_result
        cycles = tier_result.get("cycles") if isinstance(tier_result, dict) else None
        is_rtl_measurement = (sim != "spike" and isinstance(tier_result, dict)
                              and tier_result.get("derived_from_rtl") is True
                              and tier_result.get("cycle_accurate") is True)
        admitted_cycles = cycles if is_rtl_measurement else None
        exact_macs = work_volume.get("exact_macs")
        achieved = (exact_macs / admitted_cycles
                    if isinstance(exact_macs, int) and isinstance(admitted_cycles, int)
                    and admitted_cycles > 0 else None)
        result["per_sim"][sim] = {
            "cycles": admitted_cycles,
            "correctness_cycles": cycles if sim == "spike" else None,
            "tier_status": status,
            "correct": status == "pass",
            "achieved_macs_per_cycle": achieved,
            "work_volume": work_volume,
            "provenance": {
                "tier": tier,
                "simulator": sim,
                "derived_from_rtl": tier_result.get("derived_from_rtl") is True,
                "cycle_accurate": tier_result.get("cycle_accurate") is True,
                "evidence": tier_result.get("evidence"),
            } if isinstance(tier_result, dict) else None,
            "counters": tier_result.get("counters") if isinstance(tier_result, dict) else None,
            "timing_observations": (
                tier_result.get("timing_observations") if isinstance(tier_result, dict) else None),
            "timing_capability": (
                tier_result.get("timing_capability") if isinstance(tier_result, dict) else None),
            "measurement_conditions": (
                tier_result.get("measurement_conditions")
                if isinstance(tier_result, dict) else None),
            "utilization": tier_result.get("utilization") if isinstance(tier_result, dict) else None,
        }
        if sim != "spike" and _is_sha256(rtl_facts_sha256):
            result["per_sim"][sim]["rtl_facts_sha256"] = rtl_facts_sha256
    if grade.get("failure"):
        result["failure"] = {key: grade["failure"].get(key)
                             for key in ("plane", "category", "detail")}
    return result


def _write_json(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-run-id", required=True,
                        help="exact completed Arm-4 functional run directory name")
    parser.add_argument("--functional-submission-sha256", required=True,
                        help="exact frozen functional submission SHA-256")
    parser.add_argument("--waive-functional-gate", action="append", default=[], metavar="PREDICATE",
                        help="accept a NAMED completeness gap in the functional baseline instead of "
                             "refusing (repeatable). Integrity predicates -- sandbox, answer mask, "
                             "answer-access audit, cohort-admission accounting, public/hidden "
                             "identity separation -- cannot be waived and asking is an error. Every "
                             "accepted waiver is recorded in the campaign record and every result it "
                             "produces is marked functional_gate_clean=false.")
    parser.add_argument("--rtl-facts",
                        help="exact CIRCT-extracted RTL facts JSON for performance provenance")
    parser.add_argument("--kernels", default="all")
    parser.add_argument("--approach", choices=("arm4",), default="arm4",
                        help="only the Arm-4 compiler lane is admitted in this campaign")
    parser.add_argument("--cert-budget-seconds", type=float, default=None,
                        help="seconds of cycle-accurate certification this run may spend; kernels "
                             "are admitted cheapest-first from the measured cost fit and the rest "
                             "stay at the loop tier with a recorded reason (default: no budget cap)")
    parser.add_argument("--sims", default="auto",
                        help="auto (per-kernel hint), spike, verilator, or spike,verilator")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-id", default="perf_0001")
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=True,
                        help="instrument cycle windows with a counter set sized from elaborated RTL")
    parser.add_argument("--counter-unit",
                        help="byte-counter unit family for pass two (default: BYTES from target header)")
    args = parser.parse_args(argv)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise PC.CampaignGateError("performance run id must be a simple directory name")
    if args.timeout <= 0:
        raise PC.CampaignGateError("performance cell timeout must be positive")
    if args.counter_unit is not None:
        unit = str(args.counter_unit).strip()
        if (not args.hardware_counters or not unit
                or any(not (char.isalnum() or char == "_") for char in unit)):
            raise PC.CampaignGateError(
                "--counter-unit requires hardware counters and must be one identifier token")
        if unit.upper() != _PHYSICAL_BYTE_UNIT:
            raise PC.CampaignGateError(
                "the linked physical-byte pass requires the BYTES unit declared by the target header")
    physical_unit = str(args.counter_unit).upper() if args.counter_unit else _PHYSICAL_BYTE_UNIT
    if not args.rtl_facts:
        raise PC.CampaignGateError("--rtl-facts is required for content-linked RTL performance")
    rtl_identity = _load_rtl_identity(Path(args.rtl_facts), PB.TARGET)
    counter_binding = _probe_counter_byte_bindings(rtl_identity)

    functional = PC.inspect_functional_run(
        _FUNCTIONAL_RUNS, args.functional_run_id, args.functional_submission_sha256,
        waive=frozenset(args.waive_functional_gate or ()))
    _selected_corpus(args.kernels)  # validate the requested IDs before allocating the fresh run dir
    out_dir = PB.RUNS / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(
            f"performance run directory already exists; choose a fresh --run-id: {out_dir}")

    snapshot = PC.materialize_perf_workspace(functional, out_dir / "_frozen_functional")
    workload_root = out_dir / "_frozen_workload" / "kernels"
    workload_digest = PC.materialize_readonly_tree(PB.KERNELS, workload_root)
    corpus = _selected_corpus(args.kernels, workload_root)
    # Ration the cert tier BEFORE expanding cells, so the expected-cell set and the run agree on
    # which kernels were certified and every hold-back carries its reason into the record.
    _CERT_PLAN.clear()
    _CERT_PLAN.update(plan_cert_tier(corpus, target=PB.TARGET,
                                     budget_s=args.cert_budget_seconds))
    for _kid, (_admitted, _why) in sorted(_CERT_PLAN.items()):
        print(f"[tier] {_kid:34} {'CERT' if _admitted else 'loop-only'}: {_why}")
    sims_by_capsule = {str(kernel["id"]): _sims_for(kernel, args.sims) for kernel in corpus}
    expected = _expected_cells(corpus, args.sims)
    fork = PC.functional_fork(functional)
    before = PC.check_fork(fork, snapshot)
    if before.ok is not True:
        raise PC.CampaignGateError(f"functional fork does not hold before performance: {before.reason}")
    fork_record = fork.to_dict()
    fork_record.update({"functional_run_id": functional.run_id,
                        "functional_submission_sha256": functional.digest,
                        "copied_submission": str(snapshot)})
    _write_json(out_dir / "functional_fork.json", fork_record)

    target_experiment = load_target_experiment(_DESCRIPTOR)
    probe_workspace = out_dir / "_probe_workspace"
    probe_workspace.mkdir()
    probe_policy = PC.package_sandbox_policy(target_experiment, probe_workspace, snapshot)
    campaign = {
        "status": "NO_GO",
        "approach": args.approach,
        "functional_run_id": functional.run_id,
        "functional_submission_sha256": functional.digest,
        "functional_public_capsules": functional.public_capsules,
        "functional_hidden_capsules": functional.hidden_capsules,
        # A campaign launched over a waived gate is still a real measurement, but it is NOT the same
        # claim as one whose baseline was fully established. Both facts ride in the record so a reader
        # never has to reconstruct which it was: `false` here means the numbers below are conditional
        # on the named gaps, and any write-up must say so.
        "functional_gate_clean": functional.gate_clean,
        "functional_gate_deviations": [d.to_dict() for d in functional.deviations],
        "snapshot": str(snapshot),
        "snapshot_sha256": functional.digest,
        "workload_snapshot": str(workload_root),
        "workload_sha256": workload_digest,
        "instrumentation": {
            "hardware_counters": args.hardware_counters,
            "mode": "linked_multi_pass" if args.hardware_counters else "disabled",
            "applies_to": "verilator_cells" if args.hardware_counters else None,
            "passes": ([{"id": "occupancy", "selection": "joint_occupancy"},
                        {"id": "physical_bytes", "unit_family": physical_unit,
                         "semantic_resolution": "raw_named_readings_only"}]
                       if args.hardware_counters else []),
            "capacity_source": "elaborated CIRCT HW",
            "rtl_identity": rtl_identity,
            "counter_byte_binding": counter_binding,
        },
        "expected_cells": [
            {"family": cell.family, "capsule": cell.capsule, "simulator": cell.simulator,
             "replicate": cell.replicate}
            for cell in expected
        ],
        "fork_before": before.to_dict(),
        "fork_after": None,
        "sandbox": {
            "engine": "bwrap",
            "network": "unshared",
            "package_read_only": True,
            "answer_surface_coverage_gap": list(probe_policy.coverage_gap),
            "required_tool_probes": [probe.label for probe in probe_policy.required_tools],
            "tool_probe_results": [],
        },
        "completion": PC.completion_report([], expected),
        "refusal": "campaign has not completed",
    }
    _write_json(out_dir / "campaign_manifest.json", campaign)

    results: list[dict] = []
    completion_rows: list[dict] = []
    refusal: str | None = None
    try:
        campaign["sandbox"]["tool_probe_results"] = PC.run_tool_probes(probe_policy)
        _write_json(out_dir / "campaign_manifest.json", campaign)
        cells_root = out_dir / "_cell_workspaces"
        cells_root.mkdir()
        for kernel in corpus:
            name = str(kernel["id"])
            sims = sims_by_capsule[name]
            shape = kernel.get("shape") or (
                f"{kernel.get('M')}x{kernel.get('K')}x{kernel.get('N')}"
                if kernel.get("M") is not None else "?")
            print(f"\n=== Arm-4 kernel {name} ({shape}, sims={list(sims)}) ===", flush=True)
            # Each pass gets a fresh writable mount. The package cannot inspect oracle/result files
            # from an earlier pass or cell; capsule_runner copies this cell's interface MLIR into
            # generated/ before the first boxed entrypoint and keeps the source corpus outside the mount.
            cell_workspace = cells_root / name
            cell_workspace.mkdir()

            def run_one(pass_name: str) -> dict:
                pass_workspace = cell_workspace / pass_name
                pass_workspace.mkdir()
                capsule_runs = pass_workspace / "capsule_runs"
                capsule_runs.mkdir()
                cell_policy = PC.package_sandbox_policy(
                    target_experiment, pass_workspace, snapshot)
                with PC.boxed_entrypoints(cell_policy):
                    return run_arm4(
                        snapshot, kernel, workload_root / name, sims, capsule_runs,
                        args.timeout, target_experiment.target, measurement_pass=pass_name,
                        expected_package_sha256=functional.digest,
                        rtl_identity=rtl_identity)

            if args.hardware_counters and "verilator" in sims:
                arm = _collect_linked_counter_passes(
                    run_one, physical_unit=physical_unit, counter_binding=counter_binding,
                    rtl_facts_sha256=rtl_identity["rtl_facts"]["sha256"])
            else:
                with _counter_environment(enabled=False):
                    arm = run_one("unprofiled")
            cell = {"kernel": name, "shape": shape, "work_volume": arm.get("work_volume"),
                    "command_buffer_artifact": arm.get("command_buffer_artifact"),
                    "resource_bindings": _resource_bindings(arm),
                    "output_dtype": kernel.get("output_dtype", ""),
                    "source": kernel.get("source"), "sim_hint": kernel.get("sim_hint"),
                    "approaches": {"arm4": arm}}
            results.append(cell)
            completion_rows.extend(_completion_rows(name, arm, sims))
            _write_json(out_dir / f"{name}.json", cell)
            _write_json(out_dir / "completion_cells.json", completion_rows)
            campaign["completion"] = PC.completion_report(completion_rows, expected)
            _write_json(out_dir / "campaign_manifest.json", campaign)
            linked = arm.get("linked_counter_evidence")
            if (args.hardware_counters and "verilator" in sims
                    and (not isinstance(linked, dict) or linked.get("status") != "linked")):
                reasons = linked.get("refusals") if isinstance(linked, dict) else ["missing linkage"]
                raise PC.CampaignGateError(
                    f"{name} counter passes could not be linked: {reasons}")
            summary = {sim: (row.get("cycles"), row.get("correct"))
                       for sim, row in arm.get("per_sim", {}).items()}
            print(f"  [arm4] {summary}", flush=True)
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    finally:
        _write_json(out_dir / "perf_results.json", results)
        auxiliary = _roofline_auxiliary_requirements(results, rtl_identity)
        _write_json(out_dir / "roofline_auxiliary_evidence.json", auxiliary)
        campaign["roofline_evidence"] = auxiliary
        _write_json(out_dir / "completion_cells.json", completion_rows)
        after = PC.check_fork(fork, snapshot)
        campaign["fork_after"] = after.to_dict()
        if after.ok is not True:
            refusal = f"functional fork changed during performance: {after.reason}"
        try:
            campaign["completion"] = PC.completion_report(completion_rows, expected)
            if refusal is None and not campaign["completion"]["complete"]:
                counts = campaign["completion"]
                refusal = (f"Arm-4 performance reported {counts['reported']} of "
                           f"{counts['expected']} expected cells; {counts['failed']} reported "
                           "cell(s) failed simulator-specific completion evidence")
        except PC.CampaignGateError as exc:
            if refusal is None:
                refusal = str(exc)
        campaign["refusal"] = refusal
        campaign["status"] = "GO" if refusal is None else "NO_GO"
        _write_json(out_dir / "campaign_manifest.json", campaign)

    if refusal is not None:
        print(f"\nNO-GO: {refusal}\nmanifest: {out_dir / 'campaign_manifest.json'}", flush=True)
        return 2
    print(f"\nGO: completed {campaign['completion']['expected']} Arm-4 cells; "
          f"manifest: {out_dir / 'campaign_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

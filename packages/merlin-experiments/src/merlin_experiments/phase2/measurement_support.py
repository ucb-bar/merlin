"""Exact measurement identities, counter-pass evidence and resource coverage."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path

from merlin.perf.dma_volume import physical_volume_from_counters
from merlin.perf.work_volume import work_from_command_buffer
from merlin.targetgen.rtl import mlc_bridge

from . import campaign as PC

_COUNTER_ENV = ("MERLIN_HW_COUNTERS", "MERLIN_HW_COUNTER_UNIT")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_rtl_identity(path: Path, target: str) -> dict:
    """Bind this run to exact extractor JSON and the elaborated CIRCT it names."""
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PC.CampaignGateError(f"cannot read exact RTL facts {path}: {exc}") from exc
    inputs = document.get("inputs") if isinstance(document, Mapping) else None
    recorded = inputs.get("core_hw_sha256") if isinstance(inputs, Mapping) else None
    circt_path = mlc_bridge.core_hw_mlir(target)
    if not is_sha256(recorded) or circt_path is None or not Path(circt_path).is_file():
        raise PC.CampaignGateError("RTL facts do not identify one available elaborated CIRCT input by full SHA-256")
    circt_payload = Path(circt_path).read_bytes()
    actual = hashlib.sha256(circt_payload).hexdigest()
    if recorded != actual:
        raise PC.CampaignGateError("RTL facts core_hw_sha256 does not match the active elaborated CIRCT bytes")
    return {
        "rtl_facts": {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()},
        "circt_core_hw": {"path": str(Path(circt_path).resolve()), "sha256": actual},
    }


@contextmanager
def counter_environment(*, enabled: bool, unit: str | None = None):
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


def measurement_identity(
    *,
    package_before: str,
    package_after: str,
    inputs_before: str,
    inputs_after: str,
    work_volume: Mapping,
    toolchain_shas: object,
    target: str,
    expected_package_sha256: str | None,
    rtl_facts_sha256: str | None = None,
) -> tuple[dict, list[str]]:
    """Build independently observed pass identity; return every reason it is not exact."""
    refusals: list[str] = []
    program_sha256 = work_volume.get("artifact_sha256")
    if not is_sha256(program_sha256):
        refusals.append("graded command buffer has no exact SHA-256 identity")
    for label, value in (
        ("submission before pass", package_before),
        ("submission after pass", package_after),
        ("capsule inputs before pass", inputs_before),
        ("capsule inputs after pass", inputs_after),
    ):
        if not is_sha256(value):
            refusals.append(f"{label} has no exact SHA-256 identity")
    if package_before != package_after:
        refusals.append("frozen submission changed during the counter pass")
    if inputs_before != inputs_after:
        refusals.append("frozen capsule inputs changed during the counter pass")
    if expected_package_sha256 is not None and package_before != expected_package_sha256:
        refusals.append("counter pass did not execute the certified functional submission")
    if not is_sha256(rtl_facts_sha256):
        refusals.append("counter pass is not bound to exact RTL-facts bytes")

    revisions: dict[str, str] = {}
    if not isinstance(toolchain_shas, Mapping) or not toolchain_shas:
        refusals.append("grade has no exact toolchain revision map")
    else:
        for raw_name, raw_revision in toolchain_shas.items():
            if not isinstance(raw_name, str) or not raw_name:
                refusals.append("toolchain revision map contains an invalid component name")
                continue
            if not isinstance(raw_revision, str) or not raw_revision or raw_revision.strip().upper() == "UNKNOWN":
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
    return set(raw_names) if all(isinstance(name, str) and name for name in raw_names) else None


def _copy_mapping(value: object) -> dict | None:
    return dict(value) if isinstance(value, Mapping) else None


def _admissible_counter_facts(
    binding: object, readings: object, *, rtl_facts_sha256: str | None
) -> tuple[list[dict], str]:
    """Return only a complete, proved byte binding; never promote structural candidates."""
    if not isinstance(binding, Mapping):
        return [], "no counter-byte binding probe was supplied"
    facts = binding.get("counter_facts")
    if binding.get("status") not in ("exact", "proved", "resolved") or not isinstance(facts, list) or not facts:
        return [], str(binding.get("why") or "counter-byte semantics remain UNKNOWN")
    if not isinstance(readings, Mapping) or not is_sha256(rtl_facts_sha256):
        return [], "counter readings or exact RTL-facts identity are absent"
    if binding.get("rtl_facts_sha256") != rtl_facts_sha256 or any(
        not isinstance(fact, Mapping)
        or fact.get("fact_kind") != "counter_byte_binding"
        or fact.get("artifact_sha256") != rtl_facts_sha256
        or fact.get("derived_from_rtl") is not True
        or not fact.get("provenance")
        for fact in facts
    ):
        return [], "counter-byte facts are not proved from the exact RTL-facts artifact"
    fields = [fact.get("counter_field") for fact in facts]
    if (
        not all(isinstance(field, str) and field for field in fields)
        or len(set(fields)) != len(fields)
        or set(fields) != set(readings)
    ):
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
    if (
        not isinstance(program, Mapping)
        or program.get("kind") != "compiler_command_buffer"
        or not is_sha256(program.get("sha256"))
    ):
        refusals.append("program identity is not an exact compiler-command-buffer SHA-256")
    inputs = identity.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("kind") != "frozen_capsule_tree"
        or not is_sha256(inputs.get("sha256"))
    ):
        refusals.append("input identity is not an exact frozen-capsule-tree SHA-256")
    toolchain = identity.get("toolchain")
    if not isinstance(toolchain, Mapping):
        refusals.append("toolchain identity is not a mapping")
    else:
        if not isinstance(toolchain.get("target"), str) or not toolchain.get("target"):
            refusals.append("toolchain identity has no target")
        if not is_sha256(toolchain.get("frozen_submission_sha256")):
            refusals.append("toolchain identity has no exact frozen-submission SHA-256")
        revisions = toolchain.get("recorded_revisions")
        if not isinstance(revisions, Mapping) or not revisions:
            refusals.append("toolchain identity has no recorded revision map")
        elif any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            or value.strip().upper() == "UNKNOWN"
            for name, value in revisions.items()
        ):
            refusals.append("toolchain identity contains an unknown or malformed revision")
    return refusals


def link_counter_passes(
    occupancy_pass: Mapping,
    byte_pass: Mapping,
    *,
    physical_unit: str,
    counter_binding: object = None,
    rtl_facts_sha256: str | None = None,
) -> dict:
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
    for label, identity in (("occupancy", occupancy_identity), ("physical-byte", byte_identity)):
        refusals.extend(f"{label} pass: {reason}" for reason in _linked_identity_refusals(identity))
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
        if rtl.get("correct") is not True or not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
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
        if (
            not isinstance(selection, Mapping)
            or selection.get("kind") != expected_kind
            or selection.get("unit") != expected_unit
        ):
            refusals.append(f"{label} pass reports the wrong counter selection")
        names = _selected_counter_names(report, occupancy=index == 0)
        readings = report.get("readings")
        if not isinstance(readings, Mapping) or not readings:
            refusals.append(f"{label} pass has no raw named counter readings")
        elif names is None:
            refusals.append(f"{label} pass has no exact selected-counter set")
        elif set(readings) != names:
            refusals.append(f"{label} pass did not report every and only selected counter")
        elif any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in readings.values()):
            refusals.append(f"{label} pass has a non-integer or negative raw counter reading")

    if reports[0] is not None and rtl_rows:
        occupancy = reports[0].get("occupancy")
        combinations = occupancy.get("by_combination") if isinstance(occupancy, Mapping) else None
        readings = reports[0].get("readings")
        selected = (
            set(combinations.values())
            if isinstance(combinations, Mapping)
            and all(isinstance(value, str) and value for value in combinations.values())
            else set()
        )
        if (
            not selected
            or not isinstance(readings, Mapping)
            or set(readings) != selected
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in readings.values())
            or sum(readings.values()) > rtl_rows[0].get("cycles", -1)
        ):
            refusals.append("occupancy readings do not fit their own RTL cycle window")

    counter_schema_sha256 = None
    counter_capacity = None
    if all(report is not None for report in reports):
        schemas = [report.get("measured_header_sha256") for report in reports]
        discovered = [report.get("discovery") for report in reports]
        discovered_shas = [item.get("header_sha256") if isinstance(item, Mapping) else None for item in discovered]
        if (
            not all(isinstance(item, Mapping) and item.get("status") == "derived" for item in discovered)
            or not all(is_sha256(value) for value in schemas + discovered_shas)
            or len(set(schemas + discovered_shas)) != 1
        ):
            refusals.append("counter passes do not share one exact measured header identity")
        else:
            counter_schema_sha256 = schemas[0]
        capacities = [report.get("capacity") for report in reports]
        if (
            not all(
                isinstance(value, Mapping)
                and value.get("status") == "derived"
                and isinstance(value.get("slots"), int)
                and not isinstance(value.get("slots"), bool)
                and value.get("slots") > 0
                and isinstance(value.get("provenance"), Mapping)
                and is_sha256(value["provenance"].get("sha256"))
                for value in capacities
            )
            or capacities[0] != capacities[1]
        ):
            refusals.append("counter passes do not share one exact derived capacity receipt")
        else:
            counter_capacity = dict(capacities[0])

    physical_evidence = {
        "unit_family": physical_unit,
        "semantic_resolution": "raw_named_readings_only",
        "selected_counters": (_copy_mapping(reports[1].get("selected_counters")) if reports[1] is not None else None),
        "readings": (_copy_mapping(reports[1].get("readings")) if reports[1] is not None else None),
    }
    facts, binding_status = _admissible_counter_facts(
        counter_binding, physical_evidence["readings"], rtl_facts_sha256=rtl_facts_sha256
    )
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
        "rtl_facts_sha256": rtl_facts_sha256 if is_sha256(rtl_facts_sha256) else None,
        "cycle_windows": {
            "occupancy": rtl_rows[0].get("cycles") if len(rtl_rows) == 2 else None,
            "physical_bytes": rtl_rows[1].get("cycles") if len(rtl_rows) == 2 else None,
            "instrumentation_delta": (
                rtl_rows[1].get("cycles") - rtl_rows[0].get("cycles")
                if len(rtl_rows) == 2
                and isinstance(rtl_rows[0].get("cycles"), int)
                and not isinstance(rtl_rows[0].get("cycles"), bool)
                and isinstance(rtl_rows[1].get("cycles"), int)
                and not isinstance(rtl_rows[1].get("cycles"), bool)
                else None
            ),
        },
        "physical_byte_counters": physical_evidence,
    }


def collect_linked_counter_passes(
    run_one: Callable[[str], dict],
    *,
    physical_unit: str,
    counter_binding: object = None,
    rtl_facts_sha256: str | None = None,
) -> dict:
    """Execute occupancy and byte-family passes under disjoint instrumentation environments."""
    with counter_environment(enabled=True, unit=None):
        occupancy = run_one("occupancy")
    with counter_environment(enabled=True, unit=physical_unit):
        physical_bytes = run_one("physical_bytes")
    linked = link_counter_passes(
        occupancy,
        physical_bytes,
        physical_unit=physical_unit,
        counter_binding=counter_binding,
        rtl_facts_sha256=rtl_facts_sha256,
    )
    result = dict(occupancy)
    result["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical_bytes}
    result["linked_counter_evidence"] = linked
    return result


def resource_bindings(measurement: Mapping) -> dict[str, dict]:
    """Name only resource axes established by the artifacts actually carried by this run."""
    out: dict[str, dict] = {}
    work = measurement.get("work_volume")
    command_artifact = measurement.get("command_buffer_artifact")
    command = command_artifact.get("command_buffer") if isinstance(command_artifact, Mapping) else None
    derived_work = work_from_command_buffer(command) if isinstance(command, Mapping) else None
    if (
        isinstance(work, Mapping)
        and derived_work is not None
        and isinstance(work.get("exact_macs"), int)
        and not isinstance(work.get("exact_macs"), bool)
        and work.get("exact_macs") > 0
        and derived_work.exact_macs == work.get("exact_macs")
        and canonical_sha256(command) == work.get("artifact_sha256")
        and command_artifact.get("artifact_sha256") == work.get("artifact_sha256")
        and isinstance(work.get("basis"), str)
        and work.get("basis")
        and isinstance(work.get("unit"), str)
        and work.get("unit")
    ):
        out["compute"] = {
            "resource": f"compute:{work['basis']}:{work['unit']}",
            "derived_from_tool": True,
            "provenance": (
                f"resource axis derived from the exact compiler command-buffer work receipt {work['artifact_sha256']}"
            ),
        }
    linked = measurement.get("linked_counter_evidence")
    physical = linked.get("physical_byte_counters") if isinstance(linked, Mapping) else None
    facts = physical.get("counter_facts") if isinstance(physical, Mapping) else None
    if isinstance(facts, list) and facts and physical.get("semantic_resolution") == "rtl_bound_physical_bytes":
        out["movement"] = {
            "resource": "movement:physical_counters:bytes",
            "derived_from_tool": True,
            "provenance": "resource axis derived from exhaustive RTL-bound physical-byte counters",
        }
    return out


def roofline_auxiliary_requirements(results: list[dict], rtl_identity: Mapping) -> dict:
    """Expose required baselines/probe and fail closed when the runner has no honest path."""
    protocols: set[str] = set()
    profiled: list[str] = []
    raw_composition_probe: dict | None = None
    rtl_facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    circt = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    for cell in results:
        approaches = cell.get("approaches")
        candidates = (
            [value for value in approaches.values() if isinstance(value, Mapping)]
            if isinstance(approaches, Mapping)
            else []
        )
        for measurement in candidates:
            per_sim = measurement.get("per_sim")
            if not isinstance(per_sim, Mapping):
                continue
            for sim_result in per_sim.values():
                provenance = sim_result.get("provenance") if isinstance(sim_result, Mapping) else None
                conditions = sim_result.get("measurement_conditions") if isinstance(sim_result, Mapping) else None
                if (
                    not isinstance(provenance, Mapping)
                    or provenance.get("derived_from_rtl") is not True
                    or provenance.get("cycle_accurate") is not True
                    or not isinstance(conditions, Mapping)
                ):
                    continue
                values = {
                    conditions.get(key)
                    for key in ("measurement_protocol", "cache_protocol")
                    if isinstance(conditions.get(key), str) and conditions.get(key)
                }
                if len(values) == 1:
                    protocols.update(values)
                if (
                    isinstance(measurement.get("linked_counter_evidence"), Mapping)
                    and measurement["linked_counter_evidence"].get("status") == "linked"
                ):
                    profiled.append(str(cell.get("kernel") or ""))
                    linked = measurement["linked_counter_evidence"]
                    occupancy = linked.get("occupancy")
                    overlap = occupancy.get("overlap") if isinstance(occupancy, Mapping) else None
                    proof = overlap.get("partition_proof") if isinstance(overlap, Mapping) else None
                    layout = occupancy.get("occupancy") if isinstance(occupancy, Mapping) else None
                    discovery = occupancy.get("discovery") if isinstance(occupancy, Mapping) else None
                    if (
                        raw_composition_probe is None
                        and isinstance(rtl_facts, Mapping)
                        and isinstance(circt, Mapping)
                        and linked.get("rtl_facts_sha256") == rtl_facts.get("sha256")
                        and isinstance(proof, Mapping)
                        and proof.get("status") == "proved"
                        and proof.get("sha256") == circt.get("sha256")
                        and isinstance(layout, Mapping)
                        and isinstance(discovery, Mapping)
                        and isinstance(occupancy.get("readings"), Mapping)
                    ):
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
    baseline_rows = [
        {
            "measurement_protocol": protocol,
            "required_replicates": 4,
            "status": "UNKNOWN",
            "receipts": [],
            "why": (
                "the performance corpus has no structurally-empty workload emitted by the frozen "
                "compiler; running a hand-authored empty kernel would not measure the same compiler path"
            ),
        }
        for protocol in sorted(protocols)
    ]
    composition_why = (
        "joint occupancy was collected, but its engine ResourceKind mapping is not derived from "
        "RTL/tool evidence; a role inferred from an engine name is forbidden"
        if raw_composition_probe is not None
        else "no joint-occupancy reading was proved against the exact CIRCT and RTL-facts inputs"
    )
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
        f"{row['measurement_protocol']}: four compiler-produced empty RTL baselines are absent" for row in baseline_rows
    )
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


def compute_axis_coverage(cells: list[Mapping]) -> dict:
    """Count the members whose cycles have no counted work behind them, and say why for each.

    NOT A GATE, deliberately. A member whose work `work_volume` cannot count is not necessarily
    defective -- a movement member has no MACs and that is correct behaviour -- so refusing the
    campaign over it would abandon a whole run for a member doing exactly what it was written to do.
    What must not happen is the other thing: an absent compute axis reading as though none applied,
    or as though the work were zero. On a performance bench a zero denominator is not "unknown", it
    is "infinitely fast", and anybody quoting utilization or share-of-achievable needs to know the
    denominator was absent for these members BEFORE they quote it.

    So the run proceeds and the RESULT is loud: a count next to the headline, every unattributed
    member named, and the counter's own per-command refusals carried through verbatim. A bare null
    with no reason attached is exactly what lets a reader assume it was zero.
    """
    attributed: list[str] = []
    unattributed: list[dict] = []
    for index, cell in enumerate(cells):
        # TOTAL BY CONSTRUCTION. This runs inside the campaign's own try/finally, so a raise here
        # would set a refusal and NO-GO the run -- turning the report into the gate it is explicitly
        # not meant to be. A malformed cell is therefore RECORDED as unattributable, never skipped
        # (a skip would shrink the denominator and make the coverage look better than it is).
        if not isinstance(cell, Mapping):
            unattributed.append(
                {
                    "kernel": f"<malformed cell {index}>",
                    "exact_macs": None,
                    "known_macs_lower_bound": None,
                    "reasons": [f"result cell {index} is not a mapping, so this member's work cannot be read at all"],
                }
            )
            continue
        kernel = str(cell.get("kernel") or "")
        bindings = cell.get("resource_bindings")
        compute = bindings.get("compute") if isinstance(bindings, Mapping) else None
        if isinstance(compute, Mapping) and compute.get("resource"):
            attributed.append(kernel)
            continue
        work = cell.get("work_volume") if isinstance(cell.get("work_volume"), Mapping) else {}
        reasons = [str(reason) for reason in (work.get("refusals") or [])]
        if not reasons:
            # The pre-fix shape: no receipt AND no stated reason. Named as its own condition, because
            # "the grader emitted nothing" and "the counter refused this opcode" are different facts
            # and a reader who cannot tell them apart cannot act on either.
            reasons = [
                "the graded result carried no work-volume receipt, so nothing states why "
                "this member's work could not be counted"
            ]
        unattributed.append(
            {
                "kernel": kernel,
                "exact_macs": work.get("exact_macs"),
                "known_macs_lower_bound": work.get("known_macs"),
                "reasons": reasons,
            }
        )
    total = len(attributed) + len(unattributed)
    headline = (
        f"{len(unattributed)} of {total} member(s) carry NO compute axis: their cycles "
        f"cannot be attributed to counted work, so utilization and share-of-achievable "
        f"have no denominator for them"
        if unattributed
        else f"all {total} member(s) carry a compute axis derived from their own command buffer"
    )
    return {
        "schema": "compute_axis_coverage_v1",
        "members": total,
        "with_compute_axis": len(attributed),
        "without_compute_axis": len(unattributed),
        "attributed": sorted(attributed),
        "unattributed": unattributed,
        "gates_the_campaign": False,
        "headline": headline,
    }


def probe_counter_byte_bindings(rtl_identity: Mapping, *, target: str) -> dict:
    """Run the target-owned structural probe and retain UNKNOWN rather than completing semantics."""
    try:
        from merlin.runtime.backends.base import get_backend

        backend = get_backend(target)
        probe = importlib.import_module(f"{backend.__name__}.counter_byte_bindings")
        artifact = probe.probe_counter_byte_bindings()
    except Exception as exc:  # probe failure is evidence unavailability, not a campaign crash
        return {"status": "unknown", "counter_facts": [], "why": f"{type(exc).__name__}: {exc}"}
    circt = rtl_identity.get("circt_core_hw") if isinstance(rtl_identity, Mapping) else None
    probe_inputs = artifact.get("inputs") if isinstance(artifact, Mapping) else None
    probe_circt = probe_inputs.get("circt_core_hw") if isinstance(probe_inputs, Mapping) else None
    if (
        not isinstance(circt, Mapping)
        or not isinstance(probe_circt, Mapping)
        or circt.get("sha256") != probe_circt.get("sha256")
    ):
        return {
            "status": "unknown",
            "counter_facts": [],
            "why": "counter-byte probe did not inspect the exact active CIRCT bytes",
            "probe": artifact,
        }
    return dict(artifact)

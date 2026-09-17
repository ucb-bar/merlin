#!/usr/bin/env python3
"""Build an RTL-derived calibration plan and empirical roofline from raw receipts.

This orchestration edge accepts exact paths only. Caller-authored aggregate peaks, fixed terms,
composition operators, work counts, and byte volumes are refused. Rates and intercepts are fitted
from content-addressed samples that exactly cover the RTL-derived calibration plan; work is recovered
from compiler command buffers; physical traffic is recovered from target-counter readings and
RTL-provenance-bearing counter bindings; composition is derived from a complete joint-occupancy
counter partition tied to the input RTL facts digest.  A fitted asymptote is not called a peak: the
reported rate is the highest post-empty-baseline rate actually observed, a falsified empirical
ceiling.  Fit residuals with unproved scope are diagnostic only, never summed as fixed costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from merlin.perf.calibration_plan import CalibrationPlan, build_calibration_plan_from_rtl
from merlin.perf.decompose import ResourceKind, Unavailable, activity_from_busy
from merlin.perf.dma_volume import physical_volume_from_counters
from merlin.perf.envelope import Basis, FixedTerm, Peak, ResourceDemand
from merlin.perf.headroom import Composition, composition_operator
from merlin.perf.hw_counters import OccupancyCounters, eta_from_counters
from merlin.perf.preflight import RateBasis, rate_from_observations
from merlin.perf.roofline import EvidenceReceipt, EmpiricalObservation, empirical_roofline
from merlin.perf.work_volume import work_from_command_buffer


_SCHEMA = "rtl_empirical_roofline_bundle_v1"
_MIN_CALIBRATION_REPLICATES = 4


def _issue(source: str, detail: str, code: str = "INVALID_INPUT") -> dict[str, str]:
    return {"code": code, "source": source, "detail": detail}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, label: str) -> tuple[Any, dict[str, str], list[dict[str, str]]]:
    receipt = {"path": str(path), "sha256": "UNKNOWN"}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return None, receipt, [_issue(label, f"cannot read explicit path {path}: {exc}")]
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        return json.loads(payload), receipt, []
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, receipt, [_issue(label, f"explicit path {path} is not valid UTF-8 JSON: {exc}")]


def _load_circt(path: Path) -> tuple[str, dict[str, str], list[dict[str, str]]]:
    """Read elaborated CIRCT as a first-class, content-addressed input.

    A composition receipt used to carry arbitrary ``hw_text``.  Its surrounding receipt could be
    internally consistent while the text described a different, synthetic design.  CIRCT is instead
    an explicit input whose complete digest must match the digest recorded by the RTL extractor.
    """
    receipt = {"path": str(path), "sha256": "UNKNOWN"}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return "", receipt, [_issue("circt_hw", f"cannot read explicit path {path}: {exc}")]
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        return payload.decode("utf-8"), receipt, []
    except UnicodeDecodeError as exc:
        return "", receipt, [_issue("circt_hw", f"explicit path {path} is not UTF-8 CIRCT: {exc}")]


def _mapping(value: Any, source: str, issues: list[dict[str, str]]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    issues.append(_issue(source, "must be a JSON object"))
    return {}


def _records(value: Any, source: str, issues: list[dict[str, str]]) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    issues.append(_issue(source, "must be a JSON array"))
    return ()


def _text(row: Mapping[str, Any], key: str, source: str,
          issues: list[dict[str, str]]) -> str:
    value = row.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    issues.append(_issue(source, f"requires an explicit nonempty {key!r}"))
    return ""


def _receipt(raw: Any, source: str, issues: list[dict[str, str]]) -> Mapping[str, Any]:
    row = _mapping(raw, source, issues)
    artifact = row.get("artifact")
    declared = row.get("artifact_sha256")
    if not isinstance(artifact, Mapping):
        issues.append(_issue(source, "requires an explicit raw 'artifact' object"))
        return {}
    actual = _canonical_digest(artifact)
    if not isinstance(declared, str) or declared != actual:
        issues.append(_issue(
            source,
            f"artifact_sha256 does not match canonical raw artifact bytes (computed {actual})",
            "RECEIPT_MISMATCH",
        ))
        return {}
    return artifact


def _check_context(artifact: Mapping[str, Any], expected: Mapping[str, Any], source: str,
                   issues: list[dict[str, str]]) -> bool:
    if artifact.get("context") != expected:
        issues.append(_issue(
            source, "raw receipt context does not match its exact plan/workload identity",
            "RECEIPT_CONTEXT_MISMATCH"))
        return False
    return True


def _cycles(raw: Any, source: str, expected_context: Mapping[str, Any],
            issues: list[dict[str, str]]) -> tuple[int | None, str, str]:
    artifact = _receipt(raw, source, issues)
    digest = _canonical_digest(artifact) if artifact else ""
    context_ok = _check_context(artifact, expected_context, source, issues)
    value = artifact.get("cycles")
    provenance = artifact.get("provenance")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        issues.append(_issue(source, "raw cycle artifact requires a positive integer 'cycles'"))
        value = None
    if not isinstance(provenance, str) or not provenance.strip():
        issues.append(_issue(source, "raw cycle artifact requires measurement provenance"))
        provenance = ""
    if not context_ok:
        value = None
    return value, f"sha256:{digest}; {provenance}" if artifact else "", digest


def _work(raw: Any, source: str, expected_context: Mapping[str, Any],
          issues: list[dict[str, str]]) \
        -> tuple[int | None, str, str, str]:
    artifact = _receipt(raw, source, issues)
    digest = _canonical_digest(artifact) if artifact else ""
    context_ok = _check_context(artifact, expected_context, source, issues)
    command_buffer = artifact.get("command_buffer")
    compiler_provenance = artifact.get("compiler_provenance")
    resource = artifact.get("resource")
    if not isinstance(command_buffer, Mapping):
        issues.append(_issue(source, "receipt requires a raw compiler 'command_buffer' object"))
        return None, "", "", digest
    if not isinstance(compiler_provenance, str) or not compiler_provenance.strip():
        issues.append(_issue(source, "command buffer requires compiler provenance"))
    if not isinstance(resource, str) or not resource.strip():
        issues.append(_issue(source, "command-buffer receipt requires an explicit resource identity"))
        resource = ""
    derived = work_from_command_buffer(command_buffer)
    if derived.exact_macs is None or derived.exact_macs <= 0 or not context_ok:
        issues.append(_issue(
            source,
            "compiler command buffer does not yield positive exact MAC work: "
            + "; ".join(derived.refusals),
            "UNKNOWN_WORK",
        ))
        return None, resource, "", digest
    provenance = f"sha256:{digest}; {compiler_provenance}"
    return derived.exact_macs, resource, provenance, digest


def _empty_run_overheads(calibration: Mapping[str, Any], rtl_sha256: str,
                         issues: list[dict[str, str]]) \
        -> dict[str, tuple[int, EvidenceReceipt]]:
    """Measure common process/runner cost separately from resource calibration.

    A one-resource sweep's affine intercept is an elapsed-time intercept.  It cannot identify
    whether its cycles are resource fill or a shared launch/measurement cost; assigning every such
    intercept to an engine and later composing them double-counts the latter.  A structurally empty
    compiler program under the same measured protocol is the only common term this edge admits.
    Four independent runs are required because this term is itself a fitted/reportable quantity.
    """
    grouped: dict[str, list[tuple[int, str]]] = {}
    rows = _records(calibration.get("empty_run_receipts"), "calibration.empty_run_receipts", issues)
    if not rows:
        issues.append(_issue(
            "calibration.empty_run_receipts",
            "requires four raw structurally-empty run receipts per measurement protocol; a "
            "per-resource intercept has unknown shared-vs-local scope",
            "UNRESOLVED_FIXED_SCOPE"))
        return {}
    for index, raw in enumerate(rows):
        source = f"calibration.empty_run_receipts[{index}]"
        row = _mapping(raw, source, issues)
        protocol = _text(row, "measurement_protocol", source, issues)
        context = {"kind": "empty_run", "measurement_protocol": protocol,
                   "rtl_facts_sha256": rtl_sha256}
        cycles, _provenance, cycle_digest = _cycles(
            row.get("cycle_receipt"), f"{source}.cycle_receipt", context, issues)
        command = _receipt(row.get("command_buffer_receipt"),
                           f"{source}.command_buffer_receipt", issues)
        command_ok = _check_context(command, context, f"{source}.command_buffer_receipt", issues)
        command_buffer = command.get("command_buffer")
        compiler_provenance = command.get("compiler_provenance")
        is_empty = (isinstance(command_buffer, Mapping)
                    and isinstance(command_buffer.get("commands"), Sequence)
                    and not isinstance(command_buffer.get("commands"), (str, bytes))
                    and not command_buffer.get("commands"))
        derived = work_from_command_buffer(command_buffer) if isinstance(command_buffer, Mapping) else None
        if (not is_empty or derived is None or derived.exact_macs != 0
                or not isinstance(compiler_provenance, str) or not compiler_provenance.strip()
                or not command_ok):
            issues.append(_issue(
                source, "empty-run baseline requires a compiler receipt whose command sequence is "
                "structurally empty and has exactly zero derived work",
                "UNRESOLVED_FIXED_SCOPE"))
            continue
        if cycles is not None and protocol and cycle_digest:
            grouped.setdefault(protocol, []).append((cycles, cycle_digest))

    out: dict[str, tuple[int, EvidenceReceipt]] = {}
    for protocol, samples in grouped.items():
        source = f"calibration.empty_run_receipts[{protocol}]"
        sample_ids = tuple(digest for _cycles_value, digest in samples)
        values = {cycles for cycles, _digest in samples}
        if (len(samples) < _MIN_CALIBRATION_REPLICATES
                or len(set(sample_ids)) != len(sample_ids) or len(values) != 1):
            issues.append(_issue(
                source, "requires at least four distinct raw empty-run receipts with one exact "
                "common cycle baseline for this protocol",
                "UNRESOLVED_FIXED_SCOPE"))
            continue
        overhead = next(iter(values))
        out[protocol] = (
            overhead,
            EvidenceReceipt(
                artifact_sha256=_canonical_digest({"protocol": protocol, "sample_ids": sample_ids,
                                                   "cycles": overhead}),
                source_kind="calibration_fit", sample_ids=sample_ids),
        )
    return out


def _traffic(raw: Any, source: str, rtl_sha256: str, expected_context: Mapping[str, Any],
             issues: list[dict[str, str]]) -> tuple[int | None, str, str, str]:
    artifact = _receipt(raw, source, issues)
    digest = _canonical_digest(artifact) if artifact else ""
    context_ok = _check_context(artifact, expected_context, source, issues)
    resource = artifact.get("resource")
    if not isinstance(resource, str) or not resource.strip():
        issues.append(_issue(source, "traffic receipt requires an explicit resource identity"))
        resource = ""
    if artifact.get("rtl_facts_sha256") != rtl_sha256:
        issues.append(_issue(
            source, "traffic counter bindings are not tied to the exact RTL facts input digest",
            "PROVENANCE_MISMATCH",
        ))
    readings = artifact.get("readings")
    facts = artifact.get("counter_facts")
    if not isinstance(readings, Mapping):
        issues.append(_issue(source, "traffic receipt requires raw counter 'readings'"))
        return None, resource, "", digest
    if (not isinstance(facts, Sequence) or isinstance(facts, (str, bytes))
            or not all(isinstance(fact, Mapping) for fact in facts)):
        issues.append(_issue(source, "traffic receipt requires explicit counter_facts bindings"))
        return None, resource, "", digest
    if any(fact.get("fact_kind") != "counter_byte_binding"
           or fact.get("artifact_sha256") != rtl_sha256
           or fact.get("derived_from_rtl") is not True or not fact.get("provenance")
           for fact in facts):
        issues.append(_issue(
            source, "every traffic counter binding needs its schema role, the exact RTL digest, "
            "provenance, and derived_from_rtl=true",
            "UNPROVEN_COUNTER_BINDING",
        ))
        return None, resource, "", digest
    physical = physical_volume_from_counters(readings, counter_facts=facts)
    if physical.total_bytes is None or physical.total_bytes <= 0 or not context_ok:
        issues.append(_issue(
            source,
            "physical counter traffic is not positive and exact: " + "; ".join(physical.unresolved),
            "UNKNOWN_TRAFFIC",
        ))
        return None, resource, "", digest
    provenance = f"sha256:{digest}; physical RTL-bound counters"
    return physical.total_bytes, resource, provenance, digest


def _point_key(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _exact_affine(points: Sequence[tuple[int, int]]) -> tuple[Fraction, Fraction] | None:
    """Return an exact positive affine fit, refusing residuals rather than hiding them."""
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return None
    first = points[0]
    other = next(point for point in points[1:] if point[0] != first[0])
    slope = Fraction(other[1] - first[1], other[0] - first[0])
    intercept = Fraction(first[1]) - slope * first[0]
    if slope <= 0 or intercept < 0:
        return None
    if any(Fraction(y) != intercept + slope * x for x, y in points):
        return None
    return slope, intercept


def _derive_calibration(calibration: Mapping[str, Any], plan: CalibrationPlan, rtl_sha256: str,
                        shared_overheads: Mapping[str, tuple[int, EvidenceReceipt]],
                        issues: list[dict[str, str]]) \
        -> tuple[dict[str, Peak], dict[str, int], list[dict[str, Any]],
                 dict[str, EvidenceReceipt]]:
    for forbidden in ("peaks", "fixed_terms", "composition"):
        if forbidden in calibration:
            issues.append(_issue(
                f"calibration.{forbidden}",
                "caller-authored aggregate is refused; supply raw content-addressed receipts",
                "UNTRUSTED_AGGREGATE",
            ))
    raw_samples = _records(calibration.get("samples"), "calibration.samples", issues)
    by_sweep: dict[str, list[tuple[Mapping[str, Any], int, int, str, str, str, str]]] = {}
    preserved: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_samples):
        source = f"calibration.samples[{index}]"
        row = _mapping(raw, source, issues)
        sweep_id = _text(row, "sweep_id", source, issues)
        coordinates = row.get("coordinates")
        if not isinstance(coordinates, Mapping):
            issues.append(_issue(source, "requires explicit plan coordinates"))
            coordinates = {}
        protocol = row.get("measurement_protocol")
        if not isinstance(protocol, str) or not protocol.strip():
            issues.append(_issue(
                source, "requires the runner-observed measurement_protocol used by its cycle receipt",
                "UNRESOLVED_FIXED_SCOPE"))
            protocol = ""
        context = {
            "sweep_id": sweep_id,
            "coordinates": dict(coordinates),
            "measurement_protocol": protocol,
            "rtl_facts_sha256": rtl_sha256,
        }
        cycles, cycle_provenance, cycle_digest = _cycles(
            row.get("cycle_receipt"), f"{source}.cycle_receipt", context, issues)
        kind = row.get("kind")
        if kind == "compute":
            amount, resource, demand_provenance, demand_digest = _work(
                row.get("command_buffer_receipt"), f"{source}.command_buffer_receipt", context,
                issues)
            unit = "macs"
        elif kind == "movement":
            amount, resource, demand_provenance, demand_digest = _traffic(
                row.get("traffic_receipt"), f"{source}.traffic_receipt", rtl_sha256, context,
                issues)
            unit = "bytes"
        else:
            issues.append(_issue(source, "requires explicit kind 'compute' or 'movement'"))
            amount, resource, demand_provenance, demand_digest, unit = None, "", "", "", ""
        sample_id = (_canonical_digest({"cycles": cycle_digest, "demand": demand_digest})
                     if cycle_digest and demand_digest else "")
        preserved.append({
            "sweep_id": sweep_id or "UNKNOWN",
            "coordinates": dict(coordinates),
            "cycles": cycles if cycles is not None else "UNKNOWN",
            "derived_demand": amount if amount is not None else "UNKNOWN",
            "unit": unit or "UNKNOWN",
            "resource": resource or "UNKNOWN",
            "measurement_protocol": protocol or "UNKNOWN",
            "provenance": {"cycles": cycle_provenance, "demand": demand_provenance},
        })
        if sweep_id and cycles is not None and amount is not None and resource and protocol:
            by_sweep.setdefault(sweep_id, []).append(
                (coordinates, amount, cycles, resource, unit, sample_id, protocol))

    peaks: dict[str, Peak] = {}
    # These fitted residuals are retained as diagnostics only.  Without a resource-busy trace they
    # cannot be classified as engine fill rather than shared launch, so they are never FixedTerms.
    residual_intercepts: dict[str, int] = {}
    receipts: dict[str, EvidenceReceipt] = {}
    resources_seen: set[str] = set()
    for sweep in plan.sweeps:
        source = f"calibration.samples[{sweep.sweep_id}]"
        supplied = by_sweep.get(sweep.sweep_id, [])
        expected = Counter(_point_key(point.to_dict()) for point in sweep.points)
        observed = Counter(_point_key(row[0]) for row in supplied)
        if not sweep.ready:
            if supplied:
                issues.append(_issue(source, "samples cannot repair an UNKNOWN/REFUSED RTL plan"))
            continue
        if observed != expected:
            issues.append(_issue(
                source,
                f"does not exactly cover the RTL plan coordinates; missing={list((expected-observed).elements())}, "
                f"unexpected={list((observed-expected).elements())}",
                "PLAN_COVERAGE_MISMATCH",
            ))
            continue
        identities = {(row[3], row[4]) for row in supplied}
        if len(identities) != 1:
            issues.append(_issue(source, "samples do not share one exact resource identity and unit"))
            continue
        resource, unit = next(iter(identities))
        protocols = {row[6] for row in supplied}
        if len(protocols) != 1:
            issues.append(_issue(
                source, "one resource sweep mixes measurement protocols; its intercept has no "
                "single shared-overhead baseline",
                "UNRESOLVED_FIXED_SCOPE"))
            continue
        protocol = next(iter(protocols))
        overhead_record = shared_overheads.get(protocol)
        if overhead_record is None:
            issues.append(_issue(
                source, f"no four-replicate structurally-empty baseline is available for protocol "
                f"{protocol!r}", "UNRESOLVED_FIXED_SCOPE"))
            continue
        shared_overhead, _overhead_receipt = overhead_record
        if resource in resources_seen:
            issues.append(_issue(
                source,
                f"resource {resource!r} is calibrated by multiple sweeps; selecting one is forbidden",
                "AMBIGUOUS_CALIBRATION",
            ))
            peaks.pop(resource, None)
            residual_intercepts.pop(resource, None)
            continue
        resources_seen.add(resource)
        adjusted = [(row[1], row[2] - shared_overhead) for row in supplied]
        if any(cycles <= 0 for _amount, cycles in adjusted):
            issues.append(_issue(
                source, "the shared empty-run baseline is not strictly below every raw calibration "
                "cycle count", "UNRESOLVED_FIXED_SCOPE"))
            continue
        points = adjusted
        fit = rate_from_observations(points, note=f"exact plan-matched sweep {sweep.sweep_id}")
        exact = _exact_affine(points)
        if (fit.basis is not RateBasis.FITTED or len(points) < sweep.fit.required_points
                or exact is None):
            issues.append(_issue(
                source,
                "requires at least the planned fit count on one exact positive affine line; "
                "residuals or an underdetermined intercept are refused",
                "UNRESOLVED_FIT",
            ))
            continue
        slope, intercept = exact
        if intercept.denominator != 1:
            issues.append(_issue(source, "fractional-cycle intercept cannot become a FixedTerm"))
            continue
        provenance = f"{len(points)} content-addressed samples; {fit.note}"
        peak = Peak.observed_ceiling(resource, points, unit=unit, provenance=provenance)
        if not peak.known or peak.is_ceiling is not True:
            issues.append(_issue(source, peak.reason or "calibration rate is not a falsified ceiling",
                                 "UNRESOLVED_FIT"))
            continue
        peaks[resource] = peak
        residual_intercepts[resource] = int(intercept)
        sample_ids = tuple(row[5] for row in supplied)
        receipts[resource] = EvidenceReceipt(
            artifact_sha256=_canonical_digest({
                "sweep_id": sweep.sweep_id,
                "sample_ids": sample_ids,
                "slope": [slope.numerator, slope.denominator],
                "intercept": [intercept.numerator, intercept.denominator],
            }),
            source_kind="calibration_fit",
            sample_ids=sample_ids,
        )
    unexpected_sweeps = sorted(set(by_sweep) - {sweep.sweep_id for sweep in plan.sweeps})
    if unexpected_sweeps:
        issues.append(_issue(
            "calibration.samples", f"unknown sweep identities: {unexpected_sweeps}",
            "PLAN_COVERAGE_MISMATCH",
        ))
    return peaks, residual_intercepts, preserved, receipts


def _composition(calibration: Mapping[str, Any], rtl_sha256: str, *, hw_text: str,
                 circt_hw_sha256: str, circt_binding_ok: bool,
                 issues: list[dict[str, str]]) \
        -> tuple[Composition | None, float | None, str, Any, EvidenceReceipt | None]:
    source = "calibration.composition_receipt"
    artifact = _receipt(calibration.get("composition_receipt"), source, issues)
    if not circt_binding_ok:
        # Do not run a valid-looking proof against a different elaboration and leave its result
        # visible inside a refused report.  A digest mismatch makes composition itself UNKNOWN.
        return None, None, "", {"state": "unknown", "why": "CIRCT input is not bound to RTL facts"}, None
    if (artifact.get("rtl_facts_sha256") != rtl_sha256
            or not isinstance(artifact.get("source"), str) or not artifact.get("source")):
        issues.append(_issue(
            source,
            "requires provenance tying the CIRCT counter artifact to the exact RTL facts digest",
            "UNVERIFIED_COUNTER_PARTITION",
        ))
        return None, None, "", {"state": "unknown"}, None
    layout = artifact.get("counter_layout")
    readings = artifact.get("readings")
    measurement_cycles = artifact.get("cycles")
    if (not isinstance(layout, Mapping) or not isinstance(readings, Mapping)
            or isinstance(measurement_cycles, bool) or not isinstance(measurement_cycles, int)
            or measurement_cycles <= 0):
        issues.append(_issue(
            source, "requires raw counter_layout/readings and a positive measured cycle window"))
        return None, None, "", {"state": "unknown"}, None
    engines = layout.get("engines")
    combinations = layout.get("by_combination")
    kinds_raw = layout.get("kinds")
    if (not isinstance(engines, Sequence) or isinstance(engines, (str, bytes))
            or not all(isinstance(engine, str) and engine for engine in engines)
            or not isinstance(combinations, Mapping) or not isinstance(kinds_raw, Mapping)):
        issues.append(_issue(source, "counter layout lacks explicit engines/combinations/kinds"))
        return None, None, "", {"state": "unknown"}, None
    by_combination: dict[frozenset[str], str] = {}
    for raw_combo, raw_name in combinations.items():
        combo = frozenset(raw_combo.split("+")) if isinstance(raw_combo, str) else frozenset()
        if (not combo or not combo <= set(engines)
                or not isinstance(raw_name, str) or not raw_name):
            issues.append(_issue(source, "counter combination mapping is malformed"))
            return None, None, "", {"state": "unknown"}, None
        by_combination[combo] = raw_name
    counters = OccupancyCounters(
        prefix=str(layout.get("prefix") or ""), engines=tuple(engines),
        by_combination=by_combination,
    )
    if not counters.complete():
        issues.append(_issue(source, "joint occupancy counter layout is incomplete", "UNKNOWN_OVERLAP"))
        return None, None, "", {"state": "unknown", "layout": counters.to_dict()}, None
    try:
        kinds = {engine: ResourceKind(kinds_raw[engine]) for engine in engines}
    except (KeyError, TypeError, ValueError):
        issues.append(_issue(source, "every counter engine requires an explicit valid ResourceKind"))
        return None, None, "", {"state": "unknown"}, None
    codes = artifact.get("codes")
    module = artifact.get("module")
    counter_module = artifact.get("counter_module")
    if artifact.get("circt_hw_sha256") != circt_hw_sha256:
        issues.append(_issue(
            source, "receipt's CIRCT digest does not match the explicit elaborated RTL input",
            "UNVERIFIED_COUNTER_PARTITION"))
        return None, None, "", {"state": "unknown"}, None
    if "hw_text" in artifact:
        issues.append(_issue(
            source, "raw CIRCT text belongs at --circt-hw, not in a caller-authored receipt",
            "UNVERIFIED_COUNTER_PARTITION"))
        return None, None, "", {"state": "unknown"}, None
    if (not hw_text or not isinstance(codes, Mapping)
            or not isinstance(module, str) or not module
            or not isinstance(counter_module, str) or not counter_module):
        issues.append(_issue(
            source, "requires the explicit CIRCT input plus event codes, module, and counter-module identities",
            "UNVERIFIED_COUNTER_PARTITION",
        ))
        return None, None, "", {"state": "unknown"}, None
    eta_record = eta_from_counters(
        dict(readings), counters, hw_text=hw_text, codes=codes, module=module,
        counter_module=counter_module, source=artifact["source"],
        measurement_cycles=measurement_cycles)
    if eta_record.get("state") != "measured" or eta_record.get("complete") is not True:
        proof = eta_record.get("partition_proof")
        code = ("UNVERIFIED_COUNTER_PARTITION"
                if not isinstance(proof, Mapping) or proof.get("status") != "proved"
                else "UNKNOWN_OVERLAP")
        issues.append(_issue(source, str(eta_record.get("why") or "overlap eta is UNKNOWN"),
                             code))
        return None, None, "", eta_record, None
    busy = eta_record["busy_cycles"]
    proof = eta_record["partition_proof"]
    provenance = f"sha256:{_canonical_digest(artifact)}; {proof['method']}:{proof['sha256']}"
    activity = activity_from_busy(
        "composition-probe", measurement_cycles, busy, kinds,
        partitioned=False, completion_observable=True, provenance=provenance,
    )
    derived = composition_operator(
        [activity], observed_overlap_cycles={"composition-probe": eta_record["realised_cycles"]})
    if isinstance(derived, Unavailable):
        issues.append(_issue(source, derived.detail or "; ".join(derived.missing), "UNKNOWN_OVERLAP"))
        return None, None, "", eta_record, None
    operator, eta = derived
    receipt = EvidenceReceipt(
        artifact_sha256=_canonical_digest(artifact),
        source_kind="rtl_counter_partition",
        sample_ids=(_canonical_digest(readings),),
    )
    return operator, eta, provenance, eta_record, receipt


def _parse_observations(document: Mapping[str, Any], rtl_sha256: str,
                        issues: list[dict[str, str]]) \
        -> tuple[list[EmpiricalObservation], tuple[str, ...] | None, dict[str, tuple[str, ...]],
                 dict[str, EvidenceReceipt], dict[str, str]]:
    expected: tuple[str, ...] | None = None
    if "expected_workloads" not in document:
        issues.append(_issue(
            "observations.expected_workloads",
            "is absent; observations may not define their own denominator",
        ))
    else:
        values = _records(document["expected_workloads"], "observations.expected_workloads", issues)
        if values and all(isinstance(value, str) and value.strip() for value in values):
            expected = tuple(values)
        else:
            issues.append(_issue(
                "observations.expected_workloads", "must be a nonempty array of workload identities"))

    observations: list[EmpiricalObservation] = []
    resources: dict[str, tuple[str, ...]] = {}
    receipts: dict[str, EvidenceReceipt] = {}
    protocols: dict[str, str] = {}
    for index, raw in enumerate(_records(
            document.get("observations"), "observations.observations", issues)):
        source = f"observations.observations[{index}]"
        row = _mapping(raw, source, issues)
        for forbidden in ("cycles", "work", "moved_bytes"):
            if forbidden in row:
                issues.append(_issue(
                    f"{source}.{forbidden}",
                    "caller-authored aggregate is refused; supply its raw content-addressed receipt",
                    "UNTRUSTED_AGGREGATE",
                ))
        workload = _text(row, "workload", source, issues)
        protocol = _text(row, "measurement_protocol", source, issues)
        context = {"workload": workload, "measurement_protocol": protocol,
                   "rtl_facts_sha256": rtl_sha256}
        cycles, cycle_provenance, cycle_digest = _cycles(
            row.get("cycle_receipt"), f"{source}.cycle_receipt", context, issues)
        work_amount, work_resource, work_provenance, work_digest = _work(
            row.get("command_buffer_receipt"), f"{source}.command_buffer_receipt", context, issues)
        traffic_amount, traffic_resource, traffic_provenance, traffic_digest = _traffic(
            row.get("traffic_receipt"), f"{source}.traffic_receipt", rtl_sha256, context, issues)
        work = (ResourceDemand(
            work_resource, ResourceKind.COMPUTE, work_amount, "macs", Basis.MOVED,
            provenance=work_provenance,
        ) if work_amount is not None and work_resource else None)
        moved = ((ResourceDemand(
            traffic_resource, ResourceKind.MOVEMENT, traffic_amount, "bytes", Basis.MOVED,
            provenance=traffic_provenance,
        ),) if traffic_amount is not None and traffic_resource else ())
        observations.append(EmpiricalObservation(
            workload=workload, cycles=cycles if cycles is not None else 0,
            work=work, moved_bytes=moved, provenance=cycle_provenance,
        ))
        resources[workload] = tuple(
            resource for resource in (work_resource, traffic_resource) if resource)
        if workload and protocol:
            protocols[workload] = protocol
        if workload and cycles is not None and cycle_digest:
            receipts[f"observation:{workload}"] = EvidenceReceipt(
                artifact_sha256=cycle_digest,
                source_kind="rtl_cycle_measurement",
                sample_ids=(cycle_digest,),
            )
        if workload and work_amount is not None and work_resource and work_digest:
            receipts[f"work:{workload}:{work_resource}"] = EvidenceReceipt(
                artifact_sha256=work_digest,
                source_kind="compiler_ir",
                sample_ids=(work_digest,),
            )
        if workload and traffic_amount is not None and traffic_resource and traffic_digest:
            receipts[f"traffic:{workload}:{traffic_resource}"] = EvidenceReceipt(
                artifact_sha256=traffic_digest,
                source_kind="physical_counter",
                sample_ids=(traffic_digest,),
            )
    return observations, expected, resources, receipts, protocols


def _markdown(document: Mapping[str, Any]) -> str:
    plan, roofline = document["calibration_plan"], document["roofline"]
    lines = [
        "# RTL-derived empirical roofline", "", f"Status: **{document['status']}**", "",
        f"Calibration plan: {plan['ready_sweeps']}/{plan['required_sweeps']} sweeps ready.", "",
        f"Roofline status: **{roofline['status']}**.",
    ]
    if roofline.get("points"):
        lines.extend(["", "| Workload | Bound cycles | Limiter | Resolved |", "|---|---:|---|---|"])
        for name, point in sorted(roofline["points"].items()):
            lines.append(
                f"| {name} | {point['bound_cycles']} | {point['limiter']} | {point['resolved']} |")
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    paths = {
        "rtl_facts": Path(args.rtl_facts),
        "harness_capabilities": Path(args.harness_capabilities),
        "calibration": Path(args.calibration),
        "observations": Path(args.observations),
    }
    inputs: dict[str, dict[str, str]] = {}
    loaded: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    for label, path in paths.items():
        loaded[label], inputs[label], load_issues = _load_json(path, label)
        issues.extend(load_issues)
    hw_text, inputs["circt_hw"], load_issues = _load_circt(Path(args.circt_hw))
    issues.extend(load_issues)
    rtl = _mapping(loaded["rtl_facts"], "rtl_facts", issues)
    capabilities = _mapping(loaded["harness_capabilities"], "harness_capabilities", issues)
    calibration = _mapping(loaded["calibration"], "calibration", issues)
    observations_doc = _mapping(loaded["observations"], "observations", issues)
    rtl_sha256 = inputs["rtl_facts"]["sha256"]
    rtl_inputs = rtl.get("inputs") if isinstance(rtl.get("inputs"), Mapping) else {}
    recorded_circt_sha256 = rtl_inputs.get("core_hw_sha256")
    circt_binding_ok = True
    if (not isinstance(recorded_circt_sha256, str) or len(recorded_circt_sha256) != 64
            or any(char not in "0123456789abcdef" for char in recorded_circt_sha256.lower())):
        circt_binding_ok = False
        issues.append(_issue(
            "rtl_facts.inputs.core_hw_sha256",
            "requires the extractor's full SHA-256 of the CIRCT dialect; short/truncated digests "
            "cannot bind a performance observation",
            "UNVERIFIED_CIRCT_BINDING"))
    elif recorded_circt_sha256.lower() != inputs["circt_hw"]["sha256"]:
        circt_binding_ok = False
        issues.append(_issue(
            "circt_hw", "explicit CIRCT bytes do not match the complete digest recorded in RTL facts",
            "UNVERIFIED_CIRCT_BINDING"))

    plan = build_calibration_plan_from_rtl(rtl, capabilities)
    shared_overheads = _empty_run_overheads(calibration, rtl_sha256, issues)
    peaks, residual_intercepts, samples, fit_receipts = _derive_calibration(
        calibration, plan, rtl_sha256, shared_overheads, issues)
    operator, eta, composition_provenance, overlap, composition_receipt = _composition(
        calibration, rtl_sha256, hw_text=hw_text,
        circt_hw_sha256=inputs["circt_hw"]["sha256"], circt_binding_ok=circt_binding_ok,
        issues=issues)
    observations, expected, workload_resources, evidence_receipts, workload_protocols = _parse_observations(
        observations_doc, rtl_sha256, issues)
    fixed_terms: dict[str, tuple[FixedTerm, ...]] = {}
    for workload, resources in workload_resources.items():
        protocol = workload_protocols.get(workload)
        overhead_record = shared_overheads.get(protocol or "")
        if resources and all(resource in peaks for resource in resources) and overhead_record is not None:
            overhead, overhead_receipt = overhead_record
            fixed_terms[workload] = (FixedTerm(
                name=f"empty-run:{protocol}", cycles=overhead,
                law="measured_structurally_empty_run", provenance=(
                    f"protocol={protocol}; {len(overhead_receipt.sample_ids)} raw empty runs"),
                evidence_kind="calibrated"),)
            evidence_receipts[f"fixed:{workload}"] = overhead_receipt
        elif resources:
            issues.append(_issue(
                f"observations.{workload}", "the workload does not carry a protocol-matched, "
                "four-replicate empty-run baseline; fixed cost is UNKNOWN",
                "UNRESOLVED_FIXED_SCOPE"))
    for resource, receipt in fit_receipts.items():
        evidence_receipts[f"peak:{resource}"] = receipt
    if composition_receipt is not None:
        evidence_receipts["composition"] = composition_receipt
    report = empirical_roofline(
        observations, peaks=peaks, fixed_terms=fixed_terms, composition=operator,
        composition_eta=eta, composition_provenance=composition_provenance,
        evidence_receipts=evidence_receipts,
        expected_workloads=expected,
    )
    if not plan.ready:
        issues.append(_issue(
            "calibration_plan", "one or more RTL-derived sweeps are UNKNOWN or REFUSED",
            "INCOMPLETE_CALIBRATION_PLAN"))
    if not report.complete:
        issues.append(_issue(
            "roofline", "one or more derived inputs or workload identities remain UNKNOWN",
            "INCOMPLETE_ROOFLINE"))
    resolved = not issues and plan.ready and report.complete
    artifact = {
        "schema": _SCHEMA, "status": "resolved" if resolved else "refused",
        "inputs": inputs, "refusals": issues, "calibration_plan": plan.to_dict(),
        "calibration_measurements": {
            "status": "resolved" if peaks and fixed_terms and not issues else "refused",
            "sample_count": len(samples), "samples": samples,
            "derived_peaks": {
                resource: {"value": peak.value, "unit": peak.unit, "n_samples": peak.n_samples,
                           "evidence_kind": peak.evidence_kind,
                           "is_observed_ceiling": peak.is_ceiling,
                           "provenance": peak.provenance}
                for resource, peak in sorted(peaks.items())
            },
            "unattached_fit_intercepts": {
                resource: {"cycles": cycles,
                           "why": "residual scope is not proved resource-local; rate is an "
                                  "observed ceiling and this value is not composed"}
                for resource, cycles in sorted(residual_intercepts.items())
            },
        },
        "composition_evidence": overlap,
        "evidence_receipts": {
            key: receipt.to_dict() for key, receipt in sorted(evidence_receipts.items())},
        "roofline": report.to_dict(),
    }
    return artifact, 0 if resolved else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a fail-closed calibration plan and roofline from exact raw JSON receipts.")
    parser.add_argument("--rtl-facts", required=True, help="exact raw RTL facts JSON path")
    parser.add_argument("--harness-capabilities", required=True,
                        help="exact provenance-bearing harness capabilities JSON path")
    parser.add_argument("--calibration", required=True,
                        help="exact raw calibration samples and overlap receipts JSON path")
    parser.add_argument("--observations", required=True,
                        help="exact raw workload measurement receipts and denominator JSON path")
    parser.add_argument("--circt-hw", required=True,
                        help="exact elaborated CIRCT HW input whose full digest is in RTL facts")
    parser.add_argument("--output-json", required=True, help="exact output artifact JSON path")
    parser.add_argument("--output-markdown", help="optional exact concise Markdown output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {Path(getattr(args, name)).resolve()
              for name in ("rtl_facts", "harness_capabilities", "calibration", "observations", "circt_hw")}
    outputs = {Path(args.output_json).resolve()}
    if args.output_markdown:
        outputs.add(Path(args.output_markdown).resolve())
    if inputs & outputs or len(outputs) != (2 if args.output_markdown else 1):
        _parser().error("input and output paths must be distinct explicit paths")
    artifact, status = build(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown:
        markdown = Path(args.output_markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(artifact), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

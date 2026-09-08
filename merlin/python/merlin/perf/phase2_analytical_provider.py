"""Production host adapter for the fast Phase-2 portfolio gate.

The adapter is deliberately data driven.  Target plugins bind arbitrary emitted-feature JSON
pointers to resource service models; this module contains no target, model, opcode, or shape table.
It joins those models to the already host-verified whole-program analysis, derives source/task
coverage, composes resource occupancy with an explicitly measured operator, and returns the mapping
consumed by :mod:`merlin.perf.phase2_portfolio`.

An absent input remains absent.  In particular, command-buffer bytes are not physical traffic,
static instruction counts are not cycles, and a placement declaration is not executed accelerator
work.  Physical movement requires a counter-calibrated feature coefficient, cycles require a
resource model, and accelerator placement requires source-bound task instruction roles.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.decompose import ResourceKind
from merlin.perf.envelope import Basis, ResourceTime, compose
from merlin.perf.headroom import Composition
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .global_planner import OccupancySummary
from .lane_cost import dtype_bits
from .phase2_portfolio import (
    AnalyticalMetrics,
    FastEvaluationPolicy,
    FourModelQualitySchema,
    QualityObservation,
    standard_four_model_quality_schema,
    unavailable_fast_evaluation,
)

CALIBRATION_SCHEMA = "phase2_host_analytical_calibration_v1"
PROVIDER_BINDING_SCHEMA = "host_fast_analytical_evaluator_binding_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _number(value: Any, label: str, *, integer: bool = False) -> float | int:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"{label} must be {'an integer' if integer else 'numeric'}")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return int(value) if integer else float(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _json_pointer(document: Mapping[str, Any], pointer: str) -> Any:
    """Resolve one RFC-6901 pointer without pattern matching or inferred aliases."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("analytical feature selectors must be absolute JSON pointers")
    value: Any = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping) and token in value:
            value = value[token]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) \
                and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise KeyError(pointer)
    return value


def _load_document(source: Mapping[str, Any] | Path, expected_sha256: str | None
                   ) -> tuple[dict[str, Any], str, str]:
    if isinstance(source, Mapping):
        document = copy.deepcopy(dict(source))
        raw = _canonical(document)
        source_kind = "canonical_mapping"
    else:
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("analytical calibration must be a real JSON file")
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, Mapping):
            raise ValueError("analytical calibration JSON must contain an object")
        document = dict(document)
        source_kind = "exact_file_bytes"
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("analytical calibration digest mismatch")
    return document, digest, source_kind


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _parse_composition(value: Any) -> Composition:
    try:
        return Composition(str(value))
    except ValueError as exc:
        raise ValueError("calibration composition must be sum, max, or partial") from exc


@dataclass(frozen=True)
class _MovementCalibration:
    bytes_per_cycle: float
    base_latency_cycles: float
    residual_cycles: float
    domain_bytes: tuple[int, int]
    provenance_sha256: str


@dataclass(frozen=True)
class _Feature:
    id: str
    pointer: str
    resource: str
    kind: str
    cycles_per_unit: CycleInterval | None
    physical_bytes_per_unit: float | None
    commands_per_unit: int | None
    transitions_per_unit: int | None
    floor_cycles_per_unit: float | None
    effects: tuple[str, ...]
    provenance_sha256: str


@dataclass(frozen=True)
class _Calibration:
    document_sha256: str
    document_source: str
    target_sha256: str
    features: tuple[_Feature, ...]
    composition: Composition
    composition_eta: float
    composition_provenance_sha256: str
    movement: _MovementCalibration | None
    accelerator_compute_roles: tuple[str, ...]
    risk_score: float
    evidence_sha256s: tuple[str, ...]


def _interval(value: Any, label: str, provenance: str) -> CycleInterval | None:
    if value is None:
        return None
    row = _mapping(value)
    lo = float(_number(row.get("lo"), f"{label} lower endpoint"))
    hi = float(_number(row.get("hi"), f"{label} upper endpoint"))
    if hi < lo:
        raise ValueError(f"{label} interval is inverted")
    return CycleInterval(lo, hi, provenance=(provenance,))


def _parse_calibration(document: Mapping[str, Any], digest: str, source_kind: str) -> _Calibration:
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"analytical calibration must use {CALIBRATION_SCHEMA}")
    target_sha256 = str(document.get("target_sha256") or "")
    if not _sha256(target_sha256):
        raise ValueError("analytical calibration must bind an exact target descriptor")
    evidence = tuple(str(value) for value in _sequence(document.get("evidence_sha256s")))
    if not evidence or len(set(evidence)) != len(evidence) or any(not _sha256(value) for value in evidence):
        raise ValueError("calibration must bind distinct content-addressed evidence receipts")

    composition_row = _mapping(document.get("composition"))
    composition = _parse_composition(composition_row.get("operator"))
    eta = float(_number(composition_row.get("eta"), "composition eta"))
    if eta > 1:
        raise ValueError("composition eta must be in [0, 1]")
    composition_receipt = str(composition_row.get("provenance_sha256") or "")
    if composition_receipt not in evidence:
        raise ValueError("composition is not bound to a declared evidence receipt")

    movement_row = _mapping(document.get("movement_balance"))
    movement = None
    if movement_row:
        receipt = str(movement_row.get("provenance_sha256") or "")
        domain = _sequence(movement_row.get("domain_bytes"))
        if (movement_row.get("schema") != "merlin_movement_balance_v1"
                or movement_row.get("status") != "derived"
                or receipt not in evidence
                or len(domain) != 2
                or int(_number(domain[0], "movement domain minimum", integer=True)) <= 0
                or int(_number(domain[1], "movement domain maximum", integer=True))
                < int(domain[0])
                or int(_number(movement_row.get("n_distinct_sizes"),
                               "movement distinct sizes", integer=True)) < 3):
            raise ValueError("movement balance is not a derived, receipt-bound transfer series")
        rate = float(_number(movement_row.get("peak_bytes_per_cycle"), "movement rate"))
        base = float(_number(movement_row.get("base_latency_cycles"), "movement base latency"))
        if rate <= 0:
            raise ValueError("movement rate must be positive")
        residuals = tuple(float(value) for value in _sequence(movement_row.get("residual_cycles")))
        if any(not math.isfinite(value) for value in residuals):
            raise ValueError("movement residuals must be finite")
        movement = _MovementCalibration(rate, base,
                                        max((abs(value) for value in residuals), default=0.0),
                                        (int(domain[0]), int(domain[1])),
                                        receipt)

    features: list[_Feature] = []
    for raw in _sequence(document.get("features")):
        row = _mapping(raw)
        ident, pointer, resource = (str(row.get(key) or "") for key in
                                    ("id", "pointer", "resource"))
        kind = str(row.get("kind") or "")
        receipt = str(row.get("provenance_sha256") or "")
        if (not ident or not resource or kind not in {"compute", "movement", "encoding", "fixed"}
                or receipt not in evidence):
            raise ValueError("each analytical feature needs an id, resource, kind, and bound receipt")
        cycles = _interval(row.get("cycles_per_unit"), f"feature {ident} cycles",
                           f"calibration sha256:{receipt}")
        physical = row.get("physical_bytes_per_unit")
        commands = row.get("commands_per_unit")
        transitions = row.get("transitions_per_unit")
        floor = row.get("floor_cycles_per_unit")
        feature = _Feature(
            ident, pointer, resource, kind, cycles,
            None if physical is None else float(_number(physical, f"feature {ident} physical bytes")),
            None if commands is None else int(_number(commands, f"feature {ident} commands", integer=True)),
            None if transitions is None else int(_number(
                transitions, f"feature {ident} transitions", integer=True)),
            None if floor is None else float(_number(floor, f"feature {ident} floor cycles")),
            tuple(sorted(str(effect) for effect in _sequence(row.get("effects")))), receipt)
        if feature.kind in {"compute", "fixed"} and feature.cycles_per_unit is None:
            raise ValueError("compute/fixed analytical features require calibrated cycle intervals")
        if feature.kind in {"movement", "encoding"} and feature.cycles_per_unit is None \
                and movement is None:
            raise ValueError("movement features require cycle intervals or a movement balance")
        features.append(feature)
    if not features or len({feature.id for feature in features}) != len(features):
        raise ValueError("calibration requires unique analytical feature ids")
    kinds = {feature.kind for feature in features}
    if not {"compute", "movement", "encoding"} <= kinds:
        raise ValueError("calibration must cover compute, physical movement, and encoding activity")
    for feature in features:
        if feature.kind in {"movement", "encoding"} and (
                feature.physical_bytes_per_unit is None
                or feature.commands_per_unit is None):
            raise ValueError("movement features require physical-byte and command calibrations")
        if feature.kind == "encoding" and feature.transitions_per_unit is None:
            raise ValueError("encoding features require an executed-transition calibration")
    roles = tuple(sorted(str(role) for role in _sequence(document.get("accelerator_compute_roles"))))
    if not roles or any(not role for role in roles):
        raise ValueError("calibration must bind target-derived accelerator compute roles")
    risk = float(_number(document.get("risk_score"), "calibration risk score"))
    if risk > 1:
        raise ValueError("calibration risk score must be in [0, 1]")
    return _Calibration(digest, source_kind, target_sha256, tuple(features), composition, eta,
                        composition_receipt, movement, roles, risk, evidence)


def _arm_artifacts(analysis: Mapping[str, Any], artifacts: Mapping[str, Any], arm: str
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    diagnostics = _mapping(analysis.get("diagnostics"))
    emission = _mapping(analysis.get("emission"))
    if arm == "candidate":
        retained = artifacts
        plan = _mapping(diagnostics.get("verified_global_plan_emission"))
        task_evidence = _mapping(artifacts.get("task_instruction_evidence"))
    else:
        retained = _mapping(artifacts.get("baseline_artifacts"))
        plan = _mapping(diagnostics.get("verified_baseline_global_plan_emission"))
        task_evidence = _mapping(retained.get("task_instruction_evidence"))
    text = retained.get("command_buffer_text")
    if not isinstance(text, str):
        raise ValueError(f"{arm} retained command-buffer bytes are unavailable")
    command = json.loads(text)
    if not isinstance(command, Mapping):
        raise ValueError(f"{arm} command buffer is not an object")
    command_sha = hashlib.sha256(text.encode()).hexdigest()
    lowered_sha = str(retained.get("lowered_sha256") or retained.get("candidate_lowered_sha256") or "")
    expected_command = emission.get(f"{arm}_command_buffer_sha256")
    expected_lowered = emission.get(f"{arm}_lowered_sha256")
    if (command_sha != expected_command or lowered_sha != expected_lowered
            or plan.get("status") != "verified"
            or plan.get("candidate_command_buffer_sha256") != command_sha
            or plan.get("candidate_lowered_sha256") != lowered_sha):
        raise ValueError(f"{arm} analytical inputs do not match the verified emitted artifacts")
    binding = _mapping(task_evidence.get("binding"))
    if (task_evidence.get("status") not in
            {"static_ownership_verified", "short_admitted_static_ownership"}
            or binding.get("command_buffer_sha256") != command_sha
            or binding.get("lowered_sha256") != lowered_sha
            or binding.get("plan_digest") != plan.get("plan_digest")):
        raise ValueError(f"{arm} task instruction evidence is not bound to the verified plan")
    return dict(command), dict(plan), dict(task_evidence)


def _tensor_bytes(command: Mapping[str, Any], name: Any) -> int | None:
    spec = _mapping(_mapping(command.get("tensors")).get(name))
    shape = _sequence(spec.get("shape"))
    bits = dtype_bits(spec.get("dtype"))
    if not shape or bits is None:
        return None
    elements = 1
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            return None
        elements *= extent
    return elements * ((bits + 7) // 8)


def _coverage(command: Mapping[str, Any], plan: Mapping[str, Any],
              task_evidence: Mapping[str, Any], compute_roles: Sequence[str],
              provenance: tuple[str, ...]) -> dict[str, Any] | None:
    declared = _mapping(command.get("params")).get("global_program_plan")
    declared = _mapping(declared)
    tasks = list(_sequence(declared.get("tasks")))
    instruction_tasks = {
        row.get("task_index"): row for row in _sequence(task_evidence.get("tasks"))
        if isinstance(row, Mapping)
    }
    if not tasks or len(instruction_tasks) != len(tasks):
        return None
    role_set = set(compute_roles)
    classified: list[tuple[int, Mapping[str, Any], bool, int]] = []
    seen_sources: set[int] = set()
    for raw in tasks:
        task = _mapping(raw)
        index = task.get("task_index")
        sources = tuple(task.get("source_op_indices") or ())
        evidence = _mapping(instruction_tasks.get(index))
        if (isinstance(index, bool) or not isinstance(index, int)
                or not sources or any(isinstance(value, bool) or not isinstance(value, int)
                                      or value < 0 for value in sources)
                or seen_sources.intersection(sources)):
            return None
        seen_sources.update(sources)
        roles = set(_mapping(evidence.get("role_counts")))
        accelerator = bool(roles.intersection(role_set))
        classified.append((index, task, accelerator, len(sources)))
    source_total = plan.get("source_operations")
    if (isinstance(source_total, bool) or not isinstance(source_total, int)
            or source_total <= 0 or seen_sources != set(range(source_total))):
        return None

    by_index = {index: (task, accelerator, work)
                for index, task, accelerator, work in classified}
    adjacency = {index: set() for index in by_index}
    producer: dict[str, tuple[int, bool]] = {}
    consumers: dict[str, list[tuple[int, bool]]] = {}
    for index, task, accelerator, _ in classified:
        for name in _sequence(task.get("writes")):
            if not isinstance(name, str) or name in producer:
                return None
            producer[name] = (index, accelerator)
        for name in _sequence(task.get("reads")):
            if not isinstance(name, str):
                return None
            consumers.setdefault(name, []).append((index, accelerator))
    for name, readers in consumers.items():
        source = producer.get(name)
        if source is None:
            continue
        source_index, source_accelerator = source
        for reader_index, reader_accelerator in readers:
            if source_accelerator == reader_accelerator:
                adjacency[source_index].add(reader_index)
                adjacency[reader_index].add(source_index)

    def components(accelerator: bool) -> list[list[int]]:
        pending = {index for index, (_, placed, _) in by_index.items()
                   if placed is accelerator}
        result = []
        while pending:
            seed = min(pending)
            todo, component = [seed], []
            pending.remove(seed)
            while todo:
                current = todo.pop()
                component.append(current)
                neighbors = adjacency[current].intersection(pending)
                pending.difference_update(neighbors)
                todo.extend(sorted(neighbors, reverse=True))
            result.append(sorted(component))
        return result

    accelerator_components = components(True)
    regions = tuple(sorted((sum(by_index[index][2] for index in component)
                            for component in accelerator_components), reverse=True))
    islands: dict[str, list[int]] = {}
    for component in components(False):
        kinds = sorted({str(by_index[index][0].get("kind") or "unclassified")
                        for index in component})
        taxonomy = "+".join(kinds)
        aggregate = islands.setdefault(taxonomy, [0, 0])
        aggregate[0] += 1
        aggregate[1] += sum(by_index[index][2] for index in component)

    outputs = set(_sequence(declared.get("output_bindings")))
    crossings: list[str] = []
    for name, lanes in consumers.items():
        produced_on = producer.get(name, (-1, False))[1]
        crossings.extend(name for _, lane in lanes if lane != produced_on)
    crossings.extend(name for name in outputs
                     if producer.get(name, (-1, False))[1] is True)
    sizes = [_tensor_bytes(command, name) for name in crossings]
    if any(value is None for value in sizes):
        return None
    return {
        "supported_work_total": float(source_total),
        "supported_work_placed": float(sum(regions)),
        "largest_connected_region_work": float(regions[0] if regions else 0),
        "connected_region_work": list(regions),
        "host_islands": [
            {"taxonomy": name, "count": values[0], "work": float(values[1])}
            for name, values in sorted(islands.items())
        ],
        "boundary_crossings": len(crossings),
        "boundary_bytes": float(sum(value for value in sizes if value is not None)),
        "work_unit": "verified source operations",
        "provenance": list(provenance),
    }


def _encoding_activity(plan: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _mapping(plan.get("physical_transition_evidence"))
    transitions = list(_sequence(evidence.get("transitions")))
    if evidence.get("status") != "verified" or any(
            _mapping(row).get("status") != "verified" for row in transitions):
        return {"status": "UNKNOWN", "count": None, "bytes": None}
    byte_count = 0
    for raw in transitions:
        row = _mapping(raw)
        load, store = row.get("load_payload_bytes"), row.get("store_payload_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (load, store)):
            return {"status": "UNKNOWN", "count": None, "bytes": None}
        byte_count += load + store
    return {"status": "verified", "count": len(transitions), "bytes": byte_count}


def _context(analysis: Mapping[str, Any], command: Mapping[str, Any], plan: Mapping[str, Any],
             task_evidence: Mapping[str, Any], arm: str) -> dict[str, Any]:
    diagnostics = _mapping(analysis.get("diagnostics"))
    target_activity = _mapping(_mapping(
        diagnostics.get("target_artifact_activity")).get(arm))
    resolution = _mapping(target_activity.get("encoding_resolution"))
    if target_activity.get("status") != "decoded" or resolution.get("status") != "complete":
        # Static issue counts are complete only when every emitted instruction decoded.  Keeping a
        # partial histogram here would turn an omitted instruction into zero analytical demand.
        target_activity = {}
    return {
        "command_buffer": command,
        "command_buffer_activity": _mapping(_mapping(diagnostics.get("arms")).get(arm)),
        "target_activity": target_activity,
        "model_placement": _mapping(_mapping(
            diagnostics.get("model_contraction_placement")).get(arm)),
        "plan": plan,
        "host_activity": _mapping(plan.get("host_activity")),
        "task_instruction_evidence": task_evidence,
        "encoding_activity": _encoding_activity(plan),
    }


def _feature_cycles(feature: _Feature, count: float,
                    movement: _MovementCalibration | None) -> CycleInterval | None:
    if feature.cycles_per_unit is not None:
        return CycleInterval(
            float(feature.cycles_per_unit.lo) * count,
            float(feature.cycles_per_unit.hi) * count,
            provenance=feature.cycles_per_unit.provenance,
        )
    if (movement is None or feature.physical_bytes_per_unit is None
            or feature.commands_per_unit is None):
        return None
    payload_per_unit = feature.physical_bytes_per_unit
    if (payload_per_unit > 0
            and not movement.domain_bytes[0] <= payload_per_unit <= movement.domain_bytes[1]):
        # The fit is licensed only over its measured transfer-size domain. Repeating an in-domain
        # transfer is valid; pretending one whole-program byte count is one giant transfer is not.
        return None
    commands = feature.commands_per_unit * count
    estimate = payload_per_unit * count / movement.bytes_per_cycle \
        + movement.base_latency_cycles * commands
    error = movement.residual_cycles * commands
    provenance = (f"movement-balance sha256:{movement.provenance_sha256}",)
    return CycleInterval(max(0.0, estimate - error), estimate + error, provenance=provenance)


def _compose(resources: Mapping[str, tuple[ResourceKind, float]], calibration: _Calibration) -> float:
    times = tuple(ResourceTime(name, kind, value, "cycles", Basis.MOVED,
                               evidence_kind="calibration_fit",
                               provenance=("calibration sha256:"
                                           + calibration.document_sha256))
                  for name, (kind, value) in sorted(resources.items()))
    result = compose(times, operator=calibration.composition, eta=calibration.composition_eta)
    if not result.known:
        raise ValueError("resource composition remained unresolved")
    return float(result.cycles)


def _metrics(analysis: Mapping[str, Any], artifacts: Mapping[str, Any], arm: str,
             calibration: _Calibration, provider_sha256: str,
             portfolio_sha256: str) -> AnalyticalMetrics:
    command, plan, task_evidence = _arm_artifacts(analysis, artifacts, arm)
    lowered_sha = str(plan["candidate_lowered_sha256"])
    command_sha = str(plan["candidate_command_buffer_sha256"])
    plan_sha = str(plan["plan_digest"])
    target_activity = _mapping(_mapping(
        _mapping(analysis.get("diagnostics")).get("target_artifact_activity")).get(arm))
    provenance = (
        f"lowered sha256:{lowered_sha}", f"command-buffer sha256:{command_sha}",
        f"declared-plan sha256:{plan_sha}",
        f"verified-plan-evidence sha256:{_digest(plan)}",
        f"task-instruction-evidence sha256:{_digest(task_evidence)}",
        f"target-activity-evidence sha256:{_digest(target_activity)}",
        f"calibration sha256:{calibration.document_sha256}",
        f"provider sha256:{provider_sha256}",
        f"target sha256:{calibration.target_sha256}",
        f"portfolio sha256:{portfolio_sha256}",
    )
    context = _context(analysis, command, plan, task_evidence, arm)
    resource_lo: dict[str, tuple[ResourceKind, float]] = {}
    resource_hi: dict[str, tuple[ResourceKind, float]] = {}
    physical_bytes = 0.0
    physical_complete = True
    movement_commands = 0.0
    encoding_count = 0.0
    encoding_bytes = 0.0
    encoding_complete = True
    encoding_lo = encoding_hi = 0.0
    floor_resources: dict[str, tuple[ResourceKind, float]] = {}
    floor_complete = True
    effects_by_resource: dict[str, set[str]] = {}

    def unknown(reason: str) -> AnalyticalMetrics:
        return AnalyticalMetrics.from_mapping({
            "cycles": CycleInterval.unknown(reason).to_dict(),
            "movement_bytes": None, "movement_scope": "unavailable", "occupancy": None,
            "coverage": _coverage(command, plan, task_evidence,
                                  calibration.accelerator_compute_roles, provenance),
            "roofline": None,
            "encoding_conversions": {
                "count": None, "bytes": None,
                "cycles": CycleInterval.unknown(reason).to_dict(),
            },
            "risk_score": calibration.risk_score, "provenance": list(provenance),
        })

    for feature in calibration.features:
        try:
            count = float(_number(_json_pointer(context, feature.pointer),
                                  f"feature {feature.id} count"))
        except (KeyError, TypeError, ValueError):
            count = math.nan
        if not math.isfinite(count):
            return unknown(f"feature {feature.id} is UNKNOWN")
        cycles = _feature_cycles(feature, count, calibration.movement)
        if cycles is None or not cycles.resolved:
            return unknown(f"feature {feature.id} cycles are UNKNOWN")
        resource_kind = (ResourceKind.COMPUTE if feature.kind == "compute" else
                         ResourceKind.FIXED if feature.kind == "fixed" else
                         ResourceKind.MOVEMENT)
        for destination, endpoint in ((resource_lo, float(cycles.lo)),
                                      (resource_hi, float(cycles.hi))):
            previous = destination.get(feature.resource, (resource_kind, 0.0))
            if previous[0] is not resource_kind:
                raise ValueError("one resource cannot have multiple analytical kinds")
            destination[feature.resource] = (resource_kind, previous[1] + endpoint)
        effects_by_resource.setdefault(feature.resource, set()).update(feature.effects)
        if feature.kind in {"movement", "encoding"}:
            if feature.physical_bytes_per_unit is None:
                if count:
                    physical_complete = False
            else:
                physical_bytes += feature.physical_bytes_per_unit * count
            if feature.commands_per_unit is not None:
                movement_commands += feature.commands_per_unit * count
        if feature.kind == "encoding":
            if feature.transitions_per_unit is None or feature.physical_bytes_per_unit is None:
                if count:
                    encoding_complete = False
            else:
                encoding_count += feature.transitions_per_unit * count
                encoding_bytes += feature.physical_bytes_per_unit * count
            encoding_lo += float(cycles.lo)
            encoding_hi += float(cycles.hi)
        if feature.floor_cycles_per_unit is None:
            if count:
                floor_complete = False
        else:
            previous = floor_resources.get(feature.resource, (resource_kind, 0.0))
            floor_resources[feature.resource] = (
                resource_kind, previous[1] + feature.floor_cycles_per_unit * count)

    lo, hi = _compose(resource_lo, calibration), _compose(resource_hi, calibration)
    if hi < lo or lo <= 0:
        raise ValueError("analytical resource model produced an invalid cycle interval")
    engine_sum = sum(value for kind, value in resource_hi.values() if kind.is_engine)
    fixed = sum(value for kind, value in resource_hi.values() if not kind.is_engine)
    compute_resources = tuple(sorted(name for name, (kind, _) in resource_hi.items()
                                     if kind is ResourceKind.COMPUTE))
    movement_resources = tuple(sorted(name for name, (kind, _) in resource_hi.items()
                                      if kind is ResourceKind.MOVEMENT))
    # A composition operator over more than one engine on either side establishes aggregate saved
    # service time, but cannot say which part was compute/movement overlap.  Only the two-resource
    # case licenses the latency-hiding metric.  Fixed service is serial and cancels algebraically.
    two_resource_overlap = len(compute_resources) == len(movement_resources) == 1
    overlap = max(0.0, engine_sum + fixed - hi) if two_resource_overlap else None
    available = (min(resource_hi[compute_resources[0]][1],
                     resource_hi[movement_resources[0]][1])
                 if two_resource_overlap else None)
    movement_elapsed = (resource_hi[movement_resources[0]][1]
                        if len(movement_resources) == 1 else None)
    occupancy = OccupancySummary(
        total_cycles=hi,
        busy_cycles=tuple(sorted((name, value) for name, (_, value) in resource_hi.items())),
        compute_resources=compute_resources, movement_resources=movement_resources,
        movement_elapsed_cycles=movement_elapsed, overlap_cycles=overlap,
        overlap_available_cycles=available, idle_cycles=None, critical_path_cycles=hi,
        movement_bytes=physical_bytes if physical_complete else None,
        movement_commands=(int(movement_commands) if movement_commands.is_integer() else None),
        encoding_transitions=(int(encoding_count)
                              if encoding_complete and encoding_count.is_integer() else None),
        provenance=provenance,
        missing=tuple(filter(None, (
            "a resource timeline for idle cycles",
            ("a compute/movement partition of aggregate overlap"
             if not two_resource_overlap else ""),
            ("a movement elapsed union"
             if len(movement_resources) > 1 else ""),
        ))),
    )

    roofline = None
    if floor_complete and floor_resources:
        lower_bound = _compose(floor_resources, calibration)
        if lower_bound <= lo:
            maximum = max(value for _, value in floor_resources.values())
            limiters = tuple(sorted(name for name, (_, value) in floor_resources.items()
                                    if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12)))
            effects = tuple(sorted({effect for name in limiters
                                    for effect in effects_by_resource.get(name, ())}))
            if effects:
                roofline = {
                    "lower_bound_cycles": lower_bound,
                    "resource_floors": {name: value for name, (_, value)
                                        in floor_resources.items()},
                    "limiting_resources": list(limiters),
                    "optimization_effects": list(effects),
                    "composition": (f"{calibration.composition.value} with measured "
                                    f"eta={calibration.composition_eta:g}"),
                    "provenance": [*provenance,
                                   "composition sha256:"
                                   + calibration.composition_provenance_sha256],
                }
    coverage = _coverage(command, plan, task_evidence,
                         calibration.accelerator_compute_roles, provenance)
    encoding_cycles = (CycleInterval(encoding_lo, encoding_hi, provenance=provenance)
                       if encoding_complete else
                       CycleInterval.unknown("encoding conversion cycle evidence is incomplete"))
    return AnalyticalMetrics.from_mapping({
        "cycles": CycleInterval(lo, hi, provenance=provenance).to_dict(),
        "movement_bytes": physical_bytes if physical_complete else None,
        "movement_scope": "physical" if physical_complete else "unavailable",
        "occupancy": occupancy.to_dict(), "coverage": coverage, "roofline": roofline,
        "encoding_conversions": {
            "count": int(encoding_count) if encoding_complete and encoding_count.is_integer() else None,
            "bytes": encoding_bytes if encoding_complete else None,
            "cycles": encoding_cycles.to_dict(),
        },
        "risk_score": calibration.risk_score, "provenance": list(provenance),
    })


def _quality_observations(raw: Mapping[str, Any], *, model_sha256: str,
                          corpus_sha256: str, arm_sha256s: Mapping[str, str]
                          ) -> dict[str, QualityObservation]:
    result: dict[str, QualityObservation] = {}
    for arm in ("baseline", "candidate"):
        row = _mapping(raw.get(arm))
        evidence_sha = row.get("evidence_sha256")
        if (row.get("model_sha256") != model_sha256
                or row.get("corpus_sha256") != corpus_sha256
                or row.get("artifact_sha256") != arm_sha256s[arm]
                or not _sha256(evidence_sha)):
            raise ValueError(f"{arm} quality observation is not bound to model/corpus/artifact bytes")
        values = _mapping(row.get("values"))
        result[arm] = QualityObservation(
            tuple(sorted((str(name), _finite(value, f"quality {name}"))
                         for name, value in values.items())),
            (f"quality evidence sha256:{evidence_sha}",
             f"corpus sha256:{corpus_sha256}", f"artifact sha256:{arm_sha256s[arm]}"),
            row.get("complete") is True,
        )
    return result


@dataclass(frozen=True)
class FastEvaluatorInstallation:
    """Frozen provider installation or a fail-closed exact-only fallback."""

    quality_schema: FourModelQualitySchema
    provider: Callable[..., Mapping[str, Any]] | None
    policy: FastEvaluationPolicy | None
    provider_binding: Mapping[str, Any] | None
    fallback: Mapping[str, Any] | None

    def experiment_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted directly by ``GlobalPerfExperiment``."""
        if self.provider is None:
            return {}
        return {
            "fast_evaluation_provider": self.provider,
            "fast_evaluation_policy": self.policy,
            "quality_budgets": self.quality_schema.budget_map,
            "fast_evaluation_provider_binding": copy.deepcopy(dict(self.provider_binding or {})),
        }


def build_fast_evaluator_installation(
    ordered_member_sha256s: Sequence[str], *,
    classification_member_sha256: str,
    corpus_sha256_by_member: Mapping[str, str] | None,
    calibration: Mapping[str, Any] | Path | None,
    calibration_sha256: str | None = None,
    quality_observer: Callable[..., Mapping[str, Any]] | None = None,
    quality_observer_sha256: str | None = None,
    policy: FastEvaluationPolicy | None = None,
    maximum_model_seconds: float = 60.0,
) -> FastEvaluatorInstallation:
    """Build an immutable adapter/binding for the global experiment.

    Without every held-out corpus, a quality observer, and a calibration artifact, this returns an
    exact-only installation with no provider kwargs.  The global experiment can therefore continue
    exact transformations while approximate promotion remains impossible.
    """
    quality = standard_four_model_quality_schema(
        ordered_member_sha256s,
        classification_member_sha256=classification_member_sha256,
        corpus_sha256_by_member=corpus_sha256_by_member,
    )
    absent = []
    if quality.mode != "accuracy_bounded":
        absent.append("complete held-out quality corpora")
    if quality_observer is None or not _sha256(quality_observer_sha256):
        absent.append("content-addressed host quality observer")
    if calibration is None:
        absent.append("content-addressed analytical calibration")
    if absent:
        reason = "missing " + ", ".join(absent) + "; approximate transformations are disabled"
        exact = (quality if quality.mode == "exact_only" else
                 type(quality)("exact_only", quality.ordered_member_sha256s, (), reason))
        return FastEvaluatorInstallation(
            exact, None, None, None, unavailable_fast_evaluation(reason=reason))
    if (isinstance(maximum_model_seconds, bool)
            or not isinstance(maximum_model_seconds, (int, float))
            or not 0 < float(maximum_model_seconds) <= 60):
        raise ValueError("host analytical evaluation must have a per-model budget in (0, 60]")
    document, document_sha, source_kind = _load_document(calibration, calibration_sha256)
    parsed = _parse_calibration(document, document_sha, source_kind)
    implementation_sha = _implementation_sha256()
    frozen_document = copy.deepcopy(document)
    evidence_sha256s = tuple(dict.fromkeys((document_sha, *parsed.evidence_sha256s,
                                           str(quality_observer_sha256))))
    binding = {
        "schema": PROVIDER_BINDING_SCHEMA,
        "implementation_sha256": implementation_sha,
        "execution": "host_analytical_only",
        "full_model_simulation_allowed": False,
        "resource_admission": "serialized_one_model_at_a_time",
        "maximum_model_seconds": float(maximum_model_seconds),
        "calibration_sha256s": list(evidence_sha256s),
        "calibration_document_sha256": document_sha,
        "calibration_document_source": source_kind,
        "target_sha256": parsed.target_sha256,
        "quality_observer_sha256": quality_observer_sha256,
        "ordered_member_sha256s": list(quality.ordered_member_sha256s),
        "corpus_sha256_by_member": dict(corpus_sha256_by_member or {}),
        "provider_contract": "no target execution, layer simulation, or complete-model simulation",
    }
    frozen_binding = copy.deepcopy(binding)

    def provider(**kwargs: Any) -> Mapping[str, Any]:
        supplied_binding = kwargs.get("provider_binding")
        if (supplied_binding != frozen_binding
                or _digest(frozen_document) != _digest(document)
                or _implementation_sha256() != implementation_sha):
            raise ValueError("host analytical provider binding or calibration changed")
        sentinel = kwargs.get("sentinel")
        model_sha = getattr(sentinel, "capsule_sha256", None)
        if model_sha not in quality.ordered_member_sha256s:
            raise ValueError("host analytical provider received an unbound portfolio member")
        portfolio_sha = kwargs.get("portfolio_sha256")
        if (not _sha256(portfolio_sha)
                or kwargs.get("target_sha256") != parsed.target_sha256):
            raise ValueError("host analytical provider requires target and portfolio identities")
        descriptor = kwargs.get("target_descriptor")
        if descriptor is not None:
            descriptor_path = Path(descriptor)
            if (descriptor_path.is_symlink() or not descriptor_path.is_file()
                    or hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
                    != parsed.target_sha256):
                raise ValueError("host analytical provider target descriptor changed")
        analysis = _mapping(kwargs.get("analysis"))
        artifacts = _mapping(kwargs.get("artifacts"))
        if (_mapping(analysis.get("workload")).get("capsule_sha256") != model_sha):
            raise ValueError("whole-model analysis does not match the requested portfolio member")
        baseline = _metrics(
            analysis, artifacts, "baseline", parsed, implementation_sha, portfolio_sha)
        candidate = _metrics(
            analysis, artifacts, "candidate", parsed, implementation_sha, portfolio_sha)
        diagnostics = _mapping(analysis.get("diagnostics"))
        baseline_sha = str(_mapping(
            diagnostics.get("verified_baseline_global_plan_emission"))
                           .get("candidate_lowered_sha256") or "")
        candidate_sha = str(_mapping(diagnostics.get("verified_global_plan_emission"))
                            .get("candidate_lowered_sha256") or "")
        corpus_sha = str((corpus_sha256_by_member or {})[model_sha])
        quality_raw = quality_observer(
            analysis=analysis, artifacts=artifacts, sentinel=sentinel,
            corpus_sha256=corpus_sha, target_descriptor=kwargs.get("target_descriptor"),
            target_sha256=kwargs.get("target_sha256"),
            execution="host_reference_only_no_target_or_model_simulator")
        if not isinstance(quality_raw, Mapping):
            raise TypeError("host quality observer must return a mapping")
        observations = _quality_observations(
            quality_raw, model_sha256=model_sha, corpus_sha256=corpus_sha,
            arm_sha256s={"baseline": baseline_sha, "candidate": candidate_sha})
        return {
            "baseline": baseline, "candidate": candidate,
            "baseline_quality": observations["baseline"],
            "candidate_quality": observations["candidate"],
            "provider_provenance": {
                "binding_sha256": _digest(frozen_binding),
                "calibration_sha256": parsed.document_sha256,
                "target_sha256": parsed.target_sha256,
                "portfolio_sha256": portfolio_sha,
                "baseline_lowered_sha256": baseline_sha,
                "candidate_lowered_sha256": candidate_sha,
            },
        }

    return FastEvaluatorInstallation(
        quality, provider, policy or FastEvaluationPolicy(), binding, None)


__all__ = [
    "CALIBRATION_SCHEMA",
    "PROVIDER_BINDING_SCHEMA",
    "FastEvaluatorInstallation",
    "build_fast_evaluator_installation",
]

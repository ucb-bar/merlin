"""Gemmini boundary for the generic CIRCT external-counter extractor."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from merlin.perf.counter_binding import extract_external_additive_counters
from merlin.perf.hw_counters import counters_for_target
from merlin.targetgen.rtl import mlc_bridge

_TARGET = "gemmini"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value))


def _input_artifacts() -> tuple[Path, Path]:
    circt = mlc_bridge.core_hw_mlir(_TARGET)
    if circt is None or not Path(circt).is_file():
        raise FileNotFoundError("the target's elaborated core CIRCT HW artifact is unavailable")
    discovery = counters_for_target(_TARGET)
    header = discovery.get("header")
    if discovery.get("status") != "derived" or not isinstance(header, str) or not Path(header).is_file():
        raise FileNotFoundError(discovery.get("why", "the target counter header is unavailable"))
    return Path(circt), Path(header)


def probe_counter_byte_bindings() -> dict[str, Any]:
    """Probe the exact active CIRCT/header pair; never manufacture physical semantics."""
    try:
        circt_path, header_path = _input_artifacts()
        hw_text = circt_path.read_text(encoding="utf-8", errors="replace")
        header_text = header_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        artifact: dict[str, Any] = {
            "schema": "merlin.gemmini-counter-byte-binding-probe.v1",
            "target": _TARGET,
            "status": "unknown",
            "counter_facts": [],
            "why": f"{type(exc).__name__}: {exc}",
        }
    else:
        generic = extract_external_additive_counters(
            hw_text, header_text,
            top_module="Gemmini",
            counter_module="CounterController",
            counter_file_module="CounterFile",
            external_port_prefix="io_event_io_external_values_",
            external_base_define="INCREMENTAL_COUNTERS",
            declared_unit="BYTES",
            source=str(circt_path.resolve()),
            header_source=str(header_path.resolve()),
        )
        artifact = {
            **generic,
            "schema": "merlin.gemmini-counter-byte-binding-probe.v1",
            "target": _TARGET,
        }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _point_problem(point: Any, campaign_inputs: Mapping[str, Any],
                   input_fields: tuple[str, ...]) -> str | None:
    if not isinstance(point, Mapping):
        return "point is not a mapping"
    bindings = point.get("input_bindings")
    if not isinstance(bindings, Mapping):
        return "point has no exact input bindings"
    for field in input_fields:
        if not _digest(bindings.get(field)) or bindings.get(field) != campaign_inputs.get(field):
            return f"point {field} does not match the exact campaign input"
    result = point.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "measured":
        return "RTL run did not complete as a measurement"
    if result.get("correct") is not True:
        return "RTL run did not report correct completion"
    oracle = result.get("oracle")
    if not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True:
        return "measurement did not use an RTL-derived oracle"
    cycles = result.get("cycles")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        return "measurement has no positive raw cycle count"
    if not _digest(point.get("console_sha256")):
        return "measurement console is not content-addressed"
    emitter = result.get("emitter")
    if not isinstance(emitter, Mapping) or emitter.get("status") != "accepted":
        return "measurement lacks an accepted target-header emitter receipt"
    for field in ("emitted_mlir_sha256", "llvm_ir_sha256", "object_sha256"):
        if not _digest(emitter.get(field)):
            return f"emitter receipt lacks exact {field}"
    if not _digest(emitter.get("object_kernel_disassembly_sha256")):
        return "emitter receipt lacks exact kernel disassembly hash"
    header_count = emitter.get("header_custom_instruction_count")
    object_count = emitter.get("object_custom_instruction_count")
    if (isinstance(header_count, bool) or not isinstance(header_count, int) or header_count <= 0
            or object_count != header_count):
        return "compiled program custom-instruction count does not match header expansion"
    if not _digest(result.get("elf_sha256")):
        return "measurement ELF is not content-addressed"
    direction = point.get("direction")
    payload = point.get("requested_payload_bytes")
    if (result.get("direction") != direction or emitter.get("direction") != direction
            or result.get("requested_payload_bytes") != payload
            or emitter.get("requested_payload_bytes") != payload):
        return "program receipt direction/payload does not match its campaign coordinate"
    raw = result.get("raw_counter_readings")
    if not isinstance(raw, Mapping) or raw.get("counter_header_sha256") != campaign_inputs.get(
            "counter_header_sha256"):
        return "raw counter reading is not bound to the exact probed header"
    return None


def evaluate_differential_evidence(structural: Mapping[str, Any], campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Promote physical bindings only after a complete isolated differential campaign.

    Direction is learned from which structurally proved candidate responds to an independently
    generated read-only/write-only program.  Byte scale is learned from exact payload/count ratios,
    never from a counter name.  The current Gemmini RTL is expected to fail the latter check because
    its byte-named accumulators add a full transaction size for every multi-beat response.
    """
    base = {key: value for key, value in structural.items() if key not in {
        "artifact_sha256", "counter_facts", "status", "why"}}
    inputs = campaign.get("inputs")
    points = campaign.get("points")
    issues: list[str] = []
    if not isinstance(inputs, Mapping) or not isinstance(points, list):
        issues.append("differential campaign requires input hashes and raw points")
        inputs, points = {}, []
    structural_inputs = structural.get("inputs")
    core = structural_inputs.get("circt_core_hw") if isinstance(structural_inputs, Mapping) else None
    header = structural_inputs.get("counter_header") if isinstance(structural_inputs, Mapping) else None
    expected_core = core.get("sha256") if isinstance(core, Mapping) else None
    expected_header = header.get("sha256") if isinstance(header, Mapping) else None
    if inputs.get("circt_core_hw_sha256") != expected_core or not _digest(expected_core):
        issues.append("campaign is not bound to the exact structurally probed CIRCT core")
    if inputs.get("counter_header_sha256") != expected_header or not _digest(expected_header):
        issues.append("campaign is not bound to the exact structurally probed counter header")
    if inputs.get("rtl_facts_core_hw_sha256") != expected_core:
        issues.append("RTL facts do not bind the same exact CIRCT core")
    required_hashes = ("circt_core_hw_sha256", "counter_header_sha256",
                       "rtl_facts_sha256", "simulator_binary_sha256")
    if any(not _digest(inputs.get(field)) for field in required_hashes):
        issues.append("campaign inputs lack one or more full artifact hashes")
    derivation = inputs.get("coordinate_derivation")
    if not isinstance(derivation, Mapping):
        issues.append("campaign lacks an automatic coordinate-derivation receipt")
    else:
        receipt = derivation.get("capability_receipt")
        if (derivation.get("method") != "target_header_command_limit_multiples"
                or not isinstance(receipt, Mapping)
                or derivation.get("capability_receipt_sha256") != _canonical_sha256(receipt)):
            issues.append("campaign coordinate derivation is not bound to the target-header capability")

    candidates = structural.get("candidates")
    well_formed_candidates = (isinstance(candidates, list)
                              and all(isinstance(row, Mapping) for row in candidates))
    fields = ([row.get("counter_field") for row in candidates]
              if well_formed_candidates else [])
    if (len(fields) != 2 or any(not isinstance(field, str) for field in fields)
            or any(row.get("status") != "structurally_proved" for row in candidates or [])):
        issues.append("exactly two structurally proved counter candidates are required")
        fields = []

    valid: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ("read", "write", "copy")}
    invalid_points = []
    for index, point in enumerate(points):
        problem = _point_problem(point, inputs, required_hashes)
        direction = point.get("direction") if isinstance(point, Mapping) else None
        if direction not in valid:
            problem = problem or "point direction is not read/write/copy"
        if problem:
            invalid_points.append(f"point[{index}]: {problem}")
        else:
            valid[str(direction)].append(point)
    issues.extend(invalid_points)

    for direction in valid:
        sizes = [point.get("requested_payload_bytes") for point in valid[direction]]
        if (len(sizes) < 4 or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                                 for size in sizes) or len(set(sizes)) != len(sizes)):
            issues.append(f"{direction} needs at least four unique positive automatically derived sizes")
    size_sets = {direction: {point.get("requested_payload_bytes") for point in rows}
                 for direction, rows in valid.items()}
    if len({frozenset(sizes) for sizes in size_sets.values()}) != 1:
        issues.append("read/write/copy must cover the identical derived size coordinates")

    assignment: dict[str, str] = {}
    scales: dict[str, Fraction] = {}
    scale_rows: dict[str, list[dict[str, Any]]] = {"read": [], "write": []}
    if fields:
        for direction in ("read", "write"):
            active_fields: set[str] = set()
            ratios: set[Fraction] = set()
            for point in valid[direction]:
                result = point["result"]
                readings = result["raw_counter_readings"].get("readings")
                if not isinstance(readings, Mapping) or set(readings) != set(fields):
                    issues.append(f"{direction} point lacks exactly the two raw candidate readings")
                    continue
                if any(isinstance(readings[field], bool) or not isinstance(readings[field], int)
                       or readings[field] < 0 for field in fields):
                    issues.append(f"{direction} point has a non-integer/negative raw reading")
                    continue
                active = [field for field in fields if readings[field] > 0]
                silent = [field for field in fields if readings[field] == 0]
                if len(active) != 1 or len(silent) != 1:
                    issues.append(f"{direction} isolation did not activate exactly one counter")
                    continue
                active_fields.add(active[0])
                ratio = Fraction(int(point["requested_payload_bytes"]), int(readings[active[0]]))
                ratios.add(ratio)
                scale_rows[direction].append({
                    "payload_bytes": point["requested_payload_bytes"],
                    "counter_field": active[0], "raw_count": readings[active[0]],
                    "payload_bytes_per_raw_count": {
                        "numerator": ratio.numerator, "denominator": ratio.denominator},
                })
            if len(active_fields) == 1:
                assignment[direction] = next(iter(active_fields))
            else:
                issues.append(f"{direction} points do not identify one consistent counter")
            if len(ratios) == 1:
                scales[direction] = next(iter(ratios))
            else:
                issues.append(f"{direction} points do not establish one exact byte scale")
        if len(assignment) == 2 and assignment.get("read") == assignment.get("write"):
            issues.append("isolated read and write activate the same counter")

    units: dict[str, int] = {}
    byte_unit_evidence: dict[str, dict[str, Any]] = {}
    for direction, scale in scales.items():
        if scale.denominator != 1 or scale.numerator <= 0:
            issues.append(
                f"{direction} byte-unit falsified: raw_count * positive integer unit cannot equal payload "
                f"(payload/raw={scale.numerator}/{scale.denominator})")
            byte_unit_evidence[direction] = {
                "status": "falsified", "unit_bytes": None,
                "payload_bytes_per_raw_count": {
                    "numerator": scale.numerator, "denominator": scale.denominator},
                "why": "no positive integer unit converts the raw counter to the isolated payload",
            }
        else:
            units[direction] = scale.numerator
            byte_unit_evidence[direction] = {
                "status": "proved" if len(valid[direction]) >= 4 else "witnessed",
                "unit_bytes": scale.numerator,
                "why": (None if len(valid[direction]) >= 4
                        else "fewer than four coordinates; insufficient for promotion"),
            }
    for direction in ("read", "write"):
        byte_unit_evidence.setdefault(direction, {
            "status": "unknown", "unit_bytes": None,
            "why": "no consistent isolated scale was observed",
        })

    direction_evidence = {
        direction: {
            "status": ("proved" if direction in assignment and len(valid[direction]) >= 4
                       else "witnessed" if direction in assignment else "unknown"),
            "counter_field": assignment.get(direction),
            "isolated_points": len(valid[direction]),
            "promotion_eligible": direction in assignment and len(valid[direction]) >= 4,
        }
        for direction in ("read", "write")
    }

    if len(assignment) == 2 and len(units) == 2:
        for point in valid["copy"]:
            readings = point["result"]["raw_counter_readings"].get("readings")
            payload = int(point["requested_payload_bytes"])
            if (not isinstance(readings, Mapping) or set(readings) != set(fields)
                    or any(isinstance(readings.get(assignment[direction]), bool)
                           or not isinstance(readings.get(assignment[direction]), int)
                           or readings[assignment[direction]] * units[direction] != payload
                           for direction in ("read", "write"))):
                issues.append("copy point does not cross-check both directional byte scales")
                break

    evidence_body = {"inputs": dict(inputs), "points": points}
    evidence_sha = _canonical_sha256(evidence_body)
    facts = []
    if not issues and len(assignment) == 2 and len(units) == 2:
        for direction in ("read", "write"):
            facts.append({
                "fact_kind": "counter_byte_binding",
                "artifact_sha256": expected_core,
                "counter_field": assignment[direction],
                "direction": direction,
                "unit_bytes": units[direction],
                "derived_from_rtl": True,
                "provenance": ("isolated target-header DMA differential on exact RTL; "
                               f"evidence_sha256={evidence_sha}"),
            })
    artifact = {
        **base,
        "schema": "merlin.gemmini-counter-byte-binding-probe.v2",
        "target": _TARGET,
        "status": "proved" if facts else "unknown",
        "counter_facts": facts,
        "differential_evidence_sha256": evidence_sha,
        "differential_campaign": evidence_body,
        "differential_evidence": {
            "direction_assignment": assignment,
            "direction_evidence": direction_evidence,
            "scale_observations": scale_rows,
            "byte_unit_evidence": byte_unit_evidence,
            "unit_bytes": units,
            "issues": issues,
        },
        "why": None if facts else ("counter binding promotion refused: " + "; ".join(issues)),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def run_differential_probe(*, facts_path: str | Path | None, simulator: str, timeout: int,
                           workdir: str | Path, full_campaign: bool = True) -> dict[str, Any]:
    """Execute an explicit RTL differential stage and evaluate it for promotion.

    The full stage runs read/write/copy at four automatically derived coordinates.  The witness stage
    runs only the first derived read coordinate and can record a falsification cheaply, but can never
    promote bindings because it intentionally lacks the complete matrix.
    """
    from merlin.perf.calibration_plan import MIN_POINTS_PER_PARAMETER
    from merlin.targetgen.rtl.facts import ensure_facts
    from . import calibration_capabilities, gemmini, gemmini_dma_calibration as dma

    structural = probe_counter_byte_bindings()
    structural_inputs = structural.get("inputs")
    core = structural_inputs.get("circt_core_hw") if isinstance(structural_inputs, Mapping) else None
    header = structural_inputs.get("counter_header") if isinstance(structural_inputs, Mapping) else None
    core_sha = core.get("sha256") if isinstance(core, Mapping) else None
    header_sha = header.get("sha256") if isinstance(header, Mapping) else None

    facts_file = ensure_facts(_TARGET, explicit=facts_path)
    facts_raw = facts_file.read_bytes()
    rtl_facts = json.loads(facts_raw)
    facts_inputs = rtl_facts.get("inputs") if isinstance(rtl_facts, Mapping) else None
    facts_core = facts_inputs.get("core_hw_sha256") if isinstance(facts_inputs, Mapping) else None
    points_required = 2 * MIN_POINTS_PER_PARAMETER
    maximum, maximum_receipt = dma.max_command_payload_bytes(rtl_facts)
    sizes = dma.derived_transfer_ladder(rtl_facts, points=points_required)
    if not full_campaign:
        sizes = sizes[:1]

    protocol_rows, protocol_error = calibration_capabilities._measurement_protocols()
    protocols = sorted({row.get("measurement_protocol") for row in protocol_rows
                        if isinstance(row, Mapping) and isinstance(row.get("measurement_protocol"), str)})
    if protocol_error or not protocols:
        raise RuntimeError(protocol_error or "target harness exposed no measurement protocol")
    protocol = protocols[0]

    if simulator == "verilator":
        simulator_path = Path(gemmini.verilator_path())
    elif simulator == "gsim":
        simulator_path = Path(gemmini.gsim_path())
    else:
        raise ValueError("differential counter proof requires the target RTL simulator")
    if not simulator_path.is_file():
        raise FileNotFoundError(f"RTL simulator executable is unavailable: {simulator_path}")
    campaign_inputs = {
        "circt_core_hw_sha256": core_sha,
        "counter_header_sha256": header_sha,
        "rtl_facts_sha256": hashlib.sha256(facts_raw).hexdigest(),
        "rtl_facts_core_hw_sha256": facts_core,
        "simulator_binary_sha256": _file_sha256(simulator_path),
        "simulator": simulator,
        "measurement_protocol": protocol,
        "protocol_receipts_sha256": _canonical_sha256(protocol_rows),
        "coordinate_derivation": {
            "method": "target_header_command_limit_multiples",
            "required_points": points_required,
            "sizes_bytes": list(sizes),
            "maximum_payload_bytes": maximum,
            "capability_receipt": maximum_receipt,
            "capability_receipt_sha256": _canonical_sha256(maximum_receipt),
        },
    }
    hash_fields = ("circt_core_hw_sha256", "counter_header_sha256",
                   "rtl_facts_sha256", "simulator_binary_sha256")
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    directions = ("read", "write", "copy") if full_campaign else ("read",)
    points: list[dict[str, Any]] = []
    for direction in directions:
        for payload in sizes:
            point_dir = root / direction / str(payload)
            try:
                result = dma.run_dma_calibration(
                    direction, payload, rtl_facts, protocol=protocol, simulator=simulator,
                    timeout=timeout, workdir=point_dir)
            except Exception as exc:  # preserve a failed coordinate; never silently shrink the matrix
                result = {
                    "status": "unknown", "direction": direction,
                    "requested_payload_bytes": payload,
                    "why": f"{type(exc).__name__}: {exc}",
                }
            console = result.get("console") if isinstance(result, Mapping) else None
            points.append({
                "direction": direction,
                "requested_payload_bytes": payload,
                "input_bindings": {field: campaign_inputs[field] for field in hash_fields},
                "console_sha256": (hashlib.sha256(console.encode("utf-8")).hexdigest()
                                   if isinstance(console, str) else None),
                "result": result,
            })
    return evaluate_differential_evidence(structural, {
        "schema": "merlin.gemmini-counter-differential-campaign.v1",
        "inputs": campaign_inputs,
        "points": points,
    })

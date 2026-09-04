"""Fail-closed CIRCT proof and simulator-feasibility probe for physical bus beats.

Physical traffic is not a software payload size and it is not established by a counter whose name
contains ``BYTES``.  The only generally useful measurement is an accepted interface beat in an exact
measurement window.  This module therefore does two deliberately separate jobs:

* recover a *structural monitor candidate* from an already routed additive-counter proof: the update
  predicate must be the conjunction of two one-bit module ports and its encoded-extent bundle must
  contain an integer data port whose width comes from CIRCT; and
* decide whether a particular compiled simulator exposes enough observation machinery to sample that
  candidate.  Missing trace support, missing public signals, or a missing window marker is UNKNOWN.

Even a complete structural candidate is not a physical-byte semantic fact.  Port spellings and bundle
shape are useful for finding signals, but are not accepted as proof that the bundle is a host-memory
payload channel.  A target-owned integration must independently establish that semantic binding and
must content-bind it to this proof before emitting any physical-byte fact.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value))


def _split_items(text: str) -> tuple[str, ...]:
    items: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closes = {")": "(", "]": "[", "}": "{", ">": "<"}
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in depths:
            depths[char] += 1
        elif char in closes and depths[closes[char]]:
            depths[closes[char]] -= 1
        elif char == "," and not any(depths.values()):
            item = text[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    item = text[start:].strip()
    if item:
        items.append(item)
    return tuple(items)


def _balanced_slice(text: str, start: int) -> str | None:
    if start < 0 or start >= len(text) or text[start] != "(":
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return None


def _module(hw_text: str, name: str) -> tuple[str, list[str]] | None:
    marker = "@" + name + "("
    lines = (hw_text or "").splitlines()
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("hw.module") or marker not in stripped:
            continue
        body = [raw]
        cursor = index + 1
        while cursor < len(lines) and not any("{" in line for line in body):
            body.append(lines[cursor])
            cursor += 1
        depth = sum(line.count("{") - line.count("}") for line in body)
        while cursor < len(lines) and depth > 0:
            body.append(lines[cursor])
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            cursor += 1
        if depth:
            return None
        header = " ".join(line.strip() for line in body[:next(
            (position + 1 for position, line in enumerate(body) if "{" in line), len(body))])
        return header, body
    return None


def _integer_width(type_text: str) -> int | None:
    token = type_text.strip().split(None, 1)[0]
    if not token.startswith("i") or not token[1:].isdigit():
        return None
    width = int(token[1:])
    return width if width > 0 else None


def _ports(header: str, module: str) -> dict[str, dict[str, Any]]:
    marker = "@" + module
    open_at = header.find("(", header.find(marker) + len(marker))
    parameters = _balanced_slice(header, open_at)
    if parameters is None:
        return {}
    ports: dict[str, dict[str, Any]] = {}
    for parameter in _split_items(parameters):
        direction, separator, remainder = parameter.partition(" ")
        if not separator or direction not in {"in", "out"}:
            continue
        name, separator, type_text = remainder.partition(":")
        if not separator:
            continue
        name = name.strip()
        ref = name if name.startswith("%") else "%" + name
        ports[ref] = {
            "name": name.removeprefix("%"),
            "direction": direction,
            "type": type_text.strip(),
            "width_bits": _integer_width(type_text),
        }
    return ports


def _definitions(body: Sequence[str]) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for raw in body:
        lhs, separator, rhs = raw.strip().partition(" = ")
        if separator:
            for result in _split_items(lhs):
                if result.startswith("%"):
                    definitions[result] = rhs
    return definitions


def _output_aliases(body: Sequence[str], ports: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Map an ``hw.output`` SSA operand back to its module-boundary output port."""
    outputs = [ref for ref, record in ports.items() if record.get("direction") == "out"]
    output_lines = [line.strip() for line in body if line.strip().startswith("hw.output")]
    if len(output_lines) != 1:
        return {}
    operands = output_lines[0][len("hw.output"):].split(" :", 1)[0].strip()
    values = _split_items(operands)
    if len(values) != len(outputs):
        return {}
    # An input may also be forwarded to one or more outputs.  Keep that input identity: choosing one
    # of several aliases from the output list would be arbitrary.  Internal SSA values, by contrast,
    # need their unique boundary name to be monitorable.
    aliases: dict[str, str] = {}
    for value, output in zip(values, outputs):
        if value not in ports and value not in aliases:
            aliases[value] = output
    return aliases


def _operation_refs(rhs: str, operation: str) -> tuple[str, ...] | None:
    prefix = operation + " "
    if not rhs.startswith(prefix):
        return None
    operands = rhs[len(prefix):]
    for marker in (" {", " :"):
        operands = operands.split(marker, 1)[0]
    if operands.startswith("bin "):
        operands = operands[4:]
    refs = tuple(item.strip() for item in _split_items(operands))
    return refs or None


def _data_handshake_bundles(ports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All flattened data bundles with opposing ready vs valid/data port directions."""
    bundles: list[dict[str, Any]] = []
    tail = "_bits_data"
    for data_ref, data in ports.items():
        name = data.get("name")
        width = data.get("width_bits")
        if not isinstance(name, str) or not name.endswith(tail):
            continue
        bundle = name[:-len(tail)]
        valid_ref, ready_ref = "%" + bundle + "_valid", "%" + bundle + "_ready"
        valid, ready = ports.get(valid_ref), ports.get(ready_ref)
        if (not isinstance(valid, Mapping) or not isinstance(ready, Mapping)
                or valid.get("width_bits") != 1 or ready.get("width_bits") != 1
                or valid.get("direction") != data.get("direction")
                or ready.get("direction") == valid.get("direction")
                or not isinstance(width, int) or width <= 0 or width % 8):
            continue
        bundles.append({
            "valid_port": valid_ref, "ready_port": ready_ref, "data_port": data_ref,
            "data_width_bits": width, "beat_width_bytes": width // 8,
        })
    return bundles


def _candidate_monitor(hw_text: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    accumulator = candidate.get("accumulator_proof")
    if not isinstance(accumulator, Mapping) or accumulator.get("status") != "proved":
        return {"status": "unknown", "why": "candidate lacks a proved additive accumulator"}
    module_name = accumulator.get("leaf_module")
    predicate = accumulator.get("update_predicate_ssa")
    extent = accumulator.get("encoded_extent_ssa")
    if not all(isinstance(value, str) and value for value in (module_name, predicate, extent)):
        return {"status": "unknown", "why": "accumulator proof lacks exact module/predicate/extent SSA"}
    found = _module(hw_text, str(module_name))
    if found is None:
        return {"status": "unknown", "why": f"leaf module @{module_name} is absent"}
    header, body = found
    ports = _ports(header, str(module_name))
    definitions = _definitions(body)
    handshake = _operation_refs(definitions.get(str(predicate), ""), "comb.and")
    if handshake is not None:
        aliases = _output_aliases(body, ports)
        handshake = tuple(aliases.get(ref, ref) for ref in handshake)
    if handshake is None or len(handshake) != 2 or any(
            ref not in ports or ports[ref].get("width_bits") != 1 for ref in handshake):
        return {
            "status": "unknown",
            "why": "counter update is not the conjunction of exactly two one-bit leaf-module ports",
        }
    extent_ref = str(extent)
    extent_port = ports.get(extent_ref)
    if extent_port is None:
        return {"status": "unknown", "why": "encoded extent is not an exact leaf-module port"}

    # CIRCT's flattened aggregate convention prints ``..._bits_size`` and ``..._bits_data`` in the
    # same bundle.  This is only a structural association; the result below explicitly withholds
    # protocol and physical-byte semantics.
    extent_name = extent_port["name"]
    size_tail = "_bits_size"
    if not extent_name.endswith(size_tail):
        return {"status": "unknown", "why": "encoded extent has no structurally paired data field"}
    bundle = extent_name[:-len(size_tail)]
    data_ref = "%" + bundle + "_bits_data"
    valid_ref = "%" + bundle + "_valid"
    data_port = ports.get(data_ref)
    width = data_port.get("width_bits") if isinstance(data_port, Mapping) else None
    if (valid_ref not in handshake or not isinstance(width, int) or width <= 0 or width % 8):
        alternatives = _data_handshake_bundles(ports)
        if len(alternatives) == 1:
            only = alternatives[0]
            return {
                "status": "monitor_derivable",
                "method": "circt_unique_data_handshake_in_counter_leaf_v1",
                "module": module_name,
                "counter_field": candidate.get("counter_field"),
                "event_predicate_ssa": None,
                "event_expression": {
                    "operation": "and", "operands": [only["valid_port"], only["ready_port"]]},
                "derivation_anchor_predicate_ssa": predicate,
                "encoded_extent_port": extent_ref,
                **only,
                "mechanical_formula": f"accepted_event_count * {only['beat_width_bytes']}",
                "bundle_association": "unique_structural_data_handshake_in_counter_leaf",
                "physical_semantics": "unknown",
                "why": ("the counter leaf has exactly one byte-aligned decoupled data interface; "
                        "CIRCT still does not prove that it carries host-memory payload bytes"),
            }
        return {
            "status": "unknown",
            "module": module_name,
            "counter_field": candidate.get("counter_field"),
            "event_predicate_ssa": predicate,
            "handshake_ports": list(handshake),
            "encoded_extent_port": extent_ref,
            "candidate_data_handshakes": alternatives,
            "why": ("the counter event bundle has no byte-aligned data port, or its valid port is not "
                    "part of the proved update predicate; alternate data handshakes are not selected "
                    "when more than one exists"),
        }
    ready_ref = handshake[0] if handshake[1] == valid_ref else handshake[1]
    return {
        "status": "monitor_derivable",
        "method": "circt_counter_event_exact_and_flattened_bundle_v1",
        "module": module_name,
        "counter_field": candidate.get("counter_field"),
        "event_predicate_ssa": predicate,
        "valid_port": valid_ref,
        "ready_port": ready_ref,
        "encoded_extent_port": extent_ref,
        "data_port": data_ref,
        "data_width_bits": width,
        "beat_width_bytes": width // 8,
        "mechanical_formula": f"accepted_event_count * {width // 8}",
        "bundle_association": "structural_name_only",
        "physical_semantics": "unknown",
        "why": ("CIRCT proves a handshake-shaped event and a byte-aligned data width; it does not "
                "prove that this bundle carries host-memory payload bytes"),
    }


def derive_counter_beat_monitors(hw_text: str, counter_probe: Mapping[str, Any], *,
                                 source: str | None = None) -> dict[str, Any]:
    """Derive content-addressed monitor candidates without promoting byte semantics."""
    candidates = counter_probe.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = []
    monitors = [
        _candidate_monitor(hw_text, row)
        for row in candidates
        if isinstance(row, Mapping) and row.get("status") == "structurally_proved"
    ]
    derived = sum(row.get("status") == "monitor_derivable" for row in monitors)
    artifact = {
        "schema": "merlin.bus-beat-monitor-proof.v1",
        "status": "structural_only" if derived else "unknown",
        "circt_hw": {"source": source, "sha256": _sha256((hw_text or "").encode("utf-8"))},
        "counter_probe_sha256": counter_probe.get("artifact_sha256"),
        "monitors": monitors,
        "physical_byte_facts": [],
        "why": (f"{derived} monitor candidate(s) are mechanically derivable, but no protocol semantic "
                "binding was supplied; physical bytes remain UNKNOWN" if derived else
                "no structurally proved counter candidate yields a byte-aligned data-beat monitor"),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def measure_physical_beat_trace(trace: Mapping[str, Any], monitor_proof: Mapping[str, Any], *,
                                semantic_bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count exact accepted beats only from a complete, content-bound per-cycle trace.

    The independent bindings are intentionally not inferred here.  Each one must identify one
    structurally derived monitor, establish its read/write direction, bind the exact CIRCT/proof
    digests, and assert that the monitor set is exhaustive for that direction.  Samples must cover
    every integer cycle in the marked half-open window; a sparse event log could silently omit an
    accepted beat and is therefore refused.
    """
    problems: list[str] = []
    expected_proof_sha = _canonical_sha256({
        key: value for key, value in monitor_proof.items() if key != "artifact_sha256"})
    proof_sha = monitor_proof.get("artifact_sha256")
    if not _digest(proof_sha) or proof_sha != expected_proof_sha:
        problems.append("monitor proof is not self content-addressed")
    circt = monitor_proof.get("circt_hw")
    circt_sha = circt.get("sha256") if isinstance(circt, Mapping) else None
    inputs = trace.get("inputs")
    if not isinstance(inputs, Mapping):
        inputs = {}
        problems.append("trace has no content-addressed inputs")
    if inputs.get("monitor_proof_sha256") != proof_sha:
        problems.append("trace does not bind the exact monitor proof")
    if inputs.get("circt_hw_sha256") != circt_sha or not _digest(circt_sha):
        problems.append("trace does not bind the exact CIRCT HW artifact")
    if not _digest(inputs.get("simulator_binary_sha256")):
        problems.append("trace does not bind a full simulator binary digest")

    monitors_value = monitor_proof.get("monitors")
    monitors = ([row for row in monitors_value
                 if isinstance(row, Mapping) and row.get("status") == "monitor_derivable"]
                if isinstance(monitors_value, Sequence) and not isinstance(
                    monitors_value, (str, bytes)) else [])
    by_identity = {
        (row.get("module"), row.get("valid_port"), row.get("ready_port"), row.get("data_port")): row
        for row in monitors
    }
    bound: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    directions: set[str] = set()
    for position, binding in enumerate(semantic_bindings):
        identity = (binding.get("module"), binding.get("valid_port"),
                    binding.get("ready_port"), binding.get("data_port"))
        if identity not in by_identity or identity in bound:
            problems.append(f"binding[{position}] does not identify one unique derived monitor")
            continue
        if binding.get("fact_kind") != "bus_payload_binding":
            problems.append(f"binding[{position}] has the wrong fact kind")
        direction = binding.get("direction")
        if direction not in {"read", "write"}:
            problems.append(f"binding[{position}] direction is UNKNOWN")
        else:
            directions.add(str(direction))
        if (binding.get("derived_from_rtl") is not True
                or binding.get("circt_hw_sha256") != circt_sha
                or binding.get("monitor_proof_sha256") != proof_sha
                or not _digest(binding.get("evidence_sha256"))):
            problems.append(f"binding[{position}] lacks exact independent RTL evidence")
        if binding.get("exhaustive_for_direction") is not True:
            problems.append(f"binding[{position}] is not exhaustive for its direction")
        bound[identity] = binding
    if set(bound) != set(by_identity):
        problems.append("not every derived monitor has exactly one semantic binding")
    if directions != {"read", "write"}:
        problems.append("semantic bindings do not exhaust both physical payload directions")

    window = trace.get("window")
    if not isinstance(window, Mapping):
        window = {}
        problems.append("trace has no exact measurement window")
    start, end = window.get("start_cycle"), window.get("end_cycle")
    if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool)
            or not isinstance(end, int) or start < 0 or end <= start):
        problems.append("trace measurement window is not a positive half-open cycle interval")
        start, end = 0, 0
    if not _digest(window.get("marker_evidence_sha256")):
        problems.append("measurement-window markers lack independent content-addressed evidence")

    samples_value = trace.get("samples")
    samples = (list(samples_value) if isinstance(samples_value, Sequence)
               and not isinstance(samples_value, (str, bytes)) else [])
    cycles = [sample.get("cycle") for sample in samples if isinstance(sample, Mapping)]
    if len(cycles) != len(samples) or cycles != list(range(start, end)):
        problems.append("trace is not one complete ordered sample per cycle in the exact window")

    totals = {"read": 0, "write": 0}
    accepted = {"read": 0, "write": 0}
    if not problems:
        for sample in samples:
            signals = sample.get("signals")
            if not isinstance(signals, Mapping):
                problems.append("trace sample has no signal mapping")
                break
            for identity, monitor in by_identity.items():
                required = [monitor[key] for key in ("valid_port", "ready_port", "data_port")]
                if any(name not in signals for name in required):
                    problems.append("trace sample omits a derived valid/ready/data signal")
                    break
                valid, ready = signals[required[0]], signals[required[1]]
                if valid not in {0, 1, False, True} or ready not in {0, 1, False, True}:
                    problems.append("trace handshake value is not one bit")
                    break
                if bool(valid) and bool(ready):
                    direction = str(bound[identity]["direction"])
                    accepted[direction] += 1
                    totals[direction] += int(monitor["beat_width_bytes"])
            if problems:
                break
    if problems:
        return {
            "schema": "merlin.physical-beat-trace.v1", "status": "unknown",
            "physical_bytes": None, "problems": list(dict.fromkeys(problems)),
        }
    return {
        "schema": "merlin.physical-beat-trace.v1", "status": "exact",
        "basis": "accepted_valid_ready_beats_times_circt_data_width",
        "window": dict(window), "accepted_beats": accepted,
        "physical_bytes": {**totals, "total": totals["read"] + totals["write"]},
        "inputs": dict(inputs), "semantic_bindings": [dict(row) for row in semantic_bindings],
    }


def _make_assignment(text: str, key: str) -> str | None:
    values = []
    for raw in (text or "").splitlines():
        lhs, separator, rhs = raw.partition("=")
        if separator and lhs.strip() == key:
            values.append(rhs.strip().split(None, 1)[0])
    return values[0] if len(values) == 1 else None


def assess_compiled_simulator(*, simulator: Path, public_header: Path | None,
                              build_metadata: Path | None,
                              required_ports: Sequence[str],
                              exact_window_marker: str | None = None) -> dict[str, Any]:
    """Determine whether a compiled model can observe every required beat in an exact window.

    This is intentionally an availability check, not a request to rebuild or patch an external RTL
    checkout.  A prebuilt executable with tracing disabled and no public signal API cannot be turned
    into a measurement by post-processing its console output.
    """
    inputs: dict[str, Any] = {}
    problems: list[str] = []
    capability_notes: list[str] = []
    if not simulator.is_file():
        problems.append("simulator executable is absent")
    else:
        inputs["simulator"] = {"path": str(simulator), "sha256": _sha256(simulator.read_bytes())}

    public_text = ""
    if public_header is None or not public_header.is_file():
        capability_notes.append("compiled model has no public C++ signal header")
    else:
        public_text = public_header.read_text(encoding="utf-8", errors="replace")
        inputs["public_header"] = {
            "path": str(public_header), "sha256": _sha256(public_text.encode("utf-8"))}

    build_text = ""
    if build_metadata is None or not build_metadata.is_file():
        capability_notes.append("compiled model has no trace-capability build metadata")
    else:
        build_text = build_metadata.read_text(encoding="utf-8", errors="replace")
        inputs["build_metadata"] = {
            "path": str(build_metadata), "sha256": _sha256(build_text.encode("utf-8"))}

    trace_value = _make_assignment(build_text, "VM_TRACE")
    trace_compiled = trace_value not in {None, "0"}
    if not trace_compiled:
        capability_notes.append("compiled model metadata reports VM_TRACE disabled")
    missing_ports = [name for name in required_ports if name not in public_text]
    public_api = bool(required_ports) and not missing_ports
    if not public_api:
        capability_notes.append(
            "public model API does not expose every derived valid/ready/data signal")
    if not isinstance(exact_window_marker, str) or not exact_window_marker:
        problems.append("no simulator-visible exact measurement-window marker is bound")

    observable = trace_compiled or public_api
    if not observable:
        problems.extend(capability_notes)
        problems.append("neither waveform tracing nor a direct sampling API is available")
    return {
        "status": "available" if observable and not problems else "unknown",
        "inputs": inputs,
        "trace_compiled": trace_compiled,
        "public_signal_api": public_api,
        "missing_public_ports": missing_ports,
        "exact_window_marker": exact_window_marker,
        "capability_notes": capability_notes,
        "problems": list(dict.fromkeys(problems)),
        "measurement": None,
    }

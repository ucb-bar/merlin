"""Fail-closed extraction of externally routed additive counters from CIRCT HW.

This module deliberately stops one step short of assigning physical semantics.  Elaborated HW can
prove that a counter is routed, resettable, and accumulates ``1 << encoded_extent``.  It cannot, by
itself, prove that the encoded extent is log2(bytes), nor whether the transaction is a host-memory
read or write.  A counter/header spelling is not accepted as that proof.

Target-owned adapters provide only structural identities (module and port-family names).  Numeric
counter ordinals, leaf modules, registers, update predicates, and extent signals are all recovered
from the supplied header and CIRCT artifact.  No target name, counter code, direction, or transfer
unit is embedded here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

from merlin.perf.hw_counters import counters_with_unit, event_codes


def _split_items(text: str) -> tuple[str, ...]:
    """Split a comma-separated MLIR list without regular expressions."""
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
    """Contents of the balanced parenthesised region beginning at ``start``."""
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


def _module_lines(hw_text: str, module: str) -> tuple[list[str] | None, str | None]:
    marker = "@" + module + "("
    lines = (hw_text or "").splitlines()
    for at, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("hw.module") or marker not in line:
            continue
        body = [raw]
        cursor = at + 1
        while cursor < len(lines) and not any("{" in item for item in body):
            body.append(lines[cursor])
            cursor += 1
        depth = sum(item.count("{") - item.count("}") for item in body)
        while cursor < len(lines) and depth > 0:
            body.append(lines[cursor])
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            cursor += 1
        if depth:
            return None, f"hw.module @{module} has unbalanced braces"
        return body, None
    return None, f"hw.module @{module} is absent"


def _module_outputs(body: list[str], module: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    header_parts = []
    for line in body:
        header_parts.append(line.strip())
        if "{" in line:
            break
    header = " ".join(header_parts)
    marker = "@" + module
    open_at = header.find("(", header.find(marker) + len(marker))
    parameters = _balanced_slice(header, open_at)
    if parameters is None:
        return None
    names = []
    for parameter in _split_items(parameters):
        if parameter.startswith("out "):
            name = parameter[4:].split(":", 1)[0].strip()
            if name:
                names.append(name)
    output_lines = [line.strip() for line in body if line.strip().startswith("hw.output")]
    if len(output_lines) != 1:
        return None
    operands = output_lines[0][len("hw.output"):].split(" :", 1)[0].strip()
    values = tuple(item.strip() for item in _split_items(operands))
    return (tuple(names), values) if len(names) == len(values) else None


def _definitions(body: list[str]) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for raw in body:
        lhs, separator, rhs = raw.strip().partition(" = ")
        if separator:
            for result in _split_items(lhs):
                if result.startswith("%"):
                    definitions[result] = rhs
    return definitions


def _instance_parts(line: str) -> tuple[tuple[str, ...], str, str] | None:
    lhs, separator, rhs = line.strip().partition(" = ")
    if not separator or not rhs.startswith("hw.instance "):
        return None
    at = rhs.find("@")
    open_at = rhs.find("(", at)
    if at < 0 or open_at < 0:
        return None
    callee = rhs[at + 1:open_at].strip()
    arguments = _balanced_slice(rhs, open_at)
    if not callee or arguments is None:
        return None
    return _split_items(lhs), callee, arguments


def _instances(body: list[str], callee: str | None = None) -> list[tuple[tuple[str, ...], str, str]]:
    found = []
    for line in body:
        instance = _instance_parts(line)
        if instance is not None and (callee is None or instance[1] == callee):
            found.append(instance)
    return found


def _argument(arguments: str, port: str) -> str | None:
    matches = []
    prefix = port + ":"
    for item in _split_items(arguments):
        if item.startswith(prefix):
            value = item[len(prefix):].strip().split(":", 1)[0].strip()
            matches.append(value)
    return matches[0] if len(matches) == 1 and matches[0].startswith("%") else None


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


def _constant(rhs: str) -> int | None:
    if not rhs.startswith("hw.constant "):
        return None
    token = rhs[len("hw.constant "):].split(" :", 1)[0].strip()
    if token == "true":
        return 1
    if token == "false":
        return 0
    try:
        return int(token, 0)
    except ValueError:
        return None


def _trace_exact_output(
        hw_text: str, module: str, ref: str, *, active: set[tuple[str, str]]) -> dict[str, Any]:
    """Follow only identity-preserving instance outputs to their leaf SSA value."""
    key = (module, ref)
    if key in active:
        return {"status": "unknown", "why": "instance-output routing is cyclic"}
    body, error = _module_lines(hw_text, module)
    if body is None:
        return {"status": "unknown", "why": error}
    active.add(key)
    for results, callee, _arguments in _instances(body):
        if ref not in results:
            continue
        if results.count(ref) != 1:
            active.remove(key)
            return {"status": "unknown", "why": f"SSA result {ref} is ambiguous in @{module}"}
        child, child_error = _module_lines(hw_text, callee)
        if child is None:
            active.remove(key)
            return {"status": "unknown", "why": child_error}
        outputs = _module_outputs(child, callee)
        position = results.index(ref)
        if outputs is None or position >= len(outputs[1]):
            active.remove(key)
            return {"status": "unknown", "why": f"cannot align @{callee} instance results to outputs"}
        child_ref = outputs[1][position]
        traced = _trace_exact_output(hw_text, callee, child_ref, active=active)
        active.remove(key)
        if traced.get("status") != "traced":
            return traced
        return {**traced, "route": [{"module": module, "ssa": ref, "callee": callee,
                                      "output": outputs[0][position]}] + traced["route"]}
    active.remove(key)
    return {"status": "traced", "leaf_module": module, "leaf_ssa": ref, "route": []}


def _is_zero(ref: str, definitions: Mapping[str, str]) -> bool:
    return ref in definitions and _constant(definitions[ref]) == 0


def _zero_extended_source(
        ref: str, definitions: Mapping[str, str], *, active: set[str]) -> str | None:
    """Return the sole non-zero leaf of nested ``comb.concat`` zero extension."""
    if ref in active:
        return None
    rhs = definitions.get(ref)
    if rhs is None:
        return ref
    operands = _operation_refs(rhs, "comb.concat")
    if operands is None:
        return ref
    active.add(ref)
    nonzero = [operand for operand in operands if not _is_zero(operand, definitions)]
    if len(nonzero) != 1:
        active.remove(ref)
        return None
    source = _zero_extended_source(nonzero[0], definitions, active=active)
    active.remove(ref)
    return source


def _prove_additive_accumulator(hw_text: str, module: str, accumulator: str) -> dict[str, Any]:
    body, error = _module_lines(hw_text, module)
    if body is None:
        return {"status": "unknown", "why": error}
    definitions = _definitions(body)
    rhs = definitions.get(accumulator)
    if rhs is None or not rhs.startswith("seq.firreg "):
        return {"status": "unknown", "why": "routed leaf is not a seq.firreg accumulator"}
    next_ref = rhs[len("seq.firreg "):].split(None, 1)[0]
    reset_mux = _operation_refs(definitions.get(next_ref, ""), "comb.mux")
    if reset_mux is None or len(reset_mux) != 3 or not _is_zero(reset_mux[1], definitions):
        return {"status": "unknown", "why": "accumulator lacks an explicit external zero-reset mux"}
    external_reset, update_ref = reset_mux[0], reset_mux[2]
    update_mux = _operation_refs(definitions.get(update_ref, ""), "comb.mux")
    if update_mux is None or len(update_mux) != 3 or update_mux[2] != accumulator:
        return {"status": "unknown", "why": "accumulator does not hold when its update predicate is false"}
    predicate, add_ref = update_mux[0], update_mux[1]
    add = _operation_refs(definitions.get(add_ref, ""), "comb.add")
    if add is None or len(add) != 2 or accumulator not in add:
        return {"status": "unknown", "why": "accumulator update is not old value plus one term"}
    increment_ref = add[1] if add[0] == accumulator else add[0]
    shifted_ref = _zero_extended_source(increment_ref, definitions, active=set())
    shifted = _operation_refs(definitions.get(shifted_ref or "", ""), "comb.shl")
    if shifted is None or len(shifted) != 2 or _constant(definitions.get(shifted[0], "")) != 1:
        return {"status": "unknown", "why": "increment is not a zero-extended one shifted by an encoded extent"}
    extent = _zero_extended_source(shifted[1], definitions, active=set())
    if extent is None or extent in definitions:
        return {"status": "unknown", "why": "shift extent does not resolve to one module input"}
    if external_reset in definitions:
        return {"status": "unknown", "why": "external reset does not resolve to one module input"}
    return {
        "status": "proved",
        "method": "circt_exact_route_resettable_additive_power_of_two_v1",
        "leaf_module": module,
        "accumulator_ssa": accumulator,
        "external_reset_ssa": external_reset,
        "update_predicate_ssa": predicate,
        "encoded_extent_ssa": extent,
        "increment_formula": "1 << encoded_extent",
        "semantic_unit": None,
        "direction": None,
    }


def _depends_on(ref: str, wanted: str, definitions: Mapping[str, str], active: set[str]) -> bool:
    if ref == wanted:
        return True
    if ref in active:
        return False
    rhs = definitions.get(ref)
    if rhs is None:
        return False
    active.add(ref)
    operands: list[str] = []
    cursor = 0
    valid = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")
    while cursor < len(rhs):
        if rhs[cursor] != "%":
            cursor += 1
            continue
        end = cursor + 1
        while end < len(rhs) and rhs[end] in valid:
            end += 1
        if end > cursor + 1:
            operands.append(rhs[cursor:end])
        cursor = end
    result = any(_depends_on(operand, wanted, definitions, active) for operand in operands)
    active.remove(ref)
    return result


def _prove_counter_file_route(
        hw_text: str, counter_module: str, counter_file_module: str,
        external_port: str, external_index: int) -> dict[str, Any]:
    controller, error = _module_lines(hw_text, counter_module)
    if controller is None:
        return {"status": "unknown", "why": error}
    instances = _instances(controller, counter_file_module)
    if len(instances) != 1:
        return {"status": "unknown", "why": "counter-file instance is absent or ambiguous"}
    if _argument(instances[0][2], external_port) != "%" + external_port:
        return {"status": "unknown", "why": "controller does not forward the external port unchanged"}

    counter_file, error = _module_lines(hw_text, counter_file_module)
    if counter_file is None:
        return {"status": "unknown", "why": error}
    definitions = _definitions(counter_file)
    wanted = "%" + external_port
    arrays = []
    for ref, rhs in definitions.items():
        operands = _operation_refs(rhs, "hw.array_create")
        if operands is not None and operands.count(wanted) == 1:
            arrays.append((ref, operands))
    if len(arrays) != 1:
        return {"status": "unknown", "why": "external port is absent or ambiguous in counter-file arrays"}
    array_ref, operands = arrays[0]
    # ``hw.array_create`` prints the highest element first; the last operand is index zero.
    actual_index = len(operands) - 1 - operands.index(wanted)
    if actual_index != external_index:
        return {"status": "unknown", "why": "header ordinal and CIRCT external-array index disagree"}
    outputs = _module_outputs(counter_file, counter_file_module)
    if outputs is None or not any(_depends_on(value, array_ref, definitions, set())
                                  for value in outputs[1]):
        return {"status": "unknown", "why": "external array cannot reach a counter-file output"}
    return {"status": "proved", "external_array_ssa": array_ref,
            "external_index": actual_index, "counter_file_module": counter_file_module}


def extract_external_additive_counters(
        hw_text: str, header_text: str, *, top_module: str, counter_module: str,
        counter_file_module: str, external_port_prefix: str, external_base_define: str,
        declared_unit: str, source: str | None = None, header_source: str | None = None) -> dict[str, Any]:
    """Extract structural counter candidates, withholding unproved physical semantics.

    A candidate is accepted structurally only when the header-derived external ordinal agrees with
    the counter-file array, the exact top-to-leaf route is identity-preserving, and the leaf is a
    resettable additive power-of-two accumulator.  ``counter_facts`` remains empty because neither a
    header token nor an SSA name proves physical direction/unit.
    """
    circt_sha = hashlib.sha256((hw_text or "").encode("utf-8")).hexdigest()
    header_sha = hashlib.sha256((header_text or "").encode("utf-8")).hexdigest()
    base = {
        "schema": "merlin.counter-byte-binding-probe.v1",
        "inputs": {
            "circt_core_hw": {"source": source, "sha256": circt_sha},
            "counter_header": {"source": header_source, "sha256": header_sha},
        },
        "counter_facts": [],
    }
    codes = event_codes(header_text)
    base_code = codes.get(external_base_define)
    candidates = counters_with_unit(header_text, declared_unit)
    if not isinstance(base_code, int) or isinstance(base_code, bool):
        return {**base, "status": "unknown", "candidates": [],
                "why": f"header does not resolve external base define {external_base_define!r}"}
    if not candidates:
        return {**base, "status": "unknown", "candidates": [],
                "why": f"header declares no {declared_unit!r} counter candidates"}

    top, error = _module_lines(hw_text, top_module)
    if top is None:
        return {**base, "status": "unknown", "candidates": [], "why": error}
    controllers = _instances(top, counter_module)
    if len(controllers) != 1:
        return {**base, "status": "unknown", "candidates": [],
                "why": "top-level counter-controller instance is absent or ambiguous"}

    records = []
    for counter_name, code in sorted(candidates.items(), key=lambda item: item[1]):
        external_index = code - base_code
        external_port = external_port_prefix + str(external_index)
        record: dict[str, Any] = {"counter_field": counter_name, "header_code": code,
                                  "external_index": external_index, "external_port": external_port}
        if external_index < 0:
            records.append({**record, "status": "unknown",
                            "why": "candidate maps below the external counter range"})
            continue
        source_ref = _argument(controllers[0][2], external_port)
        if source_ref is None:
            records.append({**record, "status": "unknown",
                            "why": "counter-controller external port is absent or ambiguous"})
            continue
        traced = _trace_exact_output(hw_text, top_module, source_ref, active=set())
        if traced.get("status") != "traced":
            records.append({**record, "status": "unknown", "route_proof": traced,
                            "why": traced.get("why", "external value route is unproved")})
            continue
        accumulator = _prove_additive_accumulator(
            hw_text, str(traced["leaf_module"]), str(traced["leaf_ssa"]))
        counter_file = _prove_counter_file_route(
            hw_text, counter_module, counter_file_module, external_port, external_index)
        structural = accumulator.get("status") == "proved" and counter_file.get("status") == "proved"
        records.append({
            **record,
            "status": "structurally_proved" if structural else "unknown",
            "route": traced,
            "accumulator_proof": accumulator,
            "counter_file_proof": counter_file,
            "direction": None,
            "unit_bytes": None,
            "binding_status": "unknown",
            "why": ("CIRCT proves routing and arithmetic, but does not prove the encoded extent is "
                    "physical bytes or establish host-memory direction; names are not semantic proof")
                    if structural else "structural routing or accumulator proof failed",
        })
    structural_count = sum(record.get("status") == "structurally_proved" for record in records)
    return {
        **base,
        "status": "unknown",
        "candidates": records,
        "structurally_proved_candidates": structural_count,
        "why": ("no counter_byte_binding facts emitted: direction and byte unit require independent "
                "semantic evidence beyond elaborated CIRCT structure and header spellings"),
    }

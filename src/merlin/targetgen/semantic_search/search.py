"""Exact-pattern search for bounded static linalg kernels.

The representation deliberately admits only operation graphs whose typed body,
effects, indexing maps, and operand/result types are known. A direct match is a
conditional rewrite from that operation to a target instruction. Built-in
equivalences cover pure integer commutation and guarded modular addition
reassociation. No floating-point rewrite is inferred from an instruction name.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "merlin.semantic_search.v1"
KERNEL_SCHEMA = "merlin.semantic_kernel.v1"
INSTRUCTION_SCHEMA = "merlin.instruction_semantics.v1"
_COMMUTATIVE_INT = frozenset({"arith.addi", "arith.muli", "arith.andi", "arith.ori", "arith.xori"})
_RULES = [
    {
        "id": "commute_integer_scalar",
        "preconditions": "one pure integer scalar result; two distinct operands; exact arith integer commutative op",
    },
    {
        "id": "reassociate_modular_integer_add",
        "preconditions": (
            "two adjacent pure arith.addi operations; same integer type; empty attributes; "
            "single-use intermediate; all inputs precede rewrite"
        ),
    },
]


@dataclass(frozen=True)
class SearchLimits:
    """Deterministic enumeration limits plus a wall-clock and per-solver bound."""

    max_operations: int = 32
    max_variants_per_operation: int = 32
    max_candidates: int = 128
    timeout_ms: int = 10_000
    solver_timeout_ms: int = 1_000


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _type_size(spelling: str) -> int | None:
    """Return bytes for a statically shaped tensor; dynamic/packed types are unknown."""
    if not spelling.startswith("tensor<") or not spelling.endswith(">"):
        return None
    fields = spelling[7:-1].split("x")
    if len(fields) < 2 or any(not field.isdecimal() for field in fields[:-1]):
        return None
    dtype = fields[-1]
    if dtype == "index":
        return None
    if dtype.startswith(("i", "f")) and dtype[1:].isdecimal():
        bits = int(dtype[1:])
    elif dtype == "bf16":
        bits = 16
    else:
        return None
    if bits < 8 or bits % 8:
        return None
    size = bits // 8
    for field in fields[:-1]:
        size *= int(field)
    return size


def _frontend_type(record: Mapping[str, Any]) -> str | None:
    if isinstance(record.get("type"), str):
        return record["type"]
    shape, dtype = record.get("shape"), record.get("dtype")
    if not isinstance(shape, list) or not isinstance(dtype, str) or not all(_positive_int(dim) for dim in shape):
        return None
    return f"tensor<{'x'.join(str(dim) for dim in shape)}x{dtype}>"


def _from_frontend_op(row: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap one linalg interface row; dataflow beyond this op needs explicit v1."""
    operands = row.get("ins", []) + row.get("outs", [])
    results = row.get("results", [])
    if not isinstance(operands, list) or not isinstance(results, list):
        raise ValueError("frontend operands/results must be lists")
    values = []
    operand_ids = []
    for i, operand in enumerate(operands):
        if not isinstance(operand, Mapping) or (type_name := _frontend_type(operand)) is None:
            raise ValueError(f"frontend operand {i} has unknown static type")
        identity = f"input:{i}"
        values.append(
            {
                "id": identity,
                "type": type_name,
                "size_bytes": _type_size(type_name),
                "source": operand.get("source"),
                "source_result_index": operand.get("result_index"),
            }
        )
        operand_ids.append(identity)
    output_rows = []
    for i, result in enumerate(results):
        if not isinstance(result, Mapping) or (type_name := _frontend_type(result)) is None:
            raise ValueError(f"frontend result {i} has unknown static type")
        output_rows.append({"id": f"result:{i}", "type": type_name, "size_bytes": _type_size(type_name)})
    return {
        "schema": KERNEL_SCHEMA,
        "values": values,
        "operations": [
            {
                "id": str(row.get("id", "op:0")),
                "operation": row.get("operation", row.get("kind")),
                "operands": operand_ids,
                "results": output_rows,
                "indexing_maps": row.get("indexing_maps"),
                "iterator_types": row.get("iterator_types"),
                "scalar_body": row.get("scalar_body"),
                "attributes": row.get("attributes", {}),
                "effects": row.get("effects", []),
            }
        ],
        "outputs": [result["id"] for result in output_rows],
    }


def _normalize_kernel(kernel: Mapping[str, Any]) -> dict[str, Any]:
    if kernel.get("schema") != KERNEL_SCHEMA:
        kernel = _from_frontend_op(kernel)
    if kernel.get("schema") != KERNEL_SCHEMA:
        raise ValueError("unsupported kernel schema")
    operations, inputs = kernel.get("operations"), kernel.get("values")
    if not isinstance(operations, list) or not isinstance(inputs, list) or not operations:
        raise ValueError("kernel needs nonempty operations and an input value list")
    values: dict[str, dict] = {}

    def validate_value(item: Mapping[str, Any]) -> None:
        inferred = _type_size(item["type"])
        declared = item.get("size_bytes")
        if declared is not None and not _positive_int(declared):
            raise ValueError(f"value {item['id']!r} has invalid static size")
        if inferred is not None and declared is not None and inferred != declared:
            raise ValueError(f"value {item['id']!r} size disagrees with its MLIR type")

    for item in inputs:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("id"), str)
            or not isinstance(item.get("type"), str)
        ):
            raise ValueError("malformed kernel input value")
        if item["id"] in values:
            raise ValueError("duplicate kernel value id")
        validate_value(item)
        values[item["id"]] = {**item, "size_bytes": item.get("size_bytes", _type_size(item["type"]))}
    for op in operations:
        if not isinstance(op, Mapping) or not isinstance(op.get("id"), str):
            raise ValueError("malformed kernel operation")
        for operand in op.get("operands", []):
            if operand not in values:
                raise ValueError(f"operand {operand!r} has no preceding SSA value")
        for result in op.get("results", []):
            if (
                not isinstance(result, Mapping)
                or not isinstance(result.get("id"), str)
                or not isinstance(result.get("type"), str)
            ):
                raise ValueError("malformed kernel result")
            if result["id"] in values:
                raise ValueError("duplicate kernel value id")
            validate_value(result)
            values[result["id"]] = {**result, "size_bytes": result.get("size_bytes", _type_size(result["type"]))}
    outputs = kernel.get("outputs", [])
    if not isinstance(outputs, list) or any(value not in values for value in outputs):
        raise ValueError("kernel outputs must refer to SSA values")
    return {"operations": operations, "values": values, "outputs": outputs}


def _body_known(body: Any) -> bool:
    if not isinstance(body, Mapping) or body.get("effects") != []:
        return False
    if body.get("captures"):
        # A captured outer SSA value needs an explicit binding and transfer.
        return False
    operations = body.get("operations")
    return isinstance(operations, list) and all(
        isinstance(op, Mapping) and op.get("effects") == [] and op.get("regions", 0) == 0 for op in operations
    )


def _computation(op: Mapping[str, Any]) -> dict[str, Any] | None:
    if op.get("operation") != "linalg.generic" or op.get("effects", []) != []:
        return None
    attrs = op.get("attributes", {})
    if not isinstance(attrs, Mapping) or set(attrs) - {"indexing_maps", "iterator_types"}:
        return None
    maps, iterators, body = op.get("indexing_maps"), op.get("iterator_types"), op.get("scalar_body")
    if not isinstance(maps, list) or not isinstance(iterators, list) or not _body_known(body):
        return None
    # The parser and selected support both emit typed, alpha-normalized SSA refs.
    normalized_body = dict(body)
    normalized_body.setdefault("captures", [])
    return {
        "kind": "linalg.generic",
        "indexing_maps": maps,
        "iterator_types": iterators,
        "scalar_body": normalized_body,
    }


def _scalar_ref_type(body: Mapping[str, Any], ref: str) -> str | None:
    parts = ref.split(":")
    if len(parts) == 2 and parts[0] == "arg" and parts[1].isdecimal():
        index = int(parts[1])
        args = body.get("arguments", [])
        return args[index] if index < len(args) else None
    if len(parts) == 3 and parts[0] == "op" and parts[1].isdecimal() and parts[2].isdecimal():
        index, result = int(parts[1]), int(parts[2])
        operations = body.get("operations", [])
        if index < len(operations) and result < len(operations[index].get("results", [])):
            return operations[index]["results"][result]
    return None


def _available_before(ref: str, index: int) -> bool:
    parts = ref.split(":")
    return parts[0] == "arg" or (
        len(parts) == 3 and parts[0] == "op" and parts[1].isdecimal() and int(parts[1]) < index
    )


def _one_step_rewrites(computation: dict[str, Any]):
    """Generate locally proven scalar rewrites; each keeps the body SSA shape."""
    body = computation["scalar_body"]
    operations = body["operations"]
    for i, op in enumerate(operations):
        if (
            op.get("op") not in _COMMUTATIVE_INT
            or len(op.get("operands", [])) != 2
            or len(op.get("results", [])) != 1
            or not isinstance(op["results"][0], str)
            or not op["results"][0].startswith("i")
            or not op["results"][0][1:].isdecimal()
            or op["operands"][0] == op["operands"][1]
        ):
            continue
        swapped = json.loads(json.dumps(computation))
        pair = swapped["scalar_body"]["operations"][i]["operands"]
        pair[0], pair[1] = pair[1], pair[0]
        yield swapped, f"commute_integer_scalar:{i}"

    # Unflagged arith.addi is modular integer addition. Reassociate only two
    # adjacent pure operations of the same type when the intermediate has one
    # use. An overflow flag or another use would invalidate this local rewrite.
    for i in range(len(operations) - 1):
        first, second = operations[i : i + 2]
        ref = f"op:{i}:0"
        if any(
            op.get("op") != "arith.addi"
            or op.get("attributes") != {}
            or op.get("effects") != []
            or op.get("regions", 0) != 0
            or len(op.get("results", [])) != 1
            or len(op.get("operands", [])) != 2
            for op in (first, second)
        ):
            continue
        dtype = first["results"][0]
        if (
            dtype != second["results"][0]
            or not isinstance(dtype, str)
            or not dtype.startswith("i")
            or not dtype[1:].isdecimal()
        ):
            continue
        if second["operands"][0] != ref:
            continue
        uses = sum(op.get("operands", []).count(ref) for op in operations[i + 1 :]) + body.get("yields", []).count(ref)
        if uses != 1:
            continue
        a, b = first["operands"]
        c = second["operands"][1]
        if not _available_before(c, i) or any(_scalar_ref_type(body, value) != dtype for value in (a, b, c)):
            continue
        reassociated = json.loads(json.dumps(computation))
        new_ops = reassociated["scalar_body"]["operations"]
        new_ops[i]["operands"] = [b, c]
        new_ops[i + 1]["operands"] = [a, ref]
        yield reassociated, f"reassociate_modular_integer_add:{i}:{i + 1}"


def _variants(computation: dict[str, Any], limit: int) -> tuple[list[tuple[dict, list[str]]], bool]:
    """Saturate a bounded set of exact integer scalar equivalences."""
    variants: list[tuple[dict, list[str]]] = [(computation, [])]
    seen = {_digest(computation)}
    cursor = 0
    while cursor < len(variants):
        variant, rewrites = variants[cursor]
        for rewritten, label in _one_step_rewrites(variant):
            digest = _digest(rewritten)
            if digest in seen:
                continue
            if len(variants) >= max(1, limit):
                return variants, True
            seen.add(digest)
            variants.append((rewritten, [*rewrites, label]))
        cursor += 1
    return variants, False


def _types_for(op: Mapping[str, Any], values: Mapping[str, dict]) -> tuple[list[str], list[str]]:
    return [values[value]["type"] for value in op["operands"]], [result["type"] for result in op["results"]]


def _parameter_match(op: Mapping[str, Any], instruction: Mapping[str, Any]) -> bool:
    requirements = instruction.get("parameters", {})
    bindings = op.get("parameters", {})
    if not isinstance(requirements, Mapping) or not isinstance(bindings, Mapping):
        return False
    for name, bounds in requirements.items():
        value = bindings.get(name)
        if type(value) is not int or not isinstance(bounds, Mapping):
            return False
        if "min" in bounds and value < bounds["min"]:
            return False
        if "max" in bounds and value > bounds["max"]:
            return False
        if "multiple_of" in bounds and (not _positive_int(bounds["multiple_of"]) or value % bounds["multiple_of"]):
            return False
        if "choices" in bounds and value not in bounds["choices"]:
            return False
    return True


def _matching_choices(op: Mapping[str, Any], values: Mapping[str, dict], instructions: list[dict], limit: int):
    computation = _computation(op)
    if computation is None:
        return [], "operation has unknown effects/body/maps or is outside static linalg.generic subset", False
    variants, truncated = _variants(computation, limit)
    operand_types, result_types = _types_for(op, values)
    choices = []
    for instruction in sorted(instructions, key=lambda item: item["id"]):
        if instruction.get("status") == "UNKNOWN":
            continue
        if [item.get("type") for item in instruction.get("operands", [])] != operand_types:
            continue
        if [item.get("type") for item in instruction.get("results", [])] != result_types:
            continue
        if not _parameter_match(op, instruction):
            continue
        target_computation = instruction.get("computation")
        if not isinstance(target_computation, Mapping) or target_computation.get("status") == "UNKNOWN":
            continue
        # Explicit-vs-derived indexing maps is provenance. Their exact canonical
        # map strings are the semantic constraint in this v1 matcher.
        target_computation = {
            key: value for key, value in target_computation.items() if key != "indexing_maps_explicit"
        }
        for variant, rewrites in variants:
            if target_computation == variant:
                choices.append({"instruction": instruction, "rewrites": rewrites})
                break
    return choices, None if choices else "no exact typed instruction pattern matched", truncated


def _unknown_kernel_semantics(op: Mapping[str, Any]) -> bool:
    if op.get("operation") != "linalg.generic":
        return False
    body = op.get("scalar_body")
    return (
        op.get("indexing_maps") is None
        or op.get("iterator_types") is None
        or op.get("effects") is None
        or not isinstance(body, Mapping)
        or body.get("effects") is None
        or any(isinstance(inner, Mapping) and inner.get("effects") is None for inner in body.get("operations", []))
    )


def _allocation(candidate: tuple[dict, ...], kernel: dict, spaces: Mapping[str, Any], timeout_ms: int) -> dict:
    """Solve addresses for each value assigned to a selected local memory space."""
    value_spaces: dict[str, str] = {}
    access_groups: list[tuple[str, list[str]]] = []
    for op, choice in zip(kernel["operations"], candidate, strict=True):
        instruction = choice["instruction"]
        constraints = instruction.get("constraints", {})
        if not isinstance(constraints, Mapping):
            return {"status": "unknown", "reason": "malformed instruction constraints"}
        names = {}
        required_effects: list[tuple[str, str, str]] = []
        for field, bindings, sources in (
            ("operand_spaces", instruction.get("operands", []), op["operands"]),
            ("result_spaces", instruction.get("results", []), [result["id"] for result in op["results"]]),
        ):
            declared = constraints.get(field, {})
            if not isinstance(declared, Mapping):
                return {"status": "unknown", "reason": f"malformed {field}"}
            for formal, value_id in zip(bindings, sources, strict=True):
                name = formal["name"]
                names[name] = value_id
                space = declared.get(name)
                if space is None:
                    continue
                if not isinstance(space, str) or value_id in value_spaces and value_spaces[value_id] != space:
                    return {"status": "unsupported", "reason": f"value {value_id} needs a transfer between spaces"}
                value_spaces[value_id] = space
                required_effects.append(("read" if field == "operand_spaces" else "write", name, value_id))
        raw_banks = constraints.get("distinct_banks", [])
        if not isinstance(raw_banks, list):
            return {"status": "unknown", "reason": "invalid distinct_banks declaration"}
        groups = [raw_banks] if raw_banks and all(isinstance(name, str) for name in raw_banks) else raw_banks
        for group in groups:
            if not isinstance(group, list) or any(name not in names for name in group):
                return {"status": "unknown", "reason": "invalid distinct_banks group"}
            access_groups.append((op["id"], [names[name] for name in group]))
        effects = instruction.get("effects")
        if not isinstance(effects, list):
            return {"status": "unknown", "reason": "instruction memory effects are unknown"}
        effect_ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for effect in effects:
            if not isinstance(effect, Mapping) or effect.get("kind") not in {"read", "write"}:
                return {"status": "unknown", "reason": "unknown instruction memory effect"}
            name = effect.get("operand", effect.get("result"))
            if name not in names:
                return {"status": "unknown", "reason": "memory effect lacks a bound value"}
            value_id = names[name]
            if value_spaces.get(value_id) != effect.get("space"):
                return {"status": "unknown", "reason": "memory effect space disagrees with binding"}
            range_ = effect.get("range")
            size = kernel["values"][value_id].get("size_bytes")
            if (
                not isinstance(range_, Mapping)
                or type(range_.get("offset_bytes")) is not int
                or type(range_.get("length_bytes")) is not int
                or not _positive_int(size)
            ):
                return {"status": "unknown", "reason": "unresolved static memory effect range"}
            if (
                range_["offset_bytes"] < 0
                or range_["length_bytes"] < 1
                or range_["offset_bytes"] + range_["length_bytes"] > size
            ):
                return {"status": "unsupported", "reason": "memory effect exceeds value bounds"}
            effect_ranges.setdefault((effect["kind"], name), []).append(
                (range_["offset_bytes"], range_["offset_bytes"] + range_["length_bytes"])
            )
        for kind, name, value_id in required_effects:
            cursor = 0
            for lo, hi in sorted(effect_ranges.get((kind, name), [])):
                if lo > cursor:
                    break
                cursor = max(cursor, hi)
            if cursor < kernel["values"][value_id]["size_bytes"]:
                return {"status": "unknown", "reason": f"{kind} effect for {name!r} does not cover its tensor"}
    if not value_spaces:
        return {"status": "sat", "addresses": {}, "scope": "no local buffers requested"}
    try:
        import z3
    except ImportError:
        return {"status": "unavailable", "reason": "z3-solver is needed for memory allocation"}
    resolved_spaces = {}
    for name in sorted(set(value_spaces.values())):
        fact = spaces.get(name)
        if not isinstance(fact, Mapping):
            return {"status": "unknown", "reason": f"memory facts for {name!r} are absent"}
        required = ("capacity_bytes", "alignment_bytes", "banks", "bank_width_bytes")
        if any(not _positive_int(fact.get(key)) for key in required):
            return {"status": "unknown", "reason": f"memory facts for {name!r} are incomplete"}
        resolved_spaces[name] = fact
    addresses = {value: z3.Int(f"addr_{i}") for i, value in enumerate(sorted(value_spaces))}
    solver = z3.Optimize()
    solver.set(timeout=max(1, timeout_ms))
    solver.set(priority="lex")
    for value, address in addresses.items():
        space = resolved_spaces[value_spaces[value]]
        size = kernel["values"][value].get("size_bytes")
        if not _positive_int(size):
            return {"status": "unknown", "reason": f"size of {value!r} is unknown"}
        solver.add(address >= 0, address + size <= space["capacity_bytes"])
        solver.add(address % space["alignment_bytes"] == 0)
        solver.minimize(address)
    first, last = {}, {}
    for index, op in enumerate(kernel["operations"]):
        for value in op["operands"]:
            first.setdefault(value, -1)
            last[value] = index
        for result in op["results"]:
            first[result["id"]] = index
            last[result["id"]] = index
    for value in kernel["outputs"]:
        last[value] = len(kernel["operations"])
    allocated = sorted(addresses)
    for i, lhs in enumerate(allocated):
        for rhs in allocated[i + 1 :]:
            if value_spaces[lhs] != value_spaces[rhs] or last[lhs] < first[rhs] or last[rhs] < first[lhs]:
                continue
            solver.add(
                z3.Or(
                    addresses[lhs] + kernel["values"][lhs]["size_bytes"] <= addresses[rhs],
                    addresses[rhs] + kernel["values"][rhs]["size_bytes"] <= addresses[lhs],
                )
            )
    for operation_id, group in access_groups:
        if len(set(group)) != len(group):
            return {"status": "unsupported", "reason": f"bank group in {operation_id} repeats one value"}
        for lhs, rhs in itertools.combinations(group, 2):
            if lhs not in addresses or rhs not in addresses or value_spaces[lhs] != value_spaces[rhs]:
                return {
                    "status": "unknown",
                    "reason": f"bank group in {operation_id} has unallocated or mixed-space values",
                }
            space = resolved_spaces[value_spaces[lhs]]
            if space["banks"] < 2:
                return {
                    "status": "unsupported",
                    "reason": f"{operation_id} requires distinct banks in a single-bank space",
                }
            width, banks = space["bank_width_bytes"], space["banks"]
            solver.add((addresses[lhs] / width) % banks != (addresses[rhs] / width) % banks)
    outcome = solver.check()
    if outcome == z3.unsat:
        return {"status": "unsat", "reason": "capacity, liveness, alignment or bank constraints are infeasible"}
    if outcome != z3.sat:
        return {"status": "unknown", "reason": f"z3 returned {solver.reason_unknown()}"}
    model = solver.model()
    return {
        "status": "sat",
        "addresses": {
            value: {
                "space": value_spaces[value],
                "address_bytes": model.eval(address).as_long(),
                "size_bytes": kernel["values"][value]["size_bytes"],
            }
            for value, address in addresses.items()
        },
        "scope": "base addresses; conservative closed SSA lifetimes; bank constraints apply to base bank",
    }


def search(
    kernel_record: Mapping[str, Any], instruction_model: Mapping[str, Any], *, limits: SearchLimits = SearchLimits()
) -> dict[str, Any]:
    """Return a JSON receipt and selected instruction graph when a bounded search succeeds.

    The receipt proves only exact pattern compatibility and modeled allocation.
    It does not assert numerical equivalence, target encoding, or execution.
    """
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "unknown",
        "scope": "bounded exact-pattern static linalg search; selection is not execution proof",
        "limits": asdict(limits),
        "rewrite_rules": [dict(rule) for rule in _RULES],
        "variants_truncated": False,
        "search_truncated": False,
        "allocation_assumptions": [
            "only declared local value buffers are allocated",
            "external operands are assumed available to the target compiler",
            "value lifetimes overlap at their producing and consuming operation",
            "bank constraints use the base address bank",
        ],
        "kernel_sha256": _digest(kernel_record),
        "instruction_model_sha256": _digest(instruction_model),
        "attempts": [],
    }
    try:
        if instruction_model.get("schema") != INSTRUCTION_SCHEMA:
            raise ValueError("instruction model schema is not merlin.instruction_semantics.v1")
        if "canonical_sha256" in instruction_model:
            from merlin.target.spec.instructions import validate_normalized_instruction_model

            validate_normalized_instruction_model(instruction_model)
        if instruction_model.get("status") == "UNKNOWN":
            receipt.update(
                status="unknown", reason="selected instruction model contains unresolved semantic or input facts"
            )
            return receipt
        kernel = _normalize_kernel(kernel_record)
        operations = kernel["operations"]
        if len(operations) > limits.max_operations:
            receipt.update(status="unknown", reason="operation limit exceeded")
            return receipt
        instructions = instruction_model.get("instructions")
        if not isinstance(instructions, list) or any(
            not isinstance(item, Mapping) or not isinstance(item.get("id"), str) for item in instructions
        ):
            raise ValueError("instruction model has malformed instruction rows")
        shared_unknowns = instruction_model.get("unknowns", [])
        if any(
            str(item).startswith("selected_software_spec_") or str(item).startswith("selected_circt_facts_")
            for item in shared_unknowns
        ):
            receipt.update(
                status="unknown", reason="selected software specification or CIRCT facts are missing or unreviewed"
            )
            return receipt
        choice_lists = []
        truncated_variants = False
        for op in operations:
            choices, reason, truncated = _matching_choices(
                op, kernel["values"], instructions, limits.max_variants_per_operation
            )
            truncated_variants |= truncated
            if not choices:
                unknown_instruction = any(
                    item.get("status") == "UNKNOWN"
                    or isinstance(item.get("computation"), Mapping)
                    and item["computation"].get("status") == "UNKNOWN"
                    for item in instructions
                )
                receipt.update(
                    status="unknown"
                    if unknown_instruction or truncated or _unknown_kernel_semantics(op)
                    else "unsupported",
                    reason=f"{op['id']}: {reason}",
                    variants_truncated=truncated_variants,
                )
                return receipt
            choice_lists.append(choices)
        receipt["variants_truncated"] = truncated_variants
        considered = 0
        unresolved: list[dict[str, str]] = []
        spaces = instruction_model.get("memory_spaces", {})
        if not isinstance(spaces, Mapping):
            raise ValueError("instruction model memory_spaces must be a mapping")
        for candidate in itertools.product(*choice_lists):
            if considered >= limits.max_candidates or (time.monotonic() - started) * 1000 >= limits.timeout_ms:
                receipt.update(
                    status="unknown",
                    reason="candidate or wall-clock search limit exhausted",
                    candidates_considered=considered,
                    search_truncated=True,
                )
                return receipt
            considered += 1
            remaining = limits.timeout_ms - int((time.monotonic() - started) * 1000)
            allocated = _allocation(candidate, kernel, spaces, min(limits.solver_timeout_ms, max(1, remaining)))
            graph = [
                {
                    "operation_id": op["id"],
                    "instruction_id": choice["instruction"]["id"],
                    "rewrites": choice["rewrites"],
                }
                for op, choice in zip(operations, candidate, strict=True)
            ]
            receipt["attempts"].append(
                {"graph": graph, "allocation_status": allocated["status"], "reason": allocated.get("reason")}
            )
            if allocated["status"] == "sat":
                if truncated_variants:
                    # A found solution is enough; truncation only limits negative claims.
                    receipt["search_truncated"] = True
                receipt.update(
                    status="selected", instruction_graph=graph, allocation=allocated, candidates_considered=considered
                )
                return receipt
            if allocated["status"] in {"unknown", "unavailable"}:
                unresolved.append({"status": allocated["status"], "reason": allocated["reason"]})
        if unresolved:
            receipt.update(
                status="unknown" if any(item["status"] == "unknown" for item in unresolved) else "unavailable",
                reason="some instruction graphs could not be fully evaluated",
                unresolved=unresolved,
                candidates_considered=considered,
            )
            return receipt
        receipt.update(
            status="unknown" if truncated_variants else "unsupported",
            reason="all enumerated instruction graphs fail modeled allocation"
            if not truncated_variants
            else "rewritten variants were truncated",
            candidates_considered=considered,
        )
        return receipt
    except (TypeError, ValueError, KeyError) as exc:
        receipt.update(status="refused", reason=str(exc))
        return receipt


def kernel_from_linalg_inventory(parsed: Mapping[str, Any], *, operation_id: int) -> dict[str, Any]:
    """Select one exact payload operation from ``parse_linalg_mlir`` output.

    The capture reader records op-to-op edges, but this wrapper intentionally
    keeps preceding producers as typed external inputs. It is a per-operation
    kernel probe; whole-model joined scheduling belongs to the outliner.
    """
    rows = parsed.get("ops")
    if not isinstance(rows, list):
        raise ValueError("linalg inventory has no ops list")
    matching = [row for row in rows if isinstance(row, Mapping) and row.get("id") == operation_id]
    if len(matching) != 1:
        raise ValueError(f"operation id {operation_id!r} is absent or ambiguous")
    kernel = _from_frontend_op(matching[0])
    kernel["source"] = {
        "entry": parsed.get("entry"),
        "operation_id": operation_id,
        "provenance": matching[0].get("source_provenance", matching[0].get("prov", {})),
        "operand_sources": [item.get("source") for item in matching[0].get("ins", []) + matching[0].get("outs", [])],
    }
    return kernel


def search_linalg_inventory(
    parsed: Mapping[str, Any], instruction_model: Mapping[str, Any], *, limits: SearchLimits = SearchLimits()
) -> dict[str, Any]:
    """Return a per-payload-op selection census over parsed linalg MLIR."""
    rows = parsed.get("ops")
    if not isinstance(rows, list):
        raise ValueError("linalg inventory has no ops list")
    regions = []
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("id")) is not int:
            raise ValueError("linalg inventory has malformed operation ids")
        try:
            kernel = kernel_from_linalg_inventory(parsed, operation_id=row["id"])
        except ValueError:
            # Keep a per-operation refusal in the census when one operation has
            # an unknown type; other captured operations remain inspectable.
            receipt = search(row, instruction_model, limits=limits)
        else:
            receipt = search(kernel, instruction_model, limits=limits)
        regions.append(
            {
                "operation_id": row["id"],
                "operation": row.get("operation", row.get("kind")),
                "source_provenance": row.get("source_provenance", row.get("prov", {})),
                "placement": "device_candidate" if receipt["status"] == "selected" else "unresolved",
                "host_admission": "not_evaluated",
                "receipt": receipt,
            }
        )
    by_operation: dict[str, dict[str, int]] = {}
    for region in regions:
        name = str(region["operation"])
        status = region["receipt"]["status"]
        counts = by_operation.setdefault(name, {})
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema": "merlin.semantic_search.inventory.v1",
        "entry": parsed.get("entry"),
        "summary": {
            "operations": len(regions),
            "device_candidates": sum(region["placement"] == "device_candidate" for region in regions),
            "unresolved": sum(region["placement"] == "unresolved" for region in regions),
            "by_operation": {name: dict(sorted(counts.items())) for name, counts in sorted(by_operation.items())},
        },
        "regions": regions,
        "scope": "per-payload-op matching; device candidates require target compilation and execution",
    }

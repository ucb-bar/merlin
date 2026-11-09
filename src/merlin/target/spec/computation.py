"""Typed scalar bodies and linalg computation records, without execution claims."""

from __future__ import annotations

from typing import Any

UNKNOWN = "UNKNOWN"


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _names(rows: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "type"}:
            raise ValueError(f"{label}[{index}] requires only name and type")
        name = _nonempty(row["name"], f"{label}[{index}].name")
        typ = _nonempty(row["type"], f"{label}[{index}].type")
        if name in seen:
            raise ValueError(f"{label}: duplicate name {name!r}")
        seen.add(name)
        result.append({"name": name, "type": typ})
    return result


def _scalar_body(body: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a mapping")
    arguments = body.get("arguments")
    captures = body.get("captures", [])
    operations = body.get("operations")
    yields = body.get("yields")
    if not isinstance(arguments, list) or any(not isinstance(t, str) or not t for t in arguments):
        raise ValueError(f"{label}.arguments must be MLIR type strings")
    if not isinstance(captures, list):
        raise ValueError(f"{label}.captures must be a list")
    normalized_captures = []
    for index, capture in enumerate(captures):
        site = f"{label}.captures[{index}]"
        if not isinstance(capture, dict) or set(capture) - {"source", "shape", "dtype", "result_index", "const_value"}:
            raise ValueError(f"{site} has unsupported fields")
        source = capture.get("source")
        shape = capture.get("shape")
        if (
            not isinstance(source, (list, tuple))
            or len(source) != 2
            or source[0] not in {"arg", "op", "init", "const", "other"}
            or not isinstance(source[1], (int, str))
        ):
            raise ValueError(f"{site}.source must identify a captured value")
        if not isinstance(shape, list) or any(type(dim) is not int for dim in shape):
            raise ValueError(f"{site}.shape must be integer dimensions")
        dtype = _nonempty(capture.get("dtype"), f"{site}.dtype")
        result_index = capture.get("result_index")
        if result_index is not None and (type(result_index) is not int or result_index < 0):
            raise ValueError(f"{site}.result_index must be nonnegative")
        const_value = capture.get("const_value")
        if "const_value" in capture and (type(const_value) not in (int, float) or isinstance(const_value, bool)):
            raise ValueError(f"{site}.const_value must be numeric")
        normalized_capture = {"source": list(source), "shape": list(shape), "dtype": dtype}
        if result_index is not None:
            normalized_capture["result_index"] = result_index
        if "const_value" in capture:
            normalized_capture["const_value"] = const_value
        normalized_captures.append(normalized_capture)
    if not isinstance(operations, list) or not isinstance(yields, list):
        raise ValueError(f"{label} requires operations and yields lists")
    available = {f"arg:{index}" for index in range(len(arguments))}
    available.update(f"capture:{index}" for index in range(len(captures)))
    normalized_ops = []
    for index, operation in enumerate(operations):
        site = f"{label}.operations[{index}]"
        if not isinstance(operation, dict):
            raise ValueError(f"{site} must be a mapping")
        op = _nonempty(operation.get("op"), f"{site}.op")
        operands = operation.get("operands")
        results = operation.get("results")
        attrs = operation.get("attributes", {})
        regions = operation.get("regions", 0)
        if not isinstance(operands, list) or any(ref not in available for ref in operands):
            raise ValueError(f"{site}.operands must reference preceding scalar values")
        if not isinstance(results, list) or any(not isinstance(t, str) or not t for t in results):
            raise ValueError(f"{site}.results must be MLIR type strings")
        if not isinstance(attrs, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in attrs.items()
        ):
            raise ValueError(f"{site}.attributes must map names to MLIR attribute strings")
        if type(regions) is not int or regions < 0:
            raise ValueError(f"{site}.regions must be a nonnegative integer")
        effects = operation.get("effects")
        if effects is not None and (
            not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
        ):
            raise ValueError(f"{site}.effects must be a string list or null")
        if effects is None:
            unknowns.append(f"{site}.effects")
        if regions:
            unknowns.append(f"{site}.nested_regions")
        normalized_ops.append(
            {
                "op": op,
                "operands": list(operands),
                "results": list(results),
                "attributes": dict(sorted(attrs.items())),
                "effects": effects,
                "regions": regions,
            }
        )
        available.update(f"op:{index}:{result}" for result in range(len(results)))
    if any(ref not in available for ref in yields):
        raise ValueError(f"{label}.yields must reference scalar values")
    effects = body.get("effects")
    if effects is not None and (
        not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
    ):
        raise ValueError(f"{label}.effects must be a string list or null")
    if effects is None:
        unknowns.append(f"{label}.effects")
    return {
        "arguments": list(arguments),
        "captures": normalized_captures,
        "operations": normalized_ops,
        "yields": list(yields),
        "effects": effects,
    }


def _computation(value: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if value is None:
        unknowns.append(f"{label}.missing")
        return {"status": UNKNOWN, "reason": "instruction computation was not described"}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if value.get("status") == UNKNOWN:
        reason = _nonempty(value.get("reason"), f"{label}.reason")
        unknowns.append(f"{label}.{reason}")
        return {"status": UNKNOWN, "reason": reason}
    kind = _nonempty(value.get("kind"), f"{label}.kind")
    if kind != "linalg.generic":
        raise ValueError(f"{label}.kind {kind!r} is not described by v1")
    maps = value.get("indexing_maps")
    maps_explicit = value.get("indexing_maps_explicit")
    iterators = value.get("iterator_types")
    if maps is not None and (not isinstance(maps, list) or any(not isinstance(item, str) or not item for item in maps)):
        raise ValueError(f"{label}.indexing_maps must be MLIR attribute strings or null")
    if iterators is not None and (
        not isinstance(iterators, list) or any(item not in {"parallel", "reduction"} for item in iterators)
    ):
        raise ValueError(f"{label}.iterator_types must be parallel/reduction strings or null")
    if maps is None:
        unknowns.append(f"{label}.indexing_maps")
    if maps_explicit is not None and type(maps_explicit) is not bool:
        raise ValueError(f"{label}.indexing_maps_explicit must be Boolean or null")
    if iterators is None:
        unknowns.append(f"{label}.iterator_types")
    return {
        "kind": kind,
        "indexing_maps": maps,
        "indexing_maps_explicit": maps_explicit,
        "iterator_types": iterators,
        "scalar_body": _scalar_body(value.get("scalar_body"), label=f"{label}.scalar_body", unknowns=unknowns),
    }

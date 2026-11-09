"""Reconcile staged invocation and lifted tensor bindings with parsed IR."""
from __future__ import annotations

import json
from io import BytesIO

from xdsl.dialects.builtin import TensorType

from merlin.capture.safetensors import read_header
from merlin.compiler.frontend.archives import _ArchiveInfo
from merlin.compiler.frontend.constants import CONSTANT_FILE

_WEIGHT_DTYPES = {
    "float32": ("f32", "F32", 4),
    "float16": ("f16", "F16", 2),
    "bfloat16": ("bf16", "BF16", 2),
    "float64": ("f64", "F64", 8),
    "int64": ("i64", "I64", 8),
    "int32": ("i32", "I32", 4),
    "int16": ("i16", "I16", 2),
    "int8": ("i8", "I8", 1),
    "uint8": ("ui8", "U8", 1),
    "bool": ("i1", "BOOL", 1),
}

def _entry_type(module, entry: str):
    functions = [
        op for op in module.walk()
        if op.name == "func.func" and op.sym_name.data == entry
    ]
    if len(functions) != 1:
        raise ValueError("compiler input signature does not name exactly one parsed function")
    return functions[0].function_type


def _tensor_abi(type_) -> dict:
    if not isinstance(type_, TensorType) or not type_.has_static_shape():
        raise ValueError("compiler input signature needs static ranked tensor arguments")
    return {"shape": list(type_.get_shape()), "dtype": str(type_.element_type)}


def _weight_header(raw: bytes | _ArchiveInfo) -> dict:
    header = raw.header if isinstance(raw, _ArchiveInfo) else read_header(BytesIO(raw))[0]
    return {name: row for name, row in header.items() if name != "__metadata__"}


def _payload_bytes(raw: bytes | _ArchiveInfo) -> int:
    return raw.file_bytes - raw.payload_start if isinstance(raw, _ArchiveInfo) else (
        len(raw) - 8 - int.from_bytes(raw[:8], "little")
    )


def _argument_bindings(
    function_type, runtime_inputs: list, members: dict[str, bytes | _ArchiveInfo], *, materialized: bool = False,
) -> list[dict]:
    """Join captured argument indices to runtime inputs or exact weight tensors."""
    manifest = json.loads(members["weights.safetensors.manifest.json"])
    order_raw = members.get("input_order.json")
    order = json.loads(order_raw) if order_raw is not None else None
    types = list(function_type.inputs.data)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {str(index) for index in range(len(types))}
        or not isinstance(order, dict)
        or len(order) != len(runtime_inputs)
        or any(type(index) is not int for index in order.values())
        or set(order.values()) != set(range(len(runtime_inputs)))
    ):
        raise ValueError("weight manifest or input order does not cover the parsed arguments")
    files = ("weights.safetensors", CONSTANT_FILE) if materialized else ("weights.safetensors",)
    if (CONSTANT_FILE in members) != materialized:
        raise ValueError("capture constant archive differs from signature schema")
    headers = {name: _weight_header(members[name]) for name in files}
    roles = {"param", "buffer", "constant"} if materialized else {"param", "buffer"}
    fields = {"kind", "weight", "dtype", "shape"} | ({"file"} if materialized else set())
    bindings = []
    used_inputs: set[int] = set()
    used_weights = {name: set() for name in files}
    ranges = {name: [] for name in files}
    payload_lengths = {name: _payload_bytes(members[name]) for name in files}
    for index, type_ in enumerate(types):
        row = manifest[str(index)]
        actual = _tensor_abi(type_)
        if not isinstance(row, dict):
            raise ValueError("weight manifest has a malformed argument row")
        kind = row.get("kind")
        if kind == "input":
            name = row.get("name")
            if set(row) != {"kind", "name"} or not isinstance(name, str) or name not in order:
                raise ValueError("weight manifest input has no declared runtime position")
            ordinal = order[name]
            if runtime_inputs[ordinal] != actual:
                raise ValueError("runtime input argument differs from parsed function")
            used_inputs.add(ordinal)
            bindings.append({"kind": "input", "name": name, "input_index": ordinal, **actual})
            continue
        name = row.get("weight")
        file = row.get("file") if materialized else "weights.safetensors"
        dtype_name = row.get("dtype")
        dtype = _WEIGHT_DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        if (
            not isinstance(kind, str) or kind not in roles or set(row) != fields
            or not isinstance(file, str) or file not in files
            or kind == "param" and file != "weights.safetensors"
            or kind == "constant" and file != CONSTANT_FILE
            or not isinstance(name, str) or not name
            or dtype is None
            or not isinstance(row.get("shape"), list)
            or any(type(extent) is not int or extent < 0 for extent in row["shape"])
            or row["shape"] != actual["shape"]
            or dtype[0] != actual["dtype"]
        ):
            raise ValueError("weight argument shape or dtype differs from parsed function")
        stored = headers[file].get(name)
        if not isinstance(stored, dict) or stored.get("dtype") != dtype[1] or stored.get("shape") != actual["shape"]:
            raise ValueError("weight safetensors shape or dtype differs from manifest")
        offsets = stored.get("data_offsets")
        elements = 1
        for extent in actual["shape"]:
            elements *= extent
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != elements * dtype[2]
            or offsets[1] > payload_lengths[file]
        ):
            raise ValueError("weight safetensors payload differs from declared shape")
        if name not in used_weights[file]:
            ranges[file].append((offsets[0], offsets[1]))
        used_weights[file].add(name)
        bindings.append({"kind": row["kind"], "weight": name, **actual, **({"file": file} if materialized else {})})
    if used_inputs != set(range(len(runtime_inputs))) or any(
        used_weights[file] != set(headers[file]) for file in files
    ):
        raise ValueError("weight manifest leaves a runtime input or stored tensor unbound")
    for file in files:
        cursor = 0
        for start, end in sorted(ranges[file]):
            if start != cursor:
                raise ValueError("weight safetensors payload has an overlap or gap")
            cursor = end
        if cursor != payload_lengths[file]:
            raise ValueError("weight safetensors payload has undeclared bytes")
    return bindings


def _check_signature(module, signature: dict, members: dict[str, bytes]) -> None:
    """Reconcile the declared invocation ABI with the parsed entry function.

    The capture receipt authenticates bytes, but it cannot establish that its
    separately produced ABI describes the IR. Check this again when consuming
    a staged directory because its manifest is caller-owned data.
    """
    if not isinstance(signature, dict) or signature.get("schema") not in {
        "merlin.compile-signature.v1", "merlin.compile-signature.v2", "merlin.compile-signature.v3",
    }:
        raise ValueError("compiler input signature has an unsupported schema")
    entry = signature.get("entry")
    if not isinstance(entry, str) or not entry:
        raise ValueError("compiler input signature lacks an entry function")
    function_type = _entry_type(module, entry)
    if signature["schema"] != "merlin.compile-signature.v1":
        rows = signature.get("inputs")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("compiler input signature has invalid runtime inputs")
        expected = _argument_bindings(
            function_type, rows, members, materialized=signature["schema"] == "merlin.compile-signature.v3",
        )
        if signature.get("arguments") != expected:
            raise ValueError("compiler input signature arguments differ from parsed function")
    else:
        weight_manifest = json.loads(members["weights.safetensors.manifest.json"])
        if not isinstance(weight_manifest, dict):
            raise ValueError("compiler input weight manifest is not a mapping")
        if weight_manifest or CONSTANT_FILE in members:
            raise ValueError("legacy compiler input signature cannot bind externalized weights")
    directions = [("outputs", function_type.outputs.data)]
    if signature["schema"] == "merlin.compile-signature.v1":
        directions.insert(0, ("inputs", function_type.inputs.data))
    for direction, types in directions:
        rows = signature.get(direction)
        if not isinstance(rows, list) or len(rows) != len(types):
            raise ValueError(f"compiler input signature {direction} differ from parsed function")
        for row, type_ in zip(rows, types):
            if (
                not isinstance(row, dict)
                or set(row) != {"shape", "dtype"}
                or not isinstance(row["shape"], list)
                or any(type(size) is not int or size < 0 for size in row["shape"])
                or not isinstance(row["dtype"], str)
                or row != _tensor_abi(type_)
            ):
                raise ValueError(f"compiler input signature {direction} differ from parsed function")


"""Admit exact exported constant arguments without evaluator archive access."""

import json
from math import prod

from xdsl.dialects.builtin import TensorType
from xdsl.dialects.func import FuncOp

from merlin.runtime.captured_constants import CaptureConstantTable

_TYPES = {"float16": ("f16", "f16", 2), "bfloat16": ("bf16", "bf16", 2),
          "float32": ("f32", "f32", 4), "float64": ("f64", "f64", 8),
          "int8": ("i8", "i8", 1), "int16": ("i16", "i16", 2),
          "int32": ("i32", "i32", 4), "int64": ("i64", "i64", 8),
          "uint8": ("i8", "ui8", 1), "bool": ("i1", "i1", 1)}


def checked_compiler_constants(receipt, manifest_bytes, weight_bytes, module, trace, *, entry_name="forward"):
    """Return a compiler projection after source/ABI/storage correspondence checks.

    This grants byte custody only. Numeric/effect correctness, mutable state,
    physical ownership and any packing transformation retain their own gates.
    """
    lifted, profile = receipt.get("lifted_constants") or {}, receipt.get("compiler_constants")
    manifest = json.loads(manifest_bytes)
    constants = {int(key): value for key, value in manifest.items() if value.get("kind") == "constant"}
    if not lifted and profile is None and not constants:
        return None
    if (not isinstance(lifted, dict) or not isinstance(profile, dict)
            or set(profile) != {"schema", "storage", "bindings"}
            or profile.get("schema") != "m2m.compiler-constants.v1" or profile.get("storage") != "weights.safetensors"
            or not isinstance(profile.get("bindings"), list) or not profile["bindings"]):
        raise ValueError("unbound lifted constants require a complete compiler constant profile")
    entries = [op for op in module.ops if isinstance(op, FuncOp) and op.sym_name.data == entry_name]
    prepared = trace.get("graphs", {}).get("prepared", {})
    nodes = [node for node in prepared.get("nodes", [])
             if node.get("op") == "placeholder" and node.get("graph_id") == prepared.get("graph_id")]
    if len(entries) != 1 or len(nodes) != len(entries[0].function_type.inputs):
        raise ValueError("compiler constants require the actual complete source argument roster")
    table = CaptureConstantTable(manifest_bytes, weight_bytes,
        receipt["artifacts"]["weights.safetensors.manifest.json"]["sha256"], receipt["artifacts"]["weights.safetensors"]["sha256"])
    indices, names = set(), set()
    for row in profile["bindings"]:
        if (not isinstance(row, dict) or set(row) != {"argument", "name", "target", "weight", "dtype", "shape", "sha256"}
                or type(row["argument"]) is not int or not 0 <= row["argument"] < len(nodes)
                or not isinstance(row["name"], str) or row["name"] in names or row["argument"] in indices
                or not isinstance(row["dtype"], str) or row["dtype"] not in _TYPES or row["name"] not in lifted):
            raise ValueError("compiler constant bindings must be unique complete typed source arguments")
        index, name = row["argument"], row["name"]
        entry, node = constants.get(index), nodes[index]
        if (entry is None or any(entry.get(key) != row[field] for key, field in
                (("name", "name"), ("source_target", "target"), ("weight", "weight"), ("dtype", "dtype"), ("shape", "shape"), ("sha256", "sha256")))
                or node.get("input_kind") != "CONSTANT_TENSOR" or node.get("target") != name
                or node.get("input_target") != row["target"]):
            raise ValueError("compiler constant binding differs from its exported source/manifest")
        results = node.get("results", [])
        type_ = entries[0].function_type.inputs.data[index]
        logical, storage, itemsize = _TYPES[row["dtype"]]
        if (len(results) != 1 or results[0].get("kind") != "tensor" or results[0].get("dtype") != row["dtype"]
                or results[0].get("shape") != row["shape"] or not isinstance(type_, TensorType)
                or list(type_.get_shape()) != row["shape"] or str(type_.element_type) != logical
                or lifted[name].get("shape") != row["shape"]):
            raise ValueError("compiler constant shape/type differs from its actual source argument")
        value = table.read(entry_argument_index=index, source_shape=row["shape"], source_dtype=storage,
                           max_payload_bytes=max(1, prod(row["shape"]) * itemsize))
        if value.payload_sha256 != row["sha256"]:
            raise ValueError("compiler constant bytes differ from their source declaration")
        indices.add(index)
        names.add(name)
    source_indices = {i for i, node in enumerate(nodes) if node.get("input_kind") == "CONSTANT_TENSOR"}
    if indices != set(constants) or indices != source_indices or names != set(lifted):
        raise ValueError("compiler constant profile omits an actual source constant argument")
    return {"schema": "merlin.compiler-constants.v1", "source_schema": profile["schema"],
            "storage": profile["storage"], "bindings": profile["bindings"],
            "scope": "source-bound constant bytes; numeric/effect and packing admission unverified"}

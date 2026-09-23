"""Frozen certificate workload identities and declared logical tensor byte encoding."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from . import corpus as P2_CORPUS
from . import gsim_gate as GATE

OUTPUT_ENCODING = GATE.OUTPUT_ENCODING


class ProducerError(RuntimeError):
    """The available bytes cannot support a strict v1 certificate."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_mapping(path: Path, *, yaml_input: bool = False) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_input else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ProducerError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProducerError(f"{path} does not contain a mapping")
    return value


def derive_workload(capsule_manifest: str | Path) -> dict[str, Any]:
    """Derive an exact workload identity only from the frozen capsule descriptor.

    Operand symbol names and source annotations are excluded: they do not alter the operation.  Shapes,
    dtypes, output semantics, epilogue, and numeric comparison policy remain load-bearing.
    """
    path = Path(capsule_manifest)
    doc = _load_mapping(path, yaml_input=True)
    operation = doc.get("operation")
    inputs = doc.get("inputs")
    numeric = doc.get("numeric_policy")
    if not isinstance(operation, Mapping) or not isinstance(inputs, list) or not inputs:
        raise ProducerError(f"{path}: operation/inputs are absent")
    op = operation.get("op")
    attrs = operation.get("attributes")
    if not isinstance(op, str) or not op or not isinstance(attrs, Mapping):
        raise ProducerError(f"{path}: operation is malformed")
    if not isinstance(numeric, Mapping) or not numeric:
        raise ProducerError(f"{path}: numeric policy is absent")

    tensors: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise ProducerError(f"{path}: input {index} is malformed")
        name, shape, dtype = item.get("name"), item.get("shape"), item.get("dtype")
        if not isinstance(name, str) or not isinstance(shape, list) or not shape or not isinstance(dtype, str):
            raise ProducerError(f"{path}: input {index} lacks name/shape/dtype")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ProducerError(f"{path}: input {name} has a non-positive/non-integer shape")
        tensors[name] = item

    semantic_attrs = {
        str(key): value for key, value in attrs.items() if key not in ("lhs", "weight", "src", "out", "semantic")
    }
    if op == "matmul":
        lhs = tensors.get(str(attrs.get("lhs") or ""))
        weight = tensors.get(str(attrs.get("weight") or ""))
        if lhs is None or weight is None:
            raise ProducerError(f"{path}: matmul operands do not resolve to declared inputs")
        lhs_shape, weight_shape = lhs["shape"], weight["shape"]
        if len(lhs_shape) != 2 or len(weight_shape) != 2 or lhs_shape[1] != weight_shape[0]:
            raise ProducerError(f"{path}: matmul shapes are not MxK and KxN")
        shape = {"m": lhs_shape[0], "n": weight_shape[1], "k": lhs_shape[1]}
        operand_dtypes = {"lhs": lhs["dtype"], "weight": weight["dtype"]}
    elif op == "movement":
        src = tensors.get(str(attrs.get("src") or ""))
        if src is None:
            raise ProducerError(f"{path}: movement source does not resolve to a declared input")
        shape = {"dimensions": list(src["shape"])}
        operand_dtypes = {"src": src["dtype"]}
    else:
        # No guessed shape algebra for an unknown operation.  Exact named input roles/shapes are still a
        # valid envelope and are derived directly from the descriptor.
        shape = {"inputs": [{"role": item.get("role"), "shape": item["shape"]} for item in inputs]}
        operand_dtypes = {str(item.get("role") or item["name"]): item["dtype"] for item in inputs}
    semantics = {
        "operand_dtypes": operand_dtypes,
        "operation_attributes": semantic_attrs,
        "numeric_policy": dict(numeric),
    }
    return GATE.canonical_workload({"operation": op, "shape": shape, "semantics": semantics})


def derive_frozen_corpus_workloads(
    root: str | Path, *, manifest_sha256: str, capsules_sha256: str, expected_target: str
) -> dict[str, dict[str, Any]]:
    """Derive every workload after the existing frozen-corpus verifier re-hashes its bytes."""
    corpus = P2_CORPUS.load_frozen_performance_corpus(
        Path(root), manifest_sha256=manifest_sha256, capsules_sha256=capsules_sha256, expected_target=expected_target
    )
    workloads = {}
    for member in corpus.capsules:
        if member.capsule in workloads:
            raise ProducerError(f"frozen corpus has duplicate capsule {member.capsule!r}")
        workloads[member.capsule] = derive_workload(member.source_dir / "capsule.yaml")
    if not workloads:
        raise ProducerError("frozen corpus contains no workload")
    return workloads


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flat_values(item))
        return out
    return [value]


def _encode_scalar(value: Any, dtype: str) -> bytes:
    if len(dtype) > 1 and dtype[0] in ("i", "u") and dtype[1:].isdecimal():
        signed, bits = dtype[0] == "i", int(dtype[1:])
        if bits == 1:
            bits = 8
        if bits <= 0 or bits % 8:
            raise ProducerError(f"output dtype {dtype!r} has no byte-exact scalar encoding")
        try:
            return int(value).to_bytes(bits // 8, byteorder="little", signed=signed)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ProducerError(f"output value {value!r} is not representable as {dtype}") from exc
    if dtype in ("f16", "float16"):
        return struct.pack("<e", float(value))
    if dtype in ("f32", "float32"):
        return struct.pack("<f", float(value))
    if dtype in ("f64", "float64"):
        return struct.pack("<d", float(value))
    if dtype in ("bf16", "bfloat16"):
        # Round a parsed real value to IEEE bfloat16, ties-to-even, then emit its little-endian bits.
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
        return struct.pack("<H", rounded & 0xFFFF)
    raise ProducerError(f"output dtype {dtype!r} has no declared byte encoding")


def encode_declared_outputs(outputs: Any, command_buffer: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Encode parsed tensor values as the command buffer's exact logical little-endian bytes.

    This deliberately does not hash JSON text.  Shape and dtype come from the frozen command buffer,
    the flattened element count must match exactly, and each scalar is range-checked before encoding.
    The claim is logical tensor bytes; it is not a claim about padding or an engine's private memory.
    """
    if not isinstance(outputs, Mapping):
        raise ProducerError("simulator output is not a tensor mapping")
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ProducerError("command buffer has no tensor declarations")
    declarations = []
    for name, spec in tensors.items():
        if isinstance(spec, Mapping) and spec.get("role") == "output":
            declarations.append((str(name), spec))
    if not declarations:
        raise ProducerError("command buffer declares no output tensors")
    rows, aggregate = [], hashlib.sha256()
    for name, spec in sorted(declarations):
        shape, dtype = spec.get("shape"), spec.get("dtype")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            raise ProducerError(f"output tensor {name!r} has an invalid shape")
        if not isinstance(dtype, str) or not dtype:
            raise ProducerError(f"output tensor {name!r} has no dtype")
        if name not in outputs:
            raise ProducerError(f"simulator omitted declared output tensor {name!r}")
        flat = _flat_values(outputs[name])
        count = math.prod(shape)
        if len(flat) != count:
            raise ProducerError(f"output tensor {name!r} has {len(flat)} values; declaration requires {count}")
        raw = b"".join(_encode_scalar(value, dtype) for value in flat)
        identity = GATE.canonical_json({"name": name, "shape": shape, "dtype": dtype}).encode("utf-8")
        aggregate.update(len(identity).to_bytes(8, "little"))
        aggregate.update(identity)
        aggregate.update(len(raw).to_bytes(8, "little"))
        aggregate.update(raw)
        rows.append(
            {"name": name, "shape": list(shape), "dtype": dtype, "n_bytes": len(raw), "sha256": _sha_bytes(raw)}
        )
    return aggregate.hexdigest(), rows

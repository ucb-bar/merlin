"""Explicit native-storage movement/cast sources and typed capture publication."""

from __future__ import annotations

import hashlib
import json
import math
import struct

from merlin.common.digest import is_sha256

_FIELDS = ("native_input_dtype", "native_output_dtype", "native_source_abi")
_PAIRS = {"permute": {("i8", "i8"), ("f32", "f32")}, "dtype_cast": {("i32", "f32")}}
_TORCH = {"i8": "int8", "i32": "int32", "f32": "float32"}


def _shape(shape):
    if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("native-storage source requires a positive static tensor shape")
    return shape


def native_storage_source_spec(entry):
    """Validate a finite native declaration; hashes alone issue no placement authority."""
    if not any(field in entry for field in _FIELDS):
        return None
    if not all(field in entry for field in _FIELDS):
        raise ValueError("native-storage source requires its complete typed declaration")
    pair = (entry["native_input_dtype"], entry["native_output_dtype"])
    op = entry.get("op")
    if pair not in _PAIRS.get(op, set()):
        raise ValueError("unsupported native-storage operation or typed conversion")
    if (entry.get("capture_dtype", entry.get("dtype")) != "fp32"
            or entry.get("quant_scheme") or entry.get("input_palette") is not None):
        raise ValueError("native storage requires fp32 capture without quantization or a palette")
    generalization = entry.get("generalization") or {}
    if generalization.get("must_accelerate") is not False or generalization.get("eligible") is not False:
        raise ValueError("native-storage probe requires an explicit nonaccelerating declaration")
    source = entry["native_source_abi"]
    if not isinstance(source, dict) or set(source) != {
        "full_inventory_sha256", "capture_sha256", "signature_sha256", "signature",
    }:
        raise ValueError("native-storage source has no complete inventory identity")
    if any(not is_sha256(source[key]) for key in (
        "full_inventory_sha256", "capture_sha256", "signature_sha256",
    )):
        raise ValueError("native-storage source has an invalid inventory identity")
    signature = source["signature"]
    if not isinstance(signature, dict) or {"count", "ordinals"} & set(signature):
        raise ValueError("native-storage source requires the complete normalized signature")
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != source["signature_sha256"]:
        raise ValueError("native-storage source signature changed")
    from .semantic_families import from_op

    if (signature.get("provenance_op") != op
            or not isinstance(entry.get("frontend_op"), str) or not entry["frontend_op"]
            or signature.get("frontend_op") != entry["frontend_op"]
            or signature.get("semantic_family") != from_op(op)
            or signature.get("disposition") not in {"host_required", "support_required"}):
        raise ValueError("native-storage source does not match the selected operation")
    operands, results = signature.get("ordered_operand_types"), signature.get("ordered_result_types")
    if (not isinstance(operands, list) or not operands or not isinstance(operands[0], dict)
            or not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict)
            or (operands[0].get("dtype"), results[0].get("dtype")) != pair):
        raise ValueError("native-storage source has no matching ordered operand/result types")
    original_input, original_output = _shape(operands[0].get("shape")), _shape(results[0].get("shape"))
    if op == "dtype_cast" and original_input != original_output:
        raise ValueError("native cast source does not preserve its original shape")
    if op == "permute" and sorted(original_input) != sorted(original_output):
        raise ValueError("native permutation source has inconsistent tensor extents")
    if "shape" in entry and op == "permute" and len(_shape(entry["shape"])) != 2:
        raise ValueError("selected native permutation template requires rank two")
    return {"input_dtype": pair[0], "output_dtype": pair[1]}


def native_storage_loader(spec):
    """Generate byte-stable typed inputs, without computing expected answers."""
    selected = native_storage_source_spec(spec)
    if selected is None:
        return ""
    shape = _shape(spec.get("shape", [spec.get("M", 16), spec.get("K", 16)]))
    if spec["op"] == "permute" and len(shape) != 2:
        raise ValueError("selected native permutation template requires rank two")
    literal = "(" + ", ".join(map(str, shape)) + ",)"
    dtype = selected["input_dtype"]
    count, seed = math.prod(shape), int(spec.get("seed", 0))
    if dtype == "f32":
        value = "(((values * 37 + offset) % 251) - 125).to(torch.float32) / 64.0"
    else:
        bits = 8 if dtype == "i8" else 32
        span, half = 1 << bits, 1 << (bits - 1)
        value = f"(((values * 1664525 + offset) % {span}) - {half}).to(torch.{_TORCH[dtype]})"
    return ("\ndef get_model_and_inputs():\n"
            f"    values = torch.arange({count}, dtype=torch.int64).reshape({literal})\n"
            f"    offset = {seed % (1 << 32)}\n"
            f"    return Model().eval(), ({value},)\n")


def _check_values(value, dtype):
    if isinstance(value, list):
        for child in value:
            _check_values(child, dtype)
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("native-storage capture contains an invalid tensor value")
    if dtype in {"i8", "i32"}:
        bits = 8 if dtype == "i8" else 32
        if value != int(value) or not -(1 << (bits - 1)) <= value < 1 << (bits - 1):
            raise ValueError("native-storage capture contains an invalid integer value")
    else:
        try:
            exact = struct.unpack("<f", struct.pack("<f", value))[0] == value
        except (OverflowError, struct.error) as exc:
            raise ValueError("native-storage capture exceeds f32 storage") from exc
        if not exact:
            raise ValueError("native-storage capture contains an inexact f32 value")


def publish_native_storage(spec, art, cap, golden, binding):
    """Check the complete actual ABI and tensors before publishing their types."""
    selected = native_storage_source_spec(spec)
    if selected is None:
        return
    _require_empty_weights(art.weights_path)
    from merlin.common.mlir_query import forward_signature

    from .capsule_results import _shape as value_shape
    from .capsule_source import _model_output_numeric_policy

    shape = _shape(spec.get("shape", [spec["M"], spec["K"]]))
    output_shape = list(reversed(shape)) if spec["op"] == "permute" else shape
    input_abi = [{"shape": shape, "dtype": selected["input_dtype"]}]
    output_abi = [{"shape": output_shape, "dtype": selected["output_dtype"]}]
    actual_inputs, actual_outputs = forward_signature(art.linalg_mlir)
    if (art.op != spec["op"] or art.dtype != "fp32"
            or not isinstance(art.inputs, list) or len(art.inputs) != 1
            or (art.meta or {}).get("input_abi") != input_abi
            or (art.meta or {}).get("output_abi") != output_abi
            or actual_inputs != [(shape, selected["input_dtype"])]
            or actual_outputs != [(output_shape, selected["output_dtype"])]
            or value_shape(art.inputs[0]) != shape or value_shape(art.golden) != output_shape
            or len(cap["inputs"]) != 1 or len(golden["outputs"]) != 1):
        raise ValueError("native-storage capture disagrees with its complete typed source ABI")
    _check_values(art.inputs[0], selected["input_dtype"])
    _check_values(art.golden, selected["output_dtype"])
    policy = _model_output_numeric_policy(output_abi, [art.golden], binding)
    cap["inputs"][0].update(input_abi[0])
    cap["numeric_policy"] = policy
    cap["pytorch_ref"]["dtype"] = selected["input_dtype"]
    provenance = golden["oracle_provenance"]
    provenance.update(operand_dtype=selected["input_dtype"], output_dtype=selected["output_dtype"],
                      grade_policy=dict(policy))
    provenance["inputs"][cap["inputs"][0]["name"]]["dtype"] = selected["input_dtype"]
    provenance["native_source_identity"] = {
        key: spec["native_source_abi"][key]
        for key in ("full_inventory_sha256", "capture_sha256", "signature_sha256")
    }


def _require_empty_weights(path):
    """A parameter-free capture may export an empty safetensors container."""
    if path is None:
        return
    from pathlib import Path

    try:
        archive = Path(path)
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("weight archive is not an ordinary file")
        with archive.open("rb") as stream:
            prefix = stream.read(8)
            length = int.from_bytes(prefix, "little")
            # Bound metadata allocation before reading; no tensor payload is allowed.
            if len(prefix) != 8 or not 2 <= length <= 65536 or archive.stat().st_size != 8 + length:
                raise ValueError("weight archive is not metadata only")
            header = json.loads(stream.read(length))
        if not isinstance(header, dict) or set(header) - {"__metadata__"}:
            raise ValueError("weight archive contains tensors")
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
        ):
            raise ValueError("weight metadata is malformed")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("native-storage capture requires an empty weight archive") from exc

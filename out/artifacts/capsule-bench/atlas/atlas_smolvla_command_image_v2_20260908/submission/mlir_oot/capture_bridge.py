"""Fail-closed f32 capture boundary for one Atlas FP8/BF16 partition.

The bridge deliberately owns only datatype conversion and dispatch metadata. It
does not execute host graph regions or claim that the full capture is runnable.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from merlin.targetgen.fp8_codec import fp8_e4m3_decode, fp8_e4m3_encode


FP8_MAX_FINITE = 448.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_f32_safetensor(path: Path, name: str) -> tuple[np.ndarray, dict]:
    """Read one named F32 tensor without loading the multi-GB payload."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > 64 * 1024 * 1024:
            raise ValueError(f"safetensors header is unreasonably large: {header_bytes}")
        encoded_header = stream.read(header_bytes)
        if len(encoded_header) != header_bytes:
            raise ValueError("truncated safetensors header")
        header = json.loads(encoded_header)
        if name not in header:
            raise ValueError(f"tensor {name!r} absent from safetensors header")
        spec = header[name]
        if spec.get("dtype") != "F32":
            raise ValueError(f"tensor {name!r} has unsupported storage dtype {spec.get('dtype')!r}")
        shape = tuple(int(value) for value in spec.get("shape", []))
        offsets = spec.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or offsets[0] < 0 or offsets[1] < offsets[0]):
            raise ValueError(f"tensor {name!r} has malformed data offsets")
        size = offsets[1] - offsets[0]
        if size != math.prod(shape) * 4:
            raise ValueError(f"tensor {name!r} shape does not match its byte extent")
        stream.seek(8 + header_bytes + offsets[0])
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"tensor {name!r} payload is truncated")
    return np.frombuffer(raw, dtype="<f4").reshape(shape).copy(), {
        "name": name,
        "shape": list(shape),
        "raw_sha256": sha256_bytes(raw),
        "header_sha256": sha256_bytes(encoded_header),
    }


def _require_finite_f32(name: str, value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if not array.size:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def calibrate_e4m3(name: str, value: np.ndarray) -> dict:
    array = _require_finite_f32(name, value)
    max_abs = float(np.max(np.abs(array.astype(np.float64))))
    scale = max_abs / FP8_MAX_FINITE if max_abs else 1.0
    codes = np.fromiter(
        (fp8_e4m3_encode(float(element) / scale) for element in array.flat),
        dtype=np.uint8,
        count=array.size,
    ).reshape(array.shape)
    decoded = np.fromiter(
        (fp8_e4m3_decode(int(code)) for code in codes.flat),
        dtype=np.float32,
        count=codes.size,
    ).reshape(array.shape)
    reconstructed = decoded * np.float32(scale)
    error = np.abs(reconstructed - array)
    return {
        "name": name,
        "scale": scale,
        "max_abs": max_abs,
        "codes": codes,
        "decoded": decoded,
        "record": {
            "name": name,
            "shape": list(array.shape),
            "scale": scale,
            "scale_hex": float(scale).hex(),
            "max_abs": max_abs,
            "saturated_elements": int(np.count_nonzero((codes & 0x7F) == 0x7E)),
            "zero_elements": int(np.count_nonzero(codes == 0)),
            "max_abs_reconstruction_error": float(np.max(error)),
            "mean_abs_reconstruction_error": float(np.mean(error)),
            "codes_sha256": sha256_bytes(codes.tobytes()),
        },
    }


def f32_to_bf16_rne(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = _require_finite_f32("BF16 input", value)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    words = (rounded >> 16).astype("<u2")
    decoded = (words.astype(np.uint32) << 16).view(np.float32)
    return words, decoded


def validate_partition_abi(partition: dict) -> None:
    if partition.get("kind") != "matmul" or not partition.get("bias_fused"):
        raise ValueError("capture bridge requires one bias-fused rank-2 matmul partition")
    inputs = partition.get("abi", {}).get("inputs", [])
    outputs = partition.get("abi", {}).get("outputs", [])
    if [item.get("name") for item in inputs] != ["A0", "W", "B"]:
        raise ValueError("capture bridge requires ordered A0/W/B inputs")
    if [item.get("device_dtype") for item in inputs] != ["fp8_e4m3", "fp8_e4m3", "bf16"]:
        raise ValueError("capture bridge device input dtypes changed")
    if len(outputs) != 1 or outputs[0].get("device_dtype") != "bf16":
        raise ValueError("capture bridge requires one BF16 device output")
    if not all("xf32>" in item.get("capture_type", "") for item in (*inputs, *outputs)):
        raise ValueError("capture bridge accepts only f32 capture boundaries")
    weight_bridges = inputs[1].get("origin", {}).get("bridges", [])
    if [item.get("op") for item in weight_bridges] != ["linalg.transpose"]:
        raise ValueError("weight boundary must contain exactly the captured transpose")
    geometry = partition.get("geometry", {})
    if any(int(geometry.get(key, 0)) <= 0 for key in ("M", "K", "N")):
        raise ValueError("partition geometry is absent or non-positive")
    lifetime = partition.get("lifetime", {})
    if lifetime.get("last_frontier_use_op_index", -1) < lifetime.get("definition_op_index", 0):
        raise ValueError("partition output lifetime ends before its definition")


def build_dispatch_manifest(partition: dict) -> dict:
    """Build the minimal ordered dispatch/lifetime skeleton from planner ABI."""
    validate_partition_abi(partition)
    definition = partition["lifetime"]["definition_op_index"]
    release = partition["lifetime"]["last_frontier_use_op_index"]
    events = [
        {"phase": "bind_capture_inputs", "capture_op_index": partition["capture_op_index"]},
        {"phase": "quantize_A0_W_and_fold_B", "capture_op_index": partition["capture_op_index"]},
        {"phase": "launch_device_image", "capture_op_index": partition["capture_op_index"]},
        {"phase": "dequantize_and_publish_Y0", "capture_op_index": definition},
        {"phase": "release_Y0_after_frontier", "capture_op_index": release},
    ]
    if any(left["capture_op_index"] > right["capture_op_index"]
           for left, right in zip(events, events[1:])):
        raise ValueError("dispatch events violate capture order")
    return {
        "schema": "atlas_single_partition_dispatch_v1",
        "claim": "host/device skeleton for one partition; no whole-model dispatch claim",
        "partition_id": partition["partition_id"],
        "kernel_id": partition["kernel_id"],
        "capture_regions": partition["capture_regions"],
        "fqn": partition["fqn"],
        "abi": partition["abi"],
        "lifetime": partition["lifetime"],
        "events": events,
    }


def bridge_inputs(activation: np.ndarray, captured_weight: np.ndarray,
                  bias: np.ndarray, geometry: dict) -> dict:
    """Quantize capture values and return exact device bytes plus provenance."""
    m, k, n = (int(geometry[key]) for key in ("M", "K", "N"))
    activation = _require_finite_f32("A0", activation)
    captured_weight = _require_finite_f32("W", captured_weight)
    bias = _require_finite_f32("B", bias)
    if activation.shape != (m, k) or captured_weight.shape != (k, n) or bias.shape != (n,):
        raise ValueError(
            f"capture values do not match geometry {(m, k, n)}: "
            f"{activation.shape}, {captured_weight.shape}, {bias.shape}"
        )
    qa, qw = calibrate_e4m3("A0", activation), calibrate_e4m3("W", captured_weight)
    output_scale = qa["scale"] * qw["scale"]
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError(f"invalid output scale {output_scale!r}")
    bias_words, bias_quant_domain = f32_to_bf16_rne(bias / np.float32(output_scale))
    reconstructed_bias = bias_quant_domain * np.float32(output_scale)
    bias_error = np.abs(reconstructed_bias - bias)
    return {
        "preloads": {
            "A0": qa["codes"].tobytes(),
            "W": qw["codes"].tobytes(),
            "B": bias_words.tobytes(),
        },
        "decoded": {
            "A0": qa["decoded"],
            "W": qw["decoded"],
            "B_quant_domain": bias_quant_domain,
        },
        "record": {
            "activation": qa["record"],
            "weight": qw["record"],
            "output_scale": output_scale,
            "output_scale_hex": float(output_scale).hex(),
            "bias": {
                "equation": "BF16_RNE(B / output_scale)",
                "quant_domain_words_sha256": sha256_bytes(bias_words.tobytes()),
                "quant_domain_max_abs": float(np.max(np.abs(bias_quant_domain))),
                "max_abs_error_after_output_rescale": float(np.max(bias_error)),
                "mean_abs_error_after_output_rescale": float(np.mean(bias_error)),
            },
        },
    }


def comparison(actual: np.ndarray, reference: np.ndarray, relative_floor: float) -> dict:
    actual = _require_finite_f32("actual output", actual)
    reference = _require_finite_f32("reference output", reference)
    if actual.shape != reference.shape:
        raise ValueError(f"output shape mismatch: {actual.shape} != {reference.shape}")
    delta = actual - reference
    denominator = np.maximum(np.abs(reference), np.float32(relative_floor))
    norms = float(np.linalg.norm(actual)) * float(np.linalg.norm(reference))
    cosine = float(np.dot(actual.ravel(), reference.ravel()) / norms) if norms else 1.0
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_error": float(np.max(np.abs(delta) / denominator)),
        "max_relative_denominator_floor": relative_floor,
        "cosine_similarity": cosine,
    }


def passes_tolerance(metrics: dict, tolerance: dict) -> bool:
    """Apply only the two predeclared source-accuracy gates, failing closed."""
    try:
        maximum = float(metrics["max_abs_error"])
        cosine = float(metrics["cosine_similarity"])
        maximum_limit = float(tolerance["max_abs_error"])
        cosine_limit = float(tolerance["cosine_min"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed comparison metrics or tolerance contract") from error
    if not all(math.isfinite(value) for value in (
        maximum, cosine, maximum_limit, cosine_limit
    )):
        raise ValueError("non-finite comparison metric or tolerance")
    return maximum <= maximum_limit and cosine >= cosine_limit

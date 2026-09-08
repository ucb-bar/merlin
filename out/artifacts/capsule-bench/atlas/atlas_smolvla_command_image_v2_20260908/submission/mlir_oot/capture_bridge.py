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
ATLAS_GSIM_DRAM_BYTES = 1 << 20
_DEVICE_DTYPE_BYTES = {
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "i8": 1,
    "bf16": 2,
    "f16": 2,
    "i16": 2,
    "f32": 4,
    "i32": 4,
    "i64": 8,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_gsim_dram_window(
    command_buffer: dict, capacity_bytes: int = ATLAS_GSIM_DRAM_BYTES
) -> dict:
    """Reject command buffers that alias in the current 1 MiB GSIM DRAM.

    The adopted RTL harness indexes its byte array with address & (2**20-1).
    A normal, non-overlapping command buffer can therefore become silently
    overlapping when its allocated span crosses that window.  This preflight
    mirrors that hardware integration constraint before any numerical result
    is interpreted.
    """
    if not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
        raise ValueError("GSIM DRAM capacity must be a positive integer")
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("command buffer has no tensor allocation map")
    allocations = []
    for name, tensor in sorted(tensors.items()):
        try:
            base = int(tensor["base"])
            shape = [int(extent) for extent in tensor["shape"]]
            element_bytes = _DEVICE_DTYPE_BYTES[tensor["dtype"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed tensor allocation {name!r}") from error
        if base < 0 or not shape or any(extent <= 0 for extent in shape):
            raise ValueError(f"malformed tensor allocation {name!r}")
        size = math.prod(shape) * element_bytes
        allocations.append({"name": name, "base": base, "end": base + size, "bytes": size})

    low = min(item["base"] for item in allocations)
    high = max(item["end"] for item in allocations)
    span = high - low
    record = {
        "schema": "atlas_gsim_dram_window_preflight_v1",
        "capacity_bytes": capacity_bytes,
        "allocated_span_bytes": span,
        "headroom_bytes": capacity_bytes - span,
        "allocation_count": len(allocations),
        "allocations": allocations,
        "address_mapping": "physical_byte_index = device_address & (capacity_bytes - 1)",
    }
    if span > capacity_bytes:
        raise ValueError(
            "Atlas GSIM DRAM allocation span exceeds its alias-free window: "
            f"{span} > {capacity_bytes} bytes (over by {span - capacity_bytes})"
        )

    # The current capacity is a power of two, matching the harness bit mask.
    if capacity_bytes & (capacity_bytes - 1):
        raise ValueError("GSIM DRAM capacity must be a power of two")
    mapped = []
    for item in allocations:
        start = item["base"] & (capacity_bytes - 1)
        stop = start + item["bytes"]
        if stop > capacity_bytes:
            raise ValueError(
                f"Atlas GSIM tensor {item['name']} wraps within the DRAM window: "
                f"mapped [{start}, {stop}) exceeds {capacity_bytes}"
            )
        mapped.append((start, stop, item["name"]))
    for left, right in zip(sorted(mapped), sorted(mapped)[1:]):
        if left[1] > right[0]:
            raise ValueError(
                "Atlas GSIM tensor allocations alias after address masking: "
                f"{left[2]} overlaps {right[2]}"
            )
    record["alias_free"] = True
    return record


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


def calibrate_e4m3(
    name: str, value: np.ndarray, code_cap: float = FP8_MAX_FINITE
) -> dict:
    array = _require_finite_f32(name, value)
    if not math.isfinite(code_cap) or code_cap <= 0.0 or code_cap > FP8_MAX_FINITE:
        raise ValueError(f"invalid E4M3 calibration code cap {code_cap!r}")
    max_abs = float(np.max(np.abs(array.astype(np.float64))))
    scale = max_abs / code_cap if max_abs else 1.0
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
            "code_cap": code_cap,
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


def bridge_inputs(
    activation: np.ndarray, captured_weight: np.ndarray,
    bias: np.ndarray, geometry: dict, code_cap: float = FP8_MAX_FINITE,
) -> dict:
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
    qa = calibrate_e4m3("A0", activation, code_cap)
    qw = calibrate_e4m3("W", captured_weight, code_cap)
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

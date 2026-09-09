#!/usr/bin/env python3
"""Audit whether prepared W8A8 convolutions can use Jack's narrow readout.

This is intentionally an admission pass, not an approximation pass.  It follows
the integer convolution into its captured FP32 bias/requant graph, resolves the
frozen parameters through the bundle manifest, and records every reason a
native i8 LOOP_CONV result would change model semantics.

The target's ordinary execute-path D operand is hardwired away, but
LoopConvLdBias has an independent LOAD3 path which writes per-output-channel
i32 bias directly into accumulator rows before execution.  Conflating those two
paths was the v1 postmortem's original error.  Bias remains an *arithmetic*
refusal when the captured f32 bias is not an exact integer accumulator offset.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import deque
from pathlib import Path

import numpy as np
from xdsl.ir import Block, Operation

from mlir_oot.frontend.direct_conv import recognize as recognize_conv
from mlir_oot.frontend.integer_prepare import prepare_int8_text
from mlir_oot.frontend.parse import parse_module


_DTYPES = {"F32": np.dtype("<f4"), "I64": np.dtype("<i8"), "I8": np.dtype("i1")}


class SafeTensors:
    def __init__(self, path: Path):
        self.raw = path.read_bytes()
        header_size = struct.unpack("<Q", self.raw[:8])[0]
        self.header = json.loads(self.raw[8:8 + header_size])
        self.data_start = 8 + header_size

    def get(self, name: str) -> np.ndarray:
        entry = self.header[name]
        dtype = _DTYPES[entry["dtype"]]
        begin, end = entry["data_offsets"]
        return np.frombuffer(
            self.raw, dtype=dtype, count=(end - begin) // dtype.itemsize,
            offset=self.data_start + begin).reshape(entry["shape"])


def _attr(op: Operation, name: str) -> str:
    return getattr(op.attributes.get(name), "data", "")


def _op_name(op: Operation) -> str:
    return getattr(getattr(op, "op_name", None), "data", "") or op.name


def _users(op: Operation) -> list[Operation]:
    return list(dict.fromkeys(
        use.operation for result in op.results for use in result.uses
        if use.operation.parent_block() is not None))


def _shortest_quant_path(root: Operation, limit: int = 16) -> list[Operation]:
    todo = deque([(root, [])])
    seen = {root}
    while todo:
        op, path = todo.popleft()
        if len(path) >= limit:
            continue
        for user in _users(op):
            if user in seen:
                continue
            new_path = [*path, user]
            if _op_name(user) == "quant_ext.quantize_per_tensor":
                return new_path
            seen.add(user)
            todo.append((user, new_path))
    return []


def _arg_index(value, block: Block) -> int | None:
    return block.args.index(value) if isinstance(value.owner, Block) and value.owner is block else None


def _f32_bits_unique(values: np.ndarray) -> int:
    return int(np.unique(np.asarray(values, dtype=np.float32).view(np.uint32)).size)


def _scalar_constant(value) -> np.float32 | None:
    """Resolve a scalar through tensor.splat to an arith.constant."""
    owner = value.owner
    if getattr(owner, "name", "") == "tensor.splat" and owner.operands:
        owner = owner.operands[0].owner
    if getattr(owner, "name", "") != "arith.constant":
        return None
    attr = getattr(owner, "value", None)
    data = getattr(getattr(attr, "value", None), "data", None)
    return np.float32(data) if data is not None else None


def audit(source: Path, manifest_path: Path, weights_path: Path) -> dict:
    prepared, preparation = prepare_int8_text(source.read_text())
    module = parse_module(prepared)
    func = next(op for op in module.walk() if op.name == "func.func")
    block = func.regions[0].blocks[0]
    manifest = json.loads(manifest_path.read_text())
    weights = SafeTensors(weights_path)

    records = []
    for index, conv in enumerate(op for op in module.walk() if recognize_conv(op) is not None):
        spec = recognize_conv(conv)
        path = _shortest_quant_path(conv)
        requant = path[0] if path and _attr(path[0], "prov.role") == "requant" else None
        bias_op = path[1] if len(path) > 1 else None
        weight_scale_arg = _arg_index(requant.operands[2], block) if requant is not None else None
        bias_arg = (_arg_index(bias_op.operands[1], block)
                    if bias_op is not None and len(bias_op.operands) > 1 else None)
        weight_scale_name = (manifest[str(weight_scale_arg)]["weight"]
                             if weight_scale_arg is not None else None)
        bias_name = manifest[str(bias_arg)]["weight"] if bias_arg is not None else None
        weight_scale = weights.get(weight_scale_name) if weight_scale_name else np.array([])
        bias = weights.get(bias_name) if bias_name else np.array([])
        activation_scale = (_scalar_constant(requant.operands[1])
                            if requant is not None and len(requant.operands) > 1 else None)
        output_scale = (_scalar_constant(path[-1].operands[1])
                        if path and len(path[-1].operands) > 1 else None)
        bias_ratio = np.array([], dtype=np.float32)
        bias_exact_channels = 0
        bias_fractional_distance = np.array([], dtype=np.float32)
        if (bias.size and weight_scale.size and activation_scale is not None
                and bias.shape == weight_scale.shape):
            accumulator_unit = np.float32(activation_scale * weight_scale.astype(np.float32))
            bias_ratio = np.float32(bias.astype(np.float32) / accumulator_unit)
            nearest = np.rint(bias_ratio).astype(np.float32)
            bias_exact_channels = int(np.count_nonzero(bias_ratio == nearest))
            bias_fractional_distance = np.abs(np.float32(bias_ratio - nearest))
        path_ops = [_attr(op, "prov.op") or _op_name(op) for op in path]
        residual = "add" in path_ops
        maxpool = "max_pool2d" in path_ops
        global_tail = any(name in path_ops for name in ("adaptive_avg_pool2d", "view", "matmul"))
        nonzero_bias = int(np.count_nonzero(bias))
        unique_scales = _f32_bits_unique(weight_scale) if weight_scale.size else 0
        reasons = []
        if requant is None or bias_name is None or not path or _op_name(path[-1]) != "quant_ext.quantize_per_tensor":
            reasons.append("unrecognized_conv_bias_requant_quantize_chain")
        if bias.size and bias_exact_channels != int(bias.size):
            reasons.append("float_bias_not_exact_integer_accumulator_offset")
        if unique_scales > 1:
            reasons.append("per_channel_scale_requires_channel_partitioned_store_configuration")
        if residual:
            reasons.append("residual_add_precedes_quantize")
        if global_tail:
            reasons.append("global_reduction_or_fc_precedes_quantize")
        # The existing backend emits output-row LOOP_WS.  LOOP_CONV is required to make
        # first-layer maxpool part of one native narrow result, and is not yet emitted.
        if maxpool:
            reasons.append("maxpool_requires_native_loop_conv_emission")
        records.append({
            "index": index,
            "region": _attr(conv, "prov.region_id"),
            "fqn": _attr(conv, "prov.fqn"),
            "geometry": {"n": spec.batch, "ci": spec.ci, "hi": spec.hi, "wi": spec.wi,
                         "co": spec.co, "kh": spec.kh, "kw": spec.kw,
                         "ho": spec.ho, "wo": spec.wo},
            "path_to_quantize": path_ops,
            "bias": {"arg": bias_arg, "tensor": bias_name, "channels": int(bias.size),
                     "nonzero_channels": nonzero_bias,
                     "native_transport": "loop_conv_load3_to_accumulator",
                     "exact_integer_accumulator_channels": bias_exact_channels,
                     "fractional_distance_to_integer_min": (
                         float(bias_fractional_distance.min())
                         if bias_fractional_distance.size else None),
                     "fractional_distance_to_integer_max": (
                         float(bias_fractional_distance.max())
                         if bias_fractional_distance.size else None),
                     "min": float(bias.min()) if bias.size else None,
                     "max": float(bias.max()) if bias.size else None},
            "activation_scale": float(activation_scale) if activation_scale is not None else None,
            "output_scale": float(output_scale) if output_scale is not None else None,
            "weight_scale": {"arg": weight_scale_arg, "tensor": weight_scale_name,
                             "channels": int(weight_scale.size),
                             "unique_f32_values": unique_scales,
                             "min": float(weight_scale.min()) if weight_scale.size else None,
                             "max": float(weight_scale.max()) if weight_scale.size else None},
            "native_exact_admitted": not reasons,
            "refusal_reasons": reasons,
        })

    summary = {
        "schema": "merlin_gemmini_native_narrow_epilogue_inventory_v2",
        "source": str(source.resolve()),
        "weights": str(weights_path.resolve()),
        "prepared_integer_convolutions": preparation["integer_pass_counts"]["conv_int8"],
        "convolutions_audited": len(records),
        "native_exact_admitted": sum(r["native_exact_admitted"] for r in records),
        "native_exact_refused": sum(not r["native_exact_admitted"] for r in records),
        "all_channels_nonzero_bias": sum(
            r["bias"]["channels"] == r["bias"]["nonzero_channels"] for r in records),
        "all_bias_channels_exact_integer_accumulator_offset": sum(
            r["bias"]["channels"] == r["bias"]["exact_integer_accumulator_channels"]
            for r in records),
        "per_channel_scale": sum(r["weight_scale"]["unique_f32_values"] > 1 for r in records),
        "output_channels_total": sum(r["geometry"]["co"] for r in records),
        "minimum_channel_partitioned_loop_conv_launches": sum(
            r["geometry"]["co"] for r in records),
        "residual_add_paths": sum("residual_add_precedes_quantize" in r["refusal_reasons"]
                                  for r in records),
        "maxpool_paths": sum("maxpool_requires_native_loop_conv_emission" in r["refusal_reasons"]
                             for r in records),
        "global_tail_paths": sum("global_reduction_or_fc_precedes_quantize" in r["refusal_reasons"]
                                 for r in records),
        "hardware_qualification": "not_run_no_candidate_admitted",
        "erratum": (
            "nonzero bias is supported by LoopConvLdBias LOAD3 even when the "
            "ordinary execute-path D operand is hardwired to garbage; the refusal is "
            "the captured f32 arithmetic, not transport"
        ),
    }
    return {"summary": summary, "convolutions": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.manifest, args.weights)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

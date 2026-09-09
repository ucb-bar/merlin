"""Fail-closed source/command contracts for Atlas capture partitions.

This module deliberately separates static command semantics from physical
capture qualification.  A contract proves that the captured f32/BF16 operation, the
partition ABI, and a saved command buffer describe the same computation.  It
does not prove that the encoded image ran, that calibration meets the model's
accuracy gate, or that a whole-graph runtime can dispatch it.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, MutableMapping

import numpy as np

from .capture_bridge import calibrate_e4m3, f32_to_bf16_rne
from .frontend import _str_attr
from .full_graph import _tensor_dtype, _tensor_shape
from .host_semantics import HostSemanticLane, _general_affine_map_signature


class UnsupportedAcceleratorContract(ValueError):
    """The source/ABI/command tuple is outside the exact proven subset."""


def contract_sha256(signature: dict) -> str:
    return hashlib.sha256(
        (json.dumps(signature, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _dtype_bytes(dtype: str) -> int:
    try:
        return {"fp8_e4m3": 1, "bf16": 2}[dtype]
    except KeyError as error:
        raise UnsupportedAcceleratorContract(
            f"unsupported command tensor dtype {dtype!r}"
        ) from error


def _zero_initialized(value) -> bool:
    fill = getattr(value, "owner", None)
    if getattr(fill, "name", None) != "linalg.fill" or len(fill.inputs) != 1:
        return False
    constant = getattr(fill.inputs[0], "owner", None)
    if getattr(constant, "name", None) != "arith.constant":
        return False
    raw = getattr(
        getattr(constant.properties.get("value"), "value", None), "data", None
    )
    return raw is not None and float(raw) == 0.0


def _zero_splat_initialized(value) -> bool:
    splat = getattr(value, "owner", None)
    if getattr(splat, "name", None) != "tensor.splat" or len(splat.operands) != 1:
        return False
    constant = getattr(splat.operands[0], "owner", None)
    if getattr(constant, "name", None) != "arith.constant":
        return False
    raw = getattr(
        getattr(constant.properties.get("value"), "value", None), "data", None
    )
    return raw is not None and float(raw) == 0.0


def _validate_bias_epilogue(matmul, bias) -> None:
    if bias.name != "linalg.generic" or len(bias.inputs) != 2:
        raise UnsupportedAcceleratorContract("bias epilogue is not one binary generic")
    output_shape = _tensor_shape(bias.results[0])
    bias_shape = _tensor_shape(bias.inputs[1])
    if output_shape is None or bias_shape is None:
        raise UnsupportedAcceleratorContract("bias epilogue tensors are unranked")
    maps = list(bias.indexing_maps)
    if len(maps) != 3:
        raise UnsupportedAcceleratorContract("bias epilogue map arity changed")
    records = [
        _general_affine_map_signature(mapping, shape, len(output_shape))
        for mapping, shape in zip(
            maps, (_tensor_shape(bias.inputs[0]), bias_shape, output_shape)
        )
    ]
    expected = [
        [
            {"kind": "dim", "position": 0, "extent": output_shape[0]},
            {"kind": "dim", "position": 1, "extent": output_shape[1]},
        ],
        [{"kind": "dim", "position": 1, "extent": output_shape[1]}],
        [
            {"kind": "dim", "position": 0, "extent": output_shape[0]},
            {"kind": "dim", "position": 1, "extent": output_shape[1]},
        ],
    ]
    if records != expected:
        raise UnsupportedAcceleratorContract("bias epilogue affine maps changed")
    iterators = [getattr(item.data, "value", str(item.data)) for item in bias.iterator_types]
    if iterators != ["parallel", "parallel"]:
        raise UnsupportedAcceleratorContract("bias epilogue is not fully parallel")
    block = bias.body.blocks[0]
    body = list(block.ops)
    if (len(block.args) != 3
            or tuple(op.name for op in body) != ("arith.addf", "linalg.yield")
            or tuple(body[0].operands) != tuple(block.args[:2])
            or tuple(body[1].operands) != tuple(body[0].results)):
        raise UnsupportedAcceleratorContract("bias epilogue scalar dataflow changed")
    if bias.inputs[0] is not matmul.results[0]:
        raise UnsupportedAcceleratorContract("bias epilogue does not consume matmul result")


def _expected_command_buffer(geometry: dict, bias_fused: bool) -> tuple[dict, list[dict], dict]:
    m, k, n = (int(geometry[key]) for key in ("M", "K", "N"))
    tensors = {
        "W": {"shape": [k, n], "dtype": "fp8_e4m3", "role": "weight"},
        "A0": {"shape": [m, k], "dtype": "fp8_e4m3", "role": "input"},
    }
    if bias_fused:
        tensors["B"] = {"shape": [n], "dtype": "bf16", "role": "bias"}
    tensors["Y0"] = {"shape": [m, n], "dtype": "bf16", "role": "output"}
    commit_attributes = {
        "epilogue": ["bias_add"] if bias_fused else [],
        "output_dtype": "bf16",
    }
    if bias_fused:
        commit_attributes["bias"] = "B"
    commands = [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_resident"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT",
         "operands": {"lhs": "A0", "rhs": "W_resident", "dst": "acc0"}},
        {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
         "attributes": commit_attributes},
        {"opcode": "EVICT", "operands": {"handle": "W_resident"}},
    ]
    args = [{"tensor": "W", "access": "read"},
            {"tensor": "A0", "access": "read"}]
    if bias_fused:
        args.append({"tensor": "B", "access": "read"})
    args.append({"tensor": "Y0", "access": "write"})
    kernel_abi = {"kind": "whole_program", "args": args, "outputs": ["Y0"]}
    return tensors, commands, kernel_abi


def _validate_command_buffer(partition: dict, command: dict) -> dict:
    if {key: command.get(key) for key in ("abi_version", "target", "backend")} != {
        "abi_version": "0.1", "target": "atlas", "backend": "atlas-xdsl",
    }:
        raise UnsupportedAcceleratorContract("command buffer header changed")
    expected_tensors, expected_commands, expected_abi = _expected_command_buffer(
        partition["geometry"], bool(partition["bias_fused"])
    )
    tensors = command.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(expected_tensors):
        raise UnsupportedAcceleratorContract("command tensor set differs from rank-2 contract")
    allocations = []
    for name, expected in expected_tensors.items():
        actual = tensors[name]
        if {key: actual.get(key) for key in expected} != expected:
            raise UnsupportedAcceleratorContract(f"command tensor {name} signature changed")
        base = actual.get("base")
        if not isinstance(base, int) or base < 0 or base % 32:
            raise UnsupportedAcceleratorContract(f"command tensor {name} base is not 32-byte aligned")
        size = math.prod(expected["shape"]) * _dtype_bytes(expected["dtype"])
        allocations.append((base, base + size, name))
    for left, right in zip(sorted(allocations), sorted(allocations)[1:]):
        if left[1] > right[0]:
            raise UnsupportedAcceleratorContract(
                f"command allocations overlap: {left[2]} and {right[2]}"
            )
    if command.get("commands") != expected_commands:
        raise UnsupportedAcceleratorContract("rank-2 command dependency chain changed")
    if command.get("kernel_abi") != expected_abi:
        raise UnsupportedAcceleratorContract("rank-2 command kernel ABI changed")
    return {
        "command_sequence": [row["opcode"] for row in expected_commands],
        "tensor_bases": {name: tensors[name]["base"] for name in expected_tensors},
        "allocated_span_bytes": max(end for _, end, _ in allocations)
        - min(start for start, _, _ in allocations),
    }


def _expected_batched_command_buffer(geometry: dict) -> tuple[dict, list[dict], dict]:
    batch, m, k, n = (
        int(geometry[key]) for key in ("B", "M", "K", "N")
    )
    tensors = {
        "A0": {
            "shape": [batch, m, k], "dtype": "fp8_e4m3", "role": "input"
        },
        "W": {
            "shape": [batch, k, n], "dtype": "fp8_e4m3", "role": "weight"
        },
        "Y0": {
            "shape": [batch, m, n], "dtype": "bf16", "role": "output"
        },
    }
    commands = [
        {
            "opcode": "RES_PACK",
            "operands": {"src": "W", "dst": "W_resident"},
            "attributes": {"layout": "packed_rhs"},
        },
        {
            "opcode": "BATCHED_MATMUL",
            "operands": {"src": "A0", "rhs": "W_resident", "dst": "Y0"},
            "attributes": {
                "name": "Y0", "batch": batch, "output_dtype": "bf16"
            },
        },
        {"opcode": "EVICT", "operands": {"handle": "W_resident"}},
    ]
    kernel_abi = {
        "kind": "whole_program",
        "args": [
            {"tensor": "A0", "access": "read"},
            {"tensor": "W", "access": "read"},
            {"tensor": "Y0", "access": "write"},
        ],
        "outputs": ["Y0"],
    }
    return tensors, commands, kernel_abi


def _validate_batched_command_buffer(partition: dict, command: dict) -> dict:
    if {key: command.get(key) for key in ("abi_version", "target", "backend")} != {
        "abi_version": "0.1", "target": "atlas", "backend": "atlas-xdsl",
    }:
        raise UnsupportedAcceleratorContract("command buffer header changed")
    expected_tensors, expected_commands, expected_abi = (
        _expected_batched_command_buffer(partition["geometry"])
    )
    tensors = command.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(expected_tensors):
        raise UnsupportedAcceleratorContract(
            "command tensor set differs from batched contract"
        )
    allocations = []
    for name, expected in expected_tensors.items():
        actual = tensors[name]
        if {key: actual.get(key) for key in expected} != expected:
            raise UnsupportedAcceleratorContract(
                f"batched command tensor {name} signature changed"
            )
        base = actual.get("base")
        if not isinstance(base, int) or base < 0 or base % 32:
            raise UnsupportedAcceleratorContract(
                f"batched command tensor {name} base is not 32-byte aligned"
            )
        size = math.prod(expected["shape"]) * _dtype_bytes(expected["dtype"])
        allocations.append((base, base + size, name))
    for left, right in zip(sorted(allocations), sorted(allocations)[1:]):
        if left[1] > right[0]:
            raise UnsupportedAcceleratorContract(
                f"batched command allocations overlap: {left[2]} and {right[2]}"
            )
    if command.get("commands") != expected_commands:
        raise UnsupportedAcceleratorContract(
            "batched command dependency chain changed"
        )
    if command.get("kernel_abi") != expected_abi:
        raise UnsupportedAcceleratorContract("batched command kernel ABI changed")
    return {
        "command_sequence": [row["opcode"] for row in expected_commands],
        "resident_weight_source": "W",
        "resident_weight_handle": "W_resident",
        "tensor_bases": {name: tensors[name]["base"] for name in expected_tensors},
        "allocated_span_bytes": max(end for _, end, _ in allocations)
        - min(start for start, _, _ in allocations),
    }


@dataclass(frozen=True)
class AcceleratorContract:
    partition_id: str
    signature: dict


def _extract_rank2_contract(partition: dict, operations: Mapping[str, tuple], command: dict,
                            receipt: dict) -> AcceleratorContract:
    if partition.get("kind") != "matmul":
        raise UnsupportedAcceleratorContract("only rank-2 command contracts are qualified")
    if partition.get("source_semantic") not in {"matmul", "addmm"}:
        raise UnsupportedAcceleratorContract(
            "rank-2 source requires capture-specific preprocessing outside the command"
        )
    capture_regions = partition.get("capture_regions", [])
    expected_region_count = 2 if partition.get("bias_fused") else 1
    if len(capture_regions) != expected_region_count:
        raise UnsupportedAcceleratorContract("capture region count differs from rank-2 contract")
    contraction_ops = operations.get(capture_regions[0], ())
    matmuls = [op for op in contraction_ops if op.name == "linalg.matmul"]
    if len(contraction_ops) != 1 or len(matmuls) != 1:
        raise UnsupportedAcceleratorContract("source region is not one isolated linalg.matmul")
    matmul = matmuls[0]
    if len(matmul.inputs) != 2 or len(matmul.outputs) != 1 or len(matmul.results) != 1:
        raise UnsupportedAcceleratorContract("source matmul arity changed")
    m, k, n = (int(partition["geometry"][key]) for key in ("M", "K", "N"))
    values = (*matmul.inputs, matmul.results[0])
    source_dtype = _tensor_dtype(matmul.inputs[0])
    if ([_tensor_shape(value) for value in values]
            != [(m, k), (k, n), (m, n)]
            or source_dtype not in {"f32", "bf16"}
            or any(_tensor_dtype(value) != source_dtype for value in values)):
        raise UnsupportedAcceleratorContract("source matmul shape or dtype changed")
    if not _zero_initialized(matmul.outputs[0]):
        raise UnsupportedAcceleratorContract("source matmul accumulator is not proven zero")
    output = matmul.results[0]
    if partition.get("bias_fused"):
        bias_ops = operations.get(capture_regions[1], ())
        if len(bias_ops) != 1:
            raise UnsupportedAcceleratorContract("bias capture region is not isolated")
        bias = bias_ops[0]
        _validate_bias_epilogue(matmul, bias)
        if (_tensor_shape(bias.inputs[1]) != (n,)
                or any(_tensor_dtype(value) != source_dtype
                       for value in (*bias.inputs, bias.results[0]))):
            raise UnsupportedAcceleratorContract("source bias shape or dtype changed")
        output = bias.results[0]
    abi_inputs = partition.get("abi", {}).get("inputs", [])
    abi_output = partition.get("abi", {}).get("outputs", [])
    source_values = list(matmul.inputs)
    if partition.get("bias_fused"):
        source_values.append(bias.inputs[1])
    if len(abi_inputs) != len(source_values) or len(abi_output) != 1:
        raise UnsupportedAcceleratorContract("partition ABI arity changed")
    expected_names = ["A0", "W"] + (["B"] if partition.get("bias_fused") else [])
    expected_device_dtypes = ["fp8_e4m3", "fp8_e4m3"] + (
        ["bf16"] if partition.get("bias_fused") else []
    )
    for entry, value, name, dtype in zip(
        abi_inputs, source_values, expected_names, expected_device_dtypes
    ):
        if (entry.get("name") != name or entry.get("capture_type") != str(value.type)
                or entry.get("device_dtype") != dtype):
            raise UnsupportedAcceleratorContract(f"partition ABI input {name} changed")
    if (abi_output[0].get("name") != "Y0"
            or abi_output[0].get("capture_type") != str(output.type)
            or abi_output[0].get("device_dtype") != "bf16"):
        raise UnsupportedAcceleratorContract("partition ABI output changed")
    image = partition.get("image", {})
    if (image.get("kernel_id") != partition.get("kernel_id")
            or image.get("command_count") != 4 or not image.get("fits_imem")
            or not 0 < int(image.get("instruction_words", 0)) <= int(image.get("imem_words", 0))):
        raise UnsupportedAcceleratorContract("compiled image receipt is incomplete")
    if (receipt.get("kernel_id") != partition.get("kernel_id")
            or receipt.get("kind") != "matmul"
            or receipt.get("geometry") != partition.get("geometry")
            or receipt.get("bias_fused") != partition.get("bias_fused")
            or not receipt.get("fits_imem")):
        raise UnsupportedAcceleratorContract("kernel library receipt differs from partition")
    command_record = _validate_command_buffer(partition, command)
    signature = {
        "schema": "atlas_rank2_capture_command_contract_v1",
        "kind": "matmul",
        "partition_id": partition["partition_id"],
        "kernel_id": partition["kernel_id"],
        "source_semantic": partition["source_semantic"],
        "source_dtype": source_dtype,
        "fqn": partition["fqn"],
        "capture_regions": capture_regions,
        "geometry": partition["geometry"],
        "bias_fused": partition["bias_fused"],
        "input_origins": [entry["origin"]["kind"] for entry in abi_inputs],
        "capture_types": [entry["capture_type"] for entry in abi_inputs]
        + [abi_output[0]["capture_type"]],
        "device_dtypes": expected_device_dtypes + ["bf16"],
        "command": command_record,
        "image": {
            "instruction_words": image["instruction_words"],
            "imem_words": image["imem_words"],
            "assembly_sha256": receipt["assembly_sha256"],
            "interface_sha256": receipt["interface_sha256"],
        },
        "qualification_scope": (
            "static source/ABI/command agreement only; not calibration, encoded-image "
            "execution, physical partition qualification, or E2E"
        ),
    }
    return AcceleratorContract(partition["partition_id"], signature)


def _extract_patch_im2col_contract(
    partition: dict,
    command: dict,
    receipt: dict,
    host_programs: Mapping[str, object],
    argument_indices: Mapping[object, int],
) -> AcceleratorContract:
    """Bind the exact captured patch im2col to the existing rank-2 command ABI.

    The capture expresses convolution as ``kernel_matrix @ patch_matrix``.  The
    mesh command consumes those two materialized matrices; the surrounding
    reshape and bias epilogue remain explicit host work.  This is deliberately
    limited to the complete topology already accepted by HostSemanticLane.
    """
    if partition.get("kind") != "matmul":
        raise UnsupportedAcceleratorContract("patch contraction is not rank-2 matmul")
    if partition.get("source_semantic") != "convolution_im2col_matmul":
        raise UnsupportedAcceleratorContract("patch source semantic changed")
    capture_regions = partition.get("capture_regions", [])
    if len(capture_regions) != 1:
        raise UnsupportedAcceleratorContract("patch capture region count changed")
    region_id = capture_regions[0]
    program = host_programs.get(region_id)
    if (program is None
            or program.signature.get("schema")
            != "atlas_host_im2col_convolution_signature_v1"):
        raise UnsupportedAcceleratorContract(
            "patch im2col topology lacks the exact host preprocessing signature"
        )
    contraction_ops = tuple(program.operations)
    matmuls = [op for op in contraction_ops if op.name == "linalg.matmul"]
    if len(matmuls) != 1:
        raise UnsupportedAcceleratorContract("patch region does not contain one matmul")
    matmul = matmuls[0]
    if len(matmul.inputs) != 2 or len(matmul.outputs) != 1 or len(matmul.results) != 1:
        raise UnsupportedAcceleratorContract("patch matmul arity changed")
    m, k, n = (int(partition["geometry"][key]) for key in ("M", "K", "N"))
    values = (*matmul.inputs, matmul.results[0])
    if ([_tensor_shape(value) for value in values] != [(m, k), (k, n), (m, n)]
            or any(_tensor_dtype(value) != "f32" for value in values)):
        raise UnsupportedAcceleratorContract("patch matmul shape or dtype changed")
    if not _zero_splat_initialized(matmul.outputs[0]):
        raise UnsupportedAcceleratorContract("patch matmul accumulator is not proven zero")

    image, kernel, bias = program.input_values
    if any(value not in argument_indices for value in (image, kernel, bias)):
        raise UnsupportedAcceleratorContract(
            "patch source tensors are not direct function arguments"
        )
    image_shape, kernel_shape, bias_shape = (
        tuple(shape) for shape in program.signature["input_shapes"]
    )
    batch, channels, height, width = image_shape
    out_channels, weight_channels, kernel_h, kernel_w = kernel_shape
    output_shape = tuple(program.signature["output_shape"])
    out_batch, result_channels, out_h, out_w = output_shape
    if (batch != 1 or channels != weight_channels or out_batch != batch
            or result_channels != out_channels or bias_shape != (out_channels,)
            or (m, k, n) != (out_channels, channels * kernel_h * kernel_w,
                              batch * out_h * out_w)
            or height != out_h * kernel_h or width != out_w * kernel_w):
        raise UnsupportedAcceleratorContract("patch geometry differs from signed im2col bounds")
    if tuple(matmul.inputs) != (contraction_ops[4].results[0], contraction_ops[2].results[0]):
        raise UnsupportedAcceleratorContract("patch matmul does not consume signed matrices")

    abi_inputs = partition.get("abi", {}).get("inputs", [])
    abi_output = partition.get("abi", {}).get("outputs", [])
    if len(abi_inputs) != 2 or len(abi_output) != 1:
        raise UnsupportedAcceleratorContract("patch partition ABI arity changed")
    for entry, value, name in zip(abi_inputs, matmul.inputs, ("A0", "W")):
        if (entry.get("name") != name or entry.get("capture_type") != str(value.type)
                or entry.get("device_dtype") != "fp8_e4m3"):
            raise UnsupportedAcceleratorContract(f"patch partition ABI input {name} changed")
    expected_origins = [
        {
            "kind": "function_argument",
            "argument_index": argument_indices[kernel],
            "bridges": [
                {"region_id": region_id, "op": "tensor.expand_shape",
                 "execution": "host_preprocess"},
                {"region_id": region_id, "op": "tensor.collapse_shape",
                 "execution": "host_preprocess"},
            ],
        },
        {
            "kind": "function_argument",
            "argument_index": argument_indices[image],
            "bridges": [
                {"region_id": region_id, "op": "tensor.expand_shape",
                 "execution": "host_preprocess"},
                {"region_id": region_id, "op": "tensor.collapse_shape",
                 "execution": "host_preprocess"},
                {"region_id": region_id, "op": "linalg.generic",
                 "execution": "host_preprocess"},
            ],
        },
    ]
    if [entry.get("origin") for entry in abi_inputs] != expected_origins:
        raise UnsupportedAcceleratorContract("patch preprocessing ABI origin changed")
    if (abi_output[0].get("name") != "Y0"
            or abi_output[0].get("capture_type") != str(matmul.results[0].type)
            or abi_output[0].get("device_dtype") != "bf16"):
        raise UnsupportedAcceleratorContract("patch partition ABI output changed")

    image_record = contraction_ops[0]
    affine_map = [
        str(expr) for expr in list(image_record.indexing_maps)[0].data.results
    ]
    source_preprocessing = {
        "schema": "atlas_patch_im2col_preprocess_v1",
        "image_shape": list(image_shape),
        "kernel_shape": list(kernel_shape),
        "kernel_matrix": {"tensor": "A0", "shape": [m, k]},
        "patch_matrix": {"tensor": "W", "shape": [k, n]},
        "affine_map": affine_map,
        "stride": [kernel_h, kernel_w],
        "padding": [0, 0],
        "dilation": [1, 1],
    }
    host_postprocessing = {
        "schema": "atlas_patch_output_postprocess_v1",
        "matrix_shape": [m, n],
        "output_shape": list(output_shape),
        "bias_shape": list(bias_shape),
        "rule": "reshape channel-major matrix to NCHW and add per-channel f32 bias",
    }

    image_receipt = partition.get("image", {})
    if (image_receipt.get("kernel_id") != partition.get("kernel_id")
            or image_receipt.get("command_count") != 4
            or not image_receipt.get("fits_imem")
            or not 0 < int(image_receipt.get("instruction_words", 0))
            <= int(image_receipt.get("imem_words", 0))):
        raise UnsupportedAcceleratorContract("patch compiled image receipt is incomplete")
    if (receipt.get("kernel_id") != partition.get("kernel_id")
            or receipt.get("kind") != "matmul"
            or receipt.get("geometry") != partition.get("geometry")
            or receipt.get("bias_fused") is not False
            or not receipt.get("fits_imem")):
        raise UnsupportedAcceleratorContract("patch kernel library receipt differs from partition")
    command_record = _validate_command_buffer(partition, command)
    signature = {
        "schema": "atlas_patch_im2col_capture_command_contract_v1",
        "kind": "matmul",
        "partition_id": partition["partition_id"],
        "kernel_id": partition["kernel_id"],
        "source_semantic": partition["source_semantic"],
        "source_dtype": "f32",
        "fqn": partition["fqn"],
        "capture_regions": capture_regions,
        "geometry": partition["geometry"],
        "bias_fused": False,
        "input_origins": [entry["origin"]["kind"] for entry in abi_inputs],
        "capture_types": [entry["capture_type"] for entry in abi_inputs]
        + [abi_output[0]["capture_type"]],
        "device_dtypes": ["fp8_e4m3", "fp8_e4m3", "bf16"],
        "source_preprocessing": source_preprocessing,
        "host_postprocessing": host_postprocessing,
        "command": command_record,
        "image": {
            "instruction_words": image_receipt["instruction_words"],
            "imem_words": image_receipt["imem_words"],
            "assembly_sha256": receipt["assembly_sha256"],
            "interface_sha256": receipt["interface_sha256"],
        },
        "qualification_scope": (
            "static source/preprocessing/ABI/command agreement only; not calibration, "
            "encoded-image execution, physical partition qualification, or E2E"
        ),
    }
    return AcceleratorContract(partition["partition_id"], signature)


def _extract_batched_contract(
    partition: dict, operations: Mapping[str, tuple], command: dict, receipt: dict,
) -> AcceleratorContract:
    if partition.get("kind") != "matmul_batched":
        raise UnsupportedAcceleratorContract("partition is not a batched matmul")
    if partition.get("source_semantic") != "batch_matmul":
        raise UnsupportedAcceleratorContract("batched source semantic changed")
    if partition.get("bias_fused"):
        raise UnsupportedAcceleratorContract("batched bias epilogue is not qualified")
    capture_regions = partition.get("capture_regions", [])
    if len(capture_regions) != 1:
        raise UnsupportedAcceleratorContract("batched capture region count changed")
    contraction_ops = operations.get(capture_regions[0], ())
    if tuple(op.name for op in contraction_ops) != (
        "arith.constant", "tensor.splat", "linalg.generic"
    ):
        raise UnsupportedAcceleratorContract(
            "batched source is not one isolated initialized contraction"
        )
    matmul = contraction_ops[-1]
    if len(matmul.inputs) != 2 or len(matmul.outputs) != 1 or len(matmul.results) != 1:
        raise UnsupportedAcceleratorContract("source batched matmul arity changed")
    batch, m, k, n = (
        int(partition["geometry"][key]) for key in ("B", "M", "K", "N")
    )
    values = (*matmul.inputs, matmul.results[0])
    source_dtype = _tensor_dtype(matmul.inputs[0])
    if ([_tensor_shape(value) for value in values]
            != [(batch, m, k), (batch, k, n), (batch, m, n)]
            or source_dtype not in {"f32", "bf16"}
            or any(_tensor_dtype(value) != source_dtype for value in values)):
        raise UnsupportedAcceleratorContract(
            "source batched matmul shape or dtype changed"
        )
    if not _zero_splat_initialized(matmul.outputs[0]):
        raise UnsupportedAcceleratorContract(
            "source batched matmul accumulator is not proven zero"
        )
    maps = list(matmul.indexing_maps)
    if len(maps) != 3:
        raise UnsupportedAcceleratorContract("batched affine map arity changed")
    map_records = [
        _general_affine_map_signature(mapping, shape, 4)
        for mapping, shape in zip(
            maps, ((batch, m, k), (batch, k, n), (batch, m, n))
        )
    ]
    expected_maps = [
        [
            {"kind": "dim", "position": 0, "extent": batch},
            {"kind": "dim", "position": 1, "extent": m},
            {"kind": "dim", "position": 3, "extent": k},
        ],
        [
            {"kind": "dim", "position": 0, "extent": batch},
            {"kind": "dim", "position": 3, "extent": k},
            {"kind": "dim", "position": 2, "extent": n},
        ],
        [
            {"kind": "dim", "position": 0, "extent": batch},
            {"kind": "dim", "position": 1, "extent": m},
            {"kind": "dim", "position": 2, "extent": n},
        ],
    ]
    if map_records != expected_maps:
        raise UnsupportedAcceleratorContract("batched affine maps changed")
    iterators = [
        getattr(item.data, "value", str(item.data)) for item in matmul.iterator_types
    ]
    if iterators != ["parallel", "parallel", "parallel", "reduction"]:
        raise UnsupportedAcceleratorContract("batched iterator roles changed")
    block = matmul.body.blocks[0]
    body = list(block.ops)
    if (len(block.args) != 3
            or tuple(op.name for op in body)
            != ("arith.mulf", "arith.addf", "linalg.yield")
            or tuple(body[0].operands) != tuple(block.args[:2])
            or tuple(body[1].operands) != (block.args[2], body[0].results[0])
            or tuple(body[2].operands) != tuple(body[1].results)):
        raise UnsupportedAcceleratorContract(
            "batched scalar multiply-accumulate dataflow changed"
        )
    abi_inputs = partition.get("abi", {}).get("inputs", [])
    abi_output = partition.get("abi", {}).get("outputs", [])
    if len(abi_inputs) != 2 or len(abi_output) != 1:
        raise UnsupportedAcceleratorContract("batched partition ABI arity changed")
    for entry, value, name in zip(abi_inputs, matmul.inputs, ("A0", "W")):
        if (entry.get("name") != name
                or entry.get("capture_type") != str(value.type)
                or entry.get("device_dtype") != "fp8_e4m3"):
            raise UnsupportedAcceleratorContract(
                f"batched partition ABI input {name} changed"
            )
    if (abi_output[0].get("name") != "Y0"
            or abi_output[0].get("capture_type") != str(matmul.results[0].type)
            or abi_output[0].get("device_dtype") != "bf16"):
        raise UnsupportedAcceleratorContract("batched partition ABI output changed")
    image = partition.get("image", {})
    if (image.get("kernel_id") != partition.get("kernel_id")
            or image.get("command_count") != 3 or not image.get("fits_imem")
            or not 0 < int(image.get("instruction_words", 0))
            <= int(image.get("imem_words", 0))):
        raise UnsupportedAcceleratorContract("batched compiled image receipt is incomplete")
    if (receipt.get("kernel_id") != partition.get("kernel_id")
            or receipt.get("kind") != "matmul_batched"
            or receipt.get("geometry") != partition.get("geometry")
            or receipt.get("bias_fused") is not False
            or not receipt.get("fits_imem")):
        raise UnsupportedAcceleratorContract(
            "batched kernel library receipt differs from partition"
        )
    command_record = _validate_batched_command_buffer(partition, command)
    signature = {
        "schema": "atlas_batched_capture_command_contract_v1",
        "kind": "matmul_batched",
        "partition_id": partition["partition_id"],
        "kernel_id": partition["kernel_id"],
        "source_semantic": partition["source_semantic"],
        "source_dtype": source_dtype,
        "fqn": partition["fqn"],
        "capture_regions": capture_regions,
        "geometry": partition["geometry"],
        "bias_fused": False,
        "input_origins": [entry["origin"]["kind"] for entry in abi_inputs],
        "capture_types": [entry["capture_type"] for entry in abi_inputs]
        + [abi_output[0]["capture_type"]],
        "device_dtypes": ["fp8_e4m3", "fp8_e4m3", "bf16"],
        "command": command_record,
        "image": {
            "instruction_words": image["instruction_words"],
            "imem_words": image["imem_words"],
            "assembly_sha256": receipt["assembly_sha256"],
            "interface_sha256": receipt["interface_sha256"],
        },
        "qualification_scope": (
            "static source/ABI/command agreement only; not calibration, encoded-image "
            "execution, physical partition qualification, or E2E"
        ),
    }
    return AcceleratorContract(partition["partition_id"], signature)


class AcceleratorContractLane:
    """Qualify exact static capture/command contracts without promoting execution."""

    def __init__(self, workload, partition_plan: dict, command_buffers: Mapping[str, dict]):
        funcs = [op for op in workload.module.walk() if op.name == "func.func"]
        if len(funcs) != 1:
            raise ValueError(f"expected one func.func, found {len(funcs)}")
        grouped: OrderedDict[str, list] = OrderedDict()
        for op in funcs[0].body.blocks[0].ops:
            region_id = _str_attr(op, "prov.region_id")
            if region_id:
                grouped.setdefault(region_id, []).append(op)
        operations = {key: tuple(value) for key, value in grouped.items()}
        host_programs = HostSemanticLane(workload).programs
        argument_indices = {
            argument: index for index, argument in enumerate(funcs[0].body.blocks[0].args)
        }
        receipts = {
            row["kernel_id"]: row for row in partition_plan.get("kernel_library", [])
        }
        self.contracts: OrderedDict[str, AcceleratorContract] = OrderedDict()
        self.rejections: OrderedDict[str, str] = OrderedDict()
        self.partitions = {
            row["partition_id"]: row for row in partition_plan.get("partitions", [])
        }
        for partition_id, partition in self.partitions.items():
            kernel_id = partition.get("kernel_id")
            try:
                if kernel_id not in command_buffers or kernel_id not in receipts:
                    raise UnsupportedAcceleratorContract(
                        "partition lacks a saved command buffer or kernel receipt"
                    )
                if partition.get("source_semantic") == "convolution_im2col_matmul":
                    self.contracts[partition_id] = _extract_patch_im2col_contract(
                        partition,
                        command_buffers[kernel_id],
                        receipts[kernel_id],
                        host_programs,
                        argument_indices,
                    )
                    continue
                extractor = {"matmul": _extract_rank2_contract,
                             "matmul_batched": _extract_batched_contract}.get(
                    partition.get("kind")
                )
                if extractor is None:
                    raise UnsupportedAcceleratorContract(
                        "partition kind has no accelerator contract"
                    )
                self.contracts[partition_id] = extractor(
                    partition, operations, command_buffers[kernel_id], receipts[kernel_id]
                )
            except UnsupportedAcceleratorContract as error:
                self.rejections[partition_id] = str(error)

    def signature_for(self, partition_id: str) -> dict | None:
        contract = self.contracts.get(partition_id)
        return contract.signature if contract is not None else None

    def execute_device_domain(
        self, partition_id: str, values: MutableMapping[str, np.ndarray]
    ) -> np.ndarray:
        """Execute the signed mathematical contract on decoded device-domain values."""
        if partition_id not in self.contracts:
            raise UnsupportedAcceleratorContract(
                self.rejections.get(partition_id, f"unknown partition {partition_id}")
            )
        signature = self.contracts[partition_id].signature
        geometry = signature["geometry"]
        if signature["kind"] == "matmul_batched":
            expected = {
                "A0": (
                    geometry["B"], geometry["M"], geometry["K"]
                ),
                "W": (
                    geometry["B"], geometry["K"], geometry["N"]
                ),
            }
        else:
            expected = {
                "A0": (geometry["M"], geometry["K"]),
                "W": (geometry["K"], geometry["N"]),
            }
        if signature["bias_fused"]:
            expected["B"] = (geometry["N"],)
        arrays = {}
        for name, shape in expected.items():
            if name not in values:
                raise ValueError(f"missing device-domain tensor {name}")
            array = np.asarray(values[name])
            if array.dtype != np.float32 or array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"device-domain tensor {name} has wrong shape/dtype/values")
            arrays[name] = array
        result = np.matmul(arrays["A0"], arrays["W"], dtype=np.float32)
        if signature["bias_fused"]:
            result = np.asarray(result + arrays["B"], dtype=np.float32)
        _, rounded = f32_to_bf16_rne(result)
        values["Y0"] = rounded
        return rounded

    def prepare_capture_inputs(
        self, partition_id: str, values: Mapping[str, np.ndarray], *, code_cap: float = 448.0,
    ) -> dict:
        """Implement the signed f32->FP8/BF16 capture conversion semantics."""
        if partition_id not in self.contracts:
            raise UnsupportedAcceleratorContract(
                self.rejections.get(partition_id, f"unknown partition {partition_id}")
            )
        signature = self.contracts[partition_id].signature
        geometry = signature["geometry"]
        a = np.asarray(values.get("A0"))
        w = np.asarray(values.get("W"))
        if signature["kind"] == "matmul_batched":
            a_shape = (geometry["B"], geometry["M"], geometry["K"])
            w_shape = (geometry["B"], geometry["K"], geometry["N"])
        else:
            a_shape = (geometry["M"], geometry["K"])
            w_shape = (geometry["K"], geometry["N"])
        if (a.dtype != np.float32 or a.shape != a_shape
                or w.dtype != np.float32 or w.shape != w_shape):
            raise ValueError("capture A0/W shape or dtype differs from signed contract")
        if signature["source_dtype"] == "bf16":
            _, rounded_a = f32_to_bf16_rne(a)
            _, rounded_w = f32_to_bf16_rne(w)
            if not np.array_equal(a, rounded_a) or not np.array_equal(w, rounded_w):
                raise ValueError("BF16 capture input is not exactly representable as BF16")
        qa = calibrate_e4m3("A0", a, code_cap)
        qw = calibrate_e4m3("W", w, code_cap)
        output_scale = qa["scale"] * qw["scale"]
        preloads = {"A0": qa["codes"].tobytes(), "W": qw["codes"].tobytes()}
        decoded = {"A0": qa["decoded"], "W": qw["decoded"]}
        record = {
            "schema": (
                "atlas_batched_capture_conversion_v1"
                if signature["kind"] == "matmul_batched"
                else "atlas_rank2_capture_conversion_v1"
            ),
            "partition_id": partition_id,
            "activation": qa["record"],
            "weight": qw["record"],
            "output_scale": output_scale,
            "output_scale_hex": float(output_scale).hex(),
        }
        if signature["bias_fused"]:
            bias = np.asarray(values.get("B"))
            if bias.dtype != np.float32 or bias.shape != (geometry["N"],):
                raise ValueError("capture B shape or dtype differs from signed contract")
            words, bias_domain = f32_to_bf16_rne(bias / np.float32(output_scale))
            preloads["B"] = words.tobytes()
            decoded["B"] = bias_domain
            record["bias_equation"] = "BF16_RNE(B / (sA * sW))"
        return {"preloads": preloads, "decoded": decoded, "record": record}

    def materialize_source_inputs(
        self, partition_id: str, values: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Materialize a signed source preprocessing boundary into command tensors."""
        if partition_id not in self.contracts:
            raise UnsupportedAcceleratorContract(
                self.rejections.get(partition_id, f"unknown partition {partition_id}")
            )
        signature = self.contracts[partition_id].signature
        preprocess = signature.get("source_preprocessing")
        if preprocess is None:
            raise UnsupportedAcceleratorContract(
                f"partition {partition_id} has no source preprocessing contract"
            )
        if set(values) != {"IMAGE", "KERNEL"}:
            raise ValueError("patch preprocessing requires exactly IMAGE and KERNEL")
        image = np.asarray(values["IMAGE"])
        kernel = np.asarray(values["KERNEL"])
        if (image.dtype != np.float32
                or tuple(image.shape) != tuple(preprocess["image_shape"])
                or kernel.dtype != np.float32
                or tuple(kernel.shape) != tuple(preprocess["kernel_shape"])
                or not np.all(np.isfinite(image)) or not np.all(np.isfinite(kernel))):
            raise ValueError("patch source tensor shape/dtype/values differ from signed contract")
        batch, channels, height, width = image.shape
        _, _, kernel_h, kernel_w = kernel.shape
        out_h, out_w = height // kernel_h, width // kernel_w
        patches = image.reshape(
            batch, channels, out_h, kernel_h, out_w, kernel_w
        ).transpose(1, 3, 5, 0, 2, 4)
        a = np.ascontiguousarray(kernel.reshape(preprocess["kernel_matrix"]["shape"]))
        w = np.ascontiguousarray(patches.reshape(preprocess["patch_matrix"]["shape"]))
        return {"A0": a, "W": w}

    def materialize_source_output(
        self, partition_id: str, matrix: np.ndarray, bias: np.ndarray,
    ) -> np.ndarray:
        """Apply the signed host reshape/bias tail after a patch device result."""
        if partition_id not in self.contracts:
            raise UnsupportedAcceleratorContract(
                self.rejections.get(partition_id, f"unknown partition {partition_id}")
            )
        signature = self.contracts[partition_id].signature
        postprocess = signature.get("host_postprocessing")
        if postprocess is None:
            raise UnsupportedAcceleratorContract(
                f"partition {partition_id} has no source output postprocessing contract"
            )
        matrix = np.asarray(matrix)
        bias = np.asarray(bias)
        if (matrix.dtype != np.float32
                or tuple(matrix.shape) != tuple(postprocess["matrix_shape"])
                or bias.dtype != np.float32
                or tuple(bias.shape) != tuple(postprocess["bias_shape"])
                or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias))):
            raise ValueError("patch output matrix/bias differs from signed contract")
        _, channels, out_h, out_w = postprocess["output_shape"]
        result = matrix.reshape(channels, 1, out_h, out_w).reshape(
            postprocess["output_shape"]
        )
        return np.ascontiguousarray(
            np.asarray(result + bias.reshape(1, channels, 1, 1), dtype=np.float32)
        )

    def publish_device_output(
        self, partition_id: str, device_bf16: np.ndarray, output_scale: float,
    ) -> np.ndarray:
        """Apply the signed BF16-domain scale and restore the capture dtype."""
        if partition_id not in self.contracts:
            raise UnsupportedAcceleratorContract(
                self.rejections.get(partition_id, f"unknown partition {partition_id}")
            )
        signature = self.contracts[partition_id].signature
        geometry = signature["geometry"]
        value = np.asarray(device_bf16)
        output_shape = (geometry["M"], geometry["N"])
        if signature["kind"] == "matmul_batched":
            output_shape = (geometry["B"], *output_shape)
        if (value.dtype != np.float32
                or value.shape != output_shape
                or not np.all(np.isfinite(value))):
            raise ValueError("device BF16 output has wrong shape/dtype/values")
        _, rounded = f32_to_bf16_rne(value)
        if not np.array_equal(value, rounded):
            raise ValueError("device output is not exactly representable as BF16")
        if not math.isfinite(output_scale) or output_scale <= 0.0:
            raise ValueError("device output scale is not finite and positive")
        published = np.asarray(value * np.float32(output_scale), dtype=np.float32)
        if signature["source_dtype"] == "bf16":
            _, published = f32_to_bf16_rne(published)
        return published

    def census(self) -> dict:
        classes: OrderedDict[tuple, dict] = OrderedDict()
        for partition_id, partition in self.partitions.items():
            origins = tuple(entry["origin"]["kind"] for entry in partition["abi"]["inputs"])
            key = (partition["kernel_id"], partition["source_semantic"], origins)
            row = classes.setdefault(key, {
                "kernel_id": partition["kernel_id"],
                "kind": partition["kind"],
                "source_semantic": partition["source_semantic"],
                "input_origins": list(origins),
                "count": 0,
                "contract_qualified": 0,
                "partition_ids": [],
            })
            row["count"] += 1
            row["contract_qualified"] += int(partition_id in self.contracts)
            row["partition_ids"].append(partition_id)
        return {
            "schema": "atlas_accelerator_contract_census_v1",
            "exact_class_count": len(classes),
            "classes": list(classes.values()),
            "by_kind": dict(sorted(Counter(
                partition["kind"] for partition in self.partitions.values()
            ).items())),
            "contract_qualified_by_kind": dict(sorted(Counter(
                self.partitions[partition_id]["kind"]
                for partition_id in self.contracts
            ).items())),
            "contract_qualified_partitions": len(self.contracts),
            "contract_unqualified_partitions": len(self.rejections),
            "rejections_by_reason": dict(sorted(Counter(self.rejections.values()).items())),
        }

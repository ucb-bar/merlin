"""Resolve runtime descriptors against an unchanged typed program signature.

The contract belongs to compiler staging. Invocation values only size runtime
storage and supply native scalar arguments; they never rewrite the MLIR body.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.func import FuncOp

from ..common import mlir_query

SCHEMA = "merlin.runtime-invocation-abi.v1"
SCALAR_C = {"i1": "_Bool", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t",
            "i64": "int64_t", "f32": "float", "f64": "double"}


def _pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def resolve(model, contract_path, arguments, *, max_bytes=64 << 20):
    from .c_runtime import DT_BYTES, NP_OF
    model, contract_path, arguments = (Path(p) for p in (model, contract_path, arguments))
    original = [_pin(p) for p in (model, contract_path, arguments)]
    contract = json.loads(contract_path.read_text())
    if (contract.get("schema") not in {"merlin.compile-signature.v1", "merlin.compile-signature.v2"} or contract.get("entry") != "forward"
            or not isinstance(contract.get("inputs"), list) or not isinstance(contract.get("outputs"), list)):
        raise ValueError("runtime invocation requires an explicit staged signature v1/v2")
    module = mlir_query.parse(model)
    module.verify()
    entries = [op for op in module.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward"]
    if len(entries) != 1 or not entries[0].body.blocks:
        raise ValueError("runtime invocation requires one defined forward entry")
    function = entries[0]
    inputs, outputs = tuple(function.function_type.inputs), tuple(function.function_type.outputs)
    if len(inputs) != len(contract["inputs"]) or len(outputs) != len(contract["outputs"]):
        raise ValueError("runtime invocation contract differs from the complete program signature")
    scalars, words = {}, []
    with np.load(arguments, allow_pickle=False) as archive:
        if set(archive.files) != {f"in{i}" for i in range(len(inputs))}:
            raise ValueError("runtime invocation argument roster differs from staged inputs")
        arrays = [archive[f"in{i}"] for i in range(len(inputs))]
        for i, (type_, declared, array) in enumerate(zip(inputs, contract["inputs"], arrays, strict=True)):
            dtype = str(type_.element_type) if isinstance(type_, TensorType) else str(type_)
            expected = np.dtype(np.uint16 if dtype == "bf16" else np.float16 if dtype == "f16" else NP_OF.get(dtype, object))
            if dtype not in DT_BYTES or declared.get("dtype") != dtype or array.dtype != expected:
                raise ValueError("runtime invocation dtype differs from the actual program argument")
            if dtype == "i1" and np.any(array.view(np.uint8) > 1):
                raise ValueError("runtime Boolean inputs require canonical source storage")
            if not isinstance(type_, TensorType):
                name, bounds = declared.get("scalar"), declared.get("bounds")
                if (dtype not in SCALAR_C or declared.get("kind") != "scalar" or declared.get("shape") != []
                        or array.shape != () or name is not None and (
                            not isinstance(name, str) or not name.isidentifier() or name in scalars)):
                    raise ValueError("runtime invocation scalar differs from its staged native ABI")
                value = array.item()
                if bounds is not None:
                    if (dtype not in {"i8", "i16", "i32", "i64"} or not isinstance(bounds, list) or len(bounds) != 2
                            or any(type(v) is not int for v in bounds)
                            or not -(1 << (int(dtype[1:]) - 1)) <= bounds[0] <= value <= bounds[1] < 1 << (int(dtype[1:]) - 1)):
                        raise ValueError("runtime invocation scalar is outside its staged guarded bounds")
                if name is not None:
                    scalars[name] = value
                words.append({"index": i, "name": name, "dtype": dtype, "word": array.tobytes().hex()})
            elif declared.get("kind") == "scalar" or "scalar" in declared or "bounds" in declared:
                raise ValueError("runtime invocation cannot replace a tensor with a native scalar")

        def shape(type_, declared):
            dimensions = declared.get("shape")
            if not isinstance(dimensions, list):
                raise ValueError("runtime invocation requires complete ranked dimensions")
            if not isinstance(type_, TensorType):
                return []
            # merlin_model.h stores at most eight ordinary dense dimensions.
            # Encodings need an explicit transport implementation of their own.
            if len(type_.get_shape()) > 8 or not isinstance(type_.encoding, NoneAttr):
                raise ValueError("runtime invocation requires unencoded tensors of rank at most eight")
            if len(dimensions) != len(type_.get_shape()) or declared.get("dtype") != str(type_.element_type):
                raise ValueError("runtime invocation dimensions differ from the actual tensor type")
            concrete = []
            for actual, dimension in zip(type_.get_shape(), dimensions, strict=True):
                size = scalars.get(dimension) if isinstance(dimension, str) else dimension
                if (type(size) is not int or not 0 <= size < 1 << 62
                        or actual >= 0 and (type(dimension) is not int or dimension != actual)
                        or actual < 0 and not isinstance(dimension, str)):
                    raise ValueError("runtime invocation dimension lacks an exact staged scalar binding")
                concrete.append(size)
            return concrete

        resolved_inputs = []
        for type_, declared, array in zip(inputs, contract["inputs"], arrays, strict=True):
            dimensions = shape(type_, declared)
            if list(array.shape) != dimensions:
                raise ValueError("runtime invocation tensor shape disagrees with its scalar dimensions")
            resolved_inputs.append({"shape": dimensions, "dtype": declared["dtype"],
                "kind": "tensor" if isinstance(type_, TensorType) else "scalar"})
        resolved_outputs = []
        for type_, declared in zip(outputs, contract["outputs"], strict=True):
            if (not isinstance(type_, TensorType) or str(type_.element_type) not in DT_BYTES
                    or declared.get("kind") == "scalar"):
                raise ValueError("runtime invocation requires ranked tensor results")
            resolved_outputs.append({"shape": shape(type_, declared), "dtype": declared["dtype"]})
        if (type(max_bytes) is not int or max_bytes < 1
                or sum(array.nbytes for array in arrays) + sum(math.prod(row["shape"]) * DT_BYTES[row["dtype"]]
                    for row in resolved_outputs) > max_bytes):
            raise ValueError("runtime invocation exceeds its complete value byte budget")
    if original != [_pin(p) for p in (model, contract_path, arguments)]:
        raise ValueError("runtime invocation sources changed during ABI derivation")
    return {"schema": SCHEMA, "program": original[0], "contract": original[1],
        "arguments": original[2], "inputs": resolved_inputs, "outputs": resolved_outputs,
        "scalars": words, "mlir_argument_specialization": "not-performed"}


def call_bridge(scalars, count, *, ciface_name, invoke_name, observe_effects=False):
    """Use private scalar storage, preserving the real typed MLIR C prototype."""
    declarations, arguments, loads = [], [], []
    for i in range(count):
        if i not in scalars:
            declarations.append("void*")
            arguments.append(f"d[{i}]")
            continue
        type_ = SCALAR_C[scalars[i]]
        declarations.append(type_)
        loads.extend([f"  {type_} scalar_{i};", f"  const merlin_descriptor_t *arg_{i} = (const merlin_descriptor_t *)d[{i}];",
            f"  memcpy(&scalar_{i}, (const char *)arg_{i}->aligned + arg_{i}->offset * sizeof(scalar_{i}), sizeof(scalar_{i}));"])
        arguments.append(f"scalar_{i}")
    return ["/* Generated native scalar argument bridge; no MLIR argument rewriting. */",
        '#include "merlin_model.h"', "#include <string.h>",
        *(["extern void merlin_effect_begin(void);", "extern void merlin_effect_return(void);"] if observe_effects else []),
        f"extern void _mlir_ciface_{ciface_name}({','.join(declarations)});",
        f"void {invoke_name}(void **d) {{", *loads,
        *(["  merlin_effect_begin();"] if observe_effects else []),
        f"  _mlir_ciface_{ciface_name}({','.join(arguments)});",
        *(["  merlin_effect_return();"] if observe_effects else []), "}"]

"""Bound an actual integer contraction; never substitute target arithmetic or a schedule.

Extraction and independent source evaluation only. The caller must establish
before/after emitted-route correspondence and separately admit a bounded runtime.
"""
from __future__ import annotations

import hashlib
from math import prod
from typing import Sequence

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.host_source_witness import _clone_bounded_reduction, evaluate_pointwise_source


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _entry(module, entry):
    functions = [op for op in module.body.block.ops
                 if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("one explicit single-block source entry is required")
    return functions[0]


def _contract(op):
    """Recognize the complete typed multiply/add recurrence, not provenance tags."""
    if op.name not in {"linalg.matmul", "linalg.generic"} or len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError("selected operation is not a rank-two integer contraction")
    try:
        op.verify()
    except Exception as error:
        raise ValueError("source contraction does not verify") from error
    values = [*op.operands, *op.results]
    if any(not isinstance(value.type, TensorType) or len(value.type.get_shape()) != 2
           or str(value.type.get_element_type()) not in {"i8", "i16", "i32", "i64"} for value in values):
        raise ValueError("contraction requires exact static rank-two signless integer tensors")
    shapes = [tuple(value.type.get_shape()) for value in values]
    (m, k), (kk, n), output, result = shapes
    if min(m, k, kk, n) <= 0 or kk != k or output != (m, n) or result != output or values[2].type != values[3].type:
        raise ValueError("contraction tensor domains or initialized output types disagree")
    maps = [item.data for item in op.get_indexing_maps()]
    if (len(maps) != 3 or any(amap.num_symbols or amap.num_dims != 3 for amap in maps)
            or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
            or [[expr.position for expr in amap.results] for amap in maps] != [[0, 2], [2, 1], [0, 1]]
            or [item.data.value for item in op.get_iterator_types()] != ["parallel", "parallel", "reduction"]):
        raise ValueError("contraction requires canonical m,n,k affine maps and iterator order")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("contraction has unsupported scalar control flow")
    block = op.regions[0].block
    operations = list(block.ops)
    if (len(block.args) != 3 or not operations or operations[-1].name != "linalg.yield"
            or len(operations[-1].operands) != 1):
        raise ValueError("contraction scalar argument/yield contract is malformed")
    allowed = {"arith.extsi", "arith.extui", "arith.trunci", "arith.muli", "arith.addi", "linalg.yield"}
    for scalar in operations:
        if scalar.name not in allowed or scalar.regions:
            raise ValueError("noncanonical scalar contraction body")
        flags = scalar.properties.get("overflowFlags", scalar.attributes.get("overflowFlags"))
        if flags is not None and getattr(flags, "data", None) != frozenset():
            raise ValueError("poison-producing overflow flags are not modular integer arithmetic")
    visited = {operations[-1]}
    addition = operations[-1].operands[0].owner
    if (not isinstance(addition, Operation) or addition not in operations or addition.name != "arith.addi"
            or block.args[2] not in addition.operands):
        raise ValueError("contraction must add the actual initialized accumulator")
    visited.add(addition)
    term = addition.operands[1 - list(addition.operands).index(block.args[2])]
    multiply = term.owner
    if not isinstance(multiply, Operation) or multiply not in operations or multiply.name != "arith.muli":
        raise ValueError("contraction accumulator term is not the direct source product")
    visited.add(multiply)

    def input_of(value):
        while value not in block.args:
            cast = value.owner
            if (not isinstance(cast, Operation) or cast not in operations
                    or cast.name not in {"arith.extsi", "arith.extui", "arith.trunci"}
                    or len(cast.operands) != 1 or len(cast.results) != 1):
                raise ValueError("contraction product has an unsupported operand expression")
            visited.add(cast)
            value = cast.operands[0]
        return value

    if {input_of(value) for value in multiply.operands} != set(block.args[:2]) or visited != set(operations):
        raise ValueError("contraction product/accumulator def-use is noncanonical or contains extra work")
    return {"maps": maps, "n_in": 2, "iteration_shape": (m, n, k),
            "parallel_dimensions": [0, 1], "reduction_dimensions": [2]}, (m, k, n)


def extract_source_contraction(source_text: str, source_op_index: int, *, entry: str,
                               max_m: int, max_n: int, max_k: int,
                               max_macs: int = 100000) -> tuple[str, dict]:
    """Clone one source operation with host-selected bounds, preserving named/generic spelling.

    Bounds are safety limits, not inferred target tile geometry. The caller must
    check whether the reduced compilation still exercises the changed mechanism.
    """
    if (not isinstance(entry, str) or not entry or type(source_op_index) is not int
            or any(type(value) is not int or not 1 <= value <= 4096 for value in (max_m, max_n, max_k))
            or type(max_macs) is not int or not 1 <= max_macs <= 100000
            or len(source_text.encode()) > 2_000_000):
        raise ValueError("invalid bounded source contraction request")
    function = _entry(parse_mlir_text(source_text), entry)
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    if not 0 <= source_op_index < len(ops):
        raise ValueError("contraction source index is outside the explicit entry")
    operation = ops[source_op_index]
    info, geometry = _contract(operation)
    m, k, n = (min(old, bound) for old, bound in zip(geometry, (max_m, max_k, max_n)))
    if m*k*n > max_macs or (m, k, n) == geometry:
        raise ValueError("contraction witness must strictly reduce source work within the host MAC budget")
    element_widths = [int(str(value.type.get_element_type())[1:]) for value in operation.operands]
    if sum(prod(shape)*width//8 for shape, width in zip(((m,k),(k,n),(m,n)), element_widths)) > 65536:
        raise ValueError("contraction witness exceeds the logical operand byte budget")
    text, record = _clone_bounded_reduction(source_text, function, operation, info, (m,n,k), entry=entry)
    record.update(schema="actual_source_contraction_witness_v1", mechanism="integer_contraction",
                  entry=entry, source_op_index=source_op_index, source_op_name=operation.name,
                  source_geometry_mkn=list(geometry), probe_geometry_mkn=[m,k,n],
                  max_macs=max_macs, probe_macs=m*k*n,
                  input_shapes=[row["shape"] for row in record["inputs"]],
                  input_dtypes=[row["dtype"] for row in record["inputs"]],
                  output_shape=record["output"]["shape"], output_dtype=record["output"]["dtype"],
                  arithmetic="actual source casts, multiply and initialized add; modular wrap after each typed integer op",
                  emitted_route_correspondence="UNPROVEN", numerical_qualification="UNPROVEN",
                  runtime_admitted=False, full_model_executed=False,
                  scope="one exact source contraction and initializer at reduced extents; not a complete layer or full-shape proof")
    return text, record


def evaluate_source_contraction(source_text: str, extraction: dict, inputs: Sequence):
    """Evaluate only the hash-bound short source through the shared independent scalar oracle."""
    import numpy as np

    if (extraction.get("schema") != "actual_source_contraction_witness_v1"
            or _sha(source_text) != extraction.get("probe_source_sha256")
            or type(extraction.get("probe_macs")) is not int
            or not 0 < extraction["probe_macs"] <= 100000):
        raise ValueError("stale or unbounded contraction reference contract")
    function = _entry(parse_mlir_text(source_text), extraction["entry"])
    if len(inputs) != len(function.body.block.args):
        raise ValueError("contraction reference argument count mismatch")
    expected = []
    for value, data in zip(function.body.block.args, inputs):
        dtype = str(value.type.get_element_type())
        array = np.asarray(data)
        if (list(array.shape) != list(value.type.get_shape()) or dtype not in {"i8", "i16", "i32", "i64"}
                or array.dtype != np.dtype("int"+dtype[1:])):
            raise ValueError("contraction reference needs exact typed logical input bytes/shapes")
        expected.append(array)
    if (extraction["input_shapes"] != [list(array.shape) for array in expected]
            or extraction["input_dtypes"] != [str(value.type.get_element_type()) for value in function.body.block.args]):
        raise ValueError("contraction reference input metadata changed")
    operation = next(op for op in function.body.block.ops if op.name in {"linalg.matmul", "linalg.generic"})
    _, geometry = _contract(operation)
    if list(geometry) != extraction["probe_geometry_mkn"] or prod(geometry) != extraction["probe_macs"]:
        raise ValueError("contraction reference work metadata changed")
    if str(operation.results[0].type.get_element_type()) != extraction["output_dtype"]:
        raise ValueError("contraction reference output dtype metadata changed")
    outputs = evaluate_pointwise_source(source_text, expected)
    if len(outputs) != 1 or list(outputs[0].shape) != extraction["output_shape"]:
        raise ValueError("contraction reference output metadata changed")
    return outputs[0]

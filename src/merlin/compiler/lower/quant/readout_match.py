"""Shared structural recognition of static scalar/channel f32 readout.

The matcher reads operation semantics, parameter maps and immutable constants.
It performs no whole-program source coverage, partitioning, backend selection or
compiler invocation. Host source preparation and the numerical proof adapters
consume this same matcher object.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.ir import Operation

from merlin.compiler.ir.readout import constant_parameter
from merlin.compiler.lower.quant.requantization import match_integer_gemm


def _tensor_spec(value):
    tensor = value.type
    if not isinstance(tensor, TensorType) or not isinstance(tensor.encoding, NoneAttr):
        raise ValueError("readout requires an ordinary ranked tensor")
    if any(dimension <= 0 for dimension in tensor.get_shape()):
        raise ValueError("readout requires positive static dimensions")
    return {"shape": list(tensor.get_shape()), "dtype": str(tensor.element_type)}


@dataclass(frozen=True)
class F32ReadoutMatch:
    contraction: Operation
    epilogue: Operation
    dimensions: object
    parameters: tuple
    steps: tuple[tuple[str, int], ...]
    relu: bool


def match_f32_readout(epilogue):
    """Read the retained scalar SSA pattern with exact scalar/channel constants.

    Index maps establish the channel axis. Values are retained as actual SSA
    constants, never inferred from equal sizes or replaced by synthesized bias.
    No target scale selection, preload conversion or accuracy policy is applied.
    """
    from xdsl.ir.affine import AffineExpr, AffineMap

    op = epilogue
    if op.name != "linalg.generic" or len(op.operands) < 2 or len(op.results) != 1:
        raise ValueError("expected one pointwise f32 readout")
    contraction = op.operands[0].owner
    dimensions = match_integer_gemm(contraction) if isinstance(contraction, Operation) else None
    if dimensions is None or dimensions.batch != 1:
        raise ValueError("readout requires one exact rank-two signed integer GEMM")
    if len(list(contraction.results[0].uses)) != 1:
        raise ValueError("accumulator has another consumer")
    output = _tensor_spec(op.results[0])
    if output != {"shape": [dimensions.m, dimensions.n], "dtype": "i8"} or op.operands[-1].type != op.results[0].type:
        raise ValueError("readout output geometry or signed-i8 storage differs")
    inputs = tuple(op.operands[:-1])
    maps = op.properties.get("indexing_maps")
    expected = [AffineMap.identity(2)]
    for value in inputs[1:]:
        _, shape = constant_parameter(value)
        if shape == ():
            expected.append(AffineMap(2, 0, ()))
        elif shape == (dimensions.n,):
            expected.append(AffineMap(2, 0, (AffineExpr.dimension(1),)))
        else:
            raise ValueError("readout parameter is not the exact mapped output-channel vector")
    expected.append(AffineMap.identity(2))
    if maps is None or [attribute.data for attribute in maps] != expected:
        raise ValueError("readout requires exact identity data and final-axis parameter maps")
    iterators = op.properties.get("iterator_types")
    if iterators is None or [attribute.data.value for attribute in iterators] != ["parallel", "parallel"]:
        raise ValueError("readout is not pointwise")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("readout requires one exact scalar region")
    block = op.regions[0].block
    if len(block.args) != len(op.operands):
        raise ValueError("readout scalar signature differs from tensor inputs and output")
    for operation in block.ops:
        flags = operation.properties.get("fastmath")
        if flags is not None and str(flags) != "#arith.fastmath<none>":
            raise ValueError("readout fastmath assumptions require a separate preservation proof")
    operations = [operation for operation in block.ops if operation.name != "arith.constant"]
    if (
        not operations
        or operations[0].name != "arith.sitofp"
        or tuple(operations[0].operands) != (block.args[0],)
        or str(operations[0].results[0].type) != "f32"
    ):
        raise ValueError("readout must convert its signed-i32 accumulator to binary32")
    current = operations.pop(0).results[0]
    parameters, steps, biases = [], [], 0

    def parameter(value):
        if value in block.args:
            index = block.args.index(value)
            if not 0 < index < len(inputs):
                raise ValueError("readout scalar parameter is not a declared immutable tensor input")
            value = inputs[index]
        values, shape = constant_parameter(value)
        if shape not in ((), (dimensions.n,)):
            raise ValueError("readout scalar parameter has an unsupported shape")
        if value not in parameters:
            parameters.append(value)
        return parameters.index(value), values, shape

    while operations and operations[0].name in ("arith.mulf", "arith.addf"):
        operation = operations.pop(0)
        if len(operation.operands) != 2 or current not in operation.operands or str(operation.results[0].type) != "f32":
            raise ValueError("readout binary32 chain is disconnected")
        other = operation.operands[1] if operation.operands[0] is current else operation.operands[0]
        index, values, shape = parameter(other)
        kind = "scale" if operation.name == "arith.mulf" else "bias"
        if kind == "scale" and any(value <= 0 for value in values):
            raise ValueError("readout scales must be positive finite binary32")
        if kind == "bias":
            biases += 1
            if biases > 1 or shape != (dimensions.n,) or not steps:
                raise ValueError("readout supports one mapped f32 channel bias after source scales")
        steps.append((kind, index))
        current = operation.results[0]
    if not steps or not any(kind == "scale" for kind, _ in steps):
        raise ValueError("readout requires its original ordered f32 scale")
    if not operations or operations[0].name != "math.roundeven" or tuple(operations[0].operands) != (current,):
        raise ValueError("readout requires nearest-even rounding after its complete scale/bias chain")
    current = operations.pop(0).results[0]
    if [operation.name for operation in operations] != [
        "arith.maximumf",
        "arith.minimumf",
        "arith.fptosi",
        "linalg.yield",
    ]:
        raise ValueError("readout requires exact round, lower clamp, upper clamp and signed-i8 narrowing order")
    lower, upper, cast, yielded = operations

    def bound(operation, value):
        if len(operation.operands) != 2 or value not in operation.operands:
            raise ValueError("readout clamp is disconnected")
        other = operation.operands[1] if operation.operands[0] is value else operation.operands[0]
        elements, shape = constant_parameter(other)
        if shape:
            raise ValueError("readout clamp bounds must be original scalar f32 constants")
        return elements[0]

    minimum = bound(lower, current)
    maximum = bound(upper, lower.results[0])
    if (
        minimum not in (-128.0, 0.0)
        or maximum != 127.0
        or tuple(cast.operands) != (upper.results[0],)
        or str(cast.results[0].type) != "i8"
        or tuple(yielded.operands) != (cast.results[0],)
    ):
        raise ValueError("readout does not preserve the exact signed-i8 clamp/conversion")
    return F32ReadoutMatch(contraction, op, dimensions, tuple(parameters), tuple(steps), minimum == 0)

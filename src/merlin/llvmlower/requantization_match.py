"""Typed source requantization analysis without device schedule selection."""

from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap

from .integer_gemm import match_integer_gemm
from .requantization import f32, prove_scale, synthesize_bias


def _constant(value):
    op = value.owner
    if not isinstance(op, Operation) or op.name != "arith.constant" or str(value.type) != "f32":
        raise ValueError("epilogue needs uniform f32 constants")
    attr = op.properties.get("value", op.attributes.get("value"))
    data = getattr(getattr(attr, "value", None), "data", None)
    if data is None:
        raise ValueError("not a scalar float constant")
    return f32(data)


@dataclass
class RequantizedContraction:
    contraction: Operation
    epilogue: Operation
    dimensions: object
    relu: bool
    proof: dict
    integer_bias: list[int] | None = None


def match(epilogue):
    """Match one canonical pointwise i32→i8 scale/round/saturate region.

    Bias additions, residuals, nonuniform scales, zero-point shifts, alternate
    rounding, fastmath and shared accumulator outputs are deliberately refused.
    """
    op = epilogue
    if op.name != "linalg.generic" or len(op.operands) not in (2, 3) or len(op.results) != 1:
        raise ValueError("expected one-input pointwise requantization")
    has_bias = len(op.operands) == 3
    contraction = op.operands[0].owner
    if not isinstance(contraction, Operation):
        raise ValueError("requantization must consume a contraction")
    dims = match_integer_gemm(contraction)
    if dims is None or dims.batch != 1:
        raise ValueError("expected exact signed integer GEMM")
    if len(list(contraction.results[0].uses)) != 1:
        raise ValueError("accumulator has another consumer")
    ty = op.results[0].type
    if (
        not isinstance(ty, TensorType)
        or str(ty.get_element_type()) != "i8"
        or ty.get_shape() != (dims.m, dims.n)
        or op.operands[-1].type != ty
    ):
        raise ValueError("requantization output geometry or dtype differs")
    maps = op.properties.get("indexing_maps")
    iters = op.properties.get("iterator_types")
    expected_maps = [AffineMap.identity(2)]
    if has_bias:
        from xdsl.ir.affine import AffineExpr

        expected_maps.append(AffineMap(2, 0, (AffineExpr.dimension(1),)))
    expected_maps.append(AffineMap.identity(2))
    if maps is None or [x.data for x in maps.data] != expected_maps:
        raise ValueError("requantization needs identity tensor indexing")
    if iters is None or [x.data.value for x in iters.data] != ["parallel", "parallel"]:
        raise ValueError("requantization is not pointwise")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("noncanonical epilogue region")
    block = op.regions[0].block
    for scalar_op in block.ops:
        flags = scalar_op.properties.get("fastmath")
        if flags is not None and str(flags) != "#arith.fastmath<none>":
            raise ValueError("fastmath epilogue contract unsupported")
    ops = [x for x in block.ops if x.name != "arith.constant"]
    if (
        len(block.args) != (3 if has_bias else 2)
        or not ops
        or ops[0].name != "arith.sitofp"
        or list(ops[0].operands) != [block.args[0]]
        or str(ops[0].results[0].type) != "f32"
    ):
        raise ValueError("expected signed accumulator to f32 conversion")
    value = ops.pop(0).results[0]
    scales = []
    while ops and ops[0].name == "arith.mulf":
        mul = ops.pop(0)
        if value not in mul.operands:
            raise ValueError("scale chain disconnected")
        if str(mul.properties.get("fastmath", "#arith.fastmath<none>")) != "#arith.fastmath<none>":
            raise ValueError("fastmath scale contract unsupported")
        scales.append(_constant(mul.operands[1] if mul.operands[0] is value else mul.operands[0]))
        value = mul.results[0]
    if not scales:
        raise ValueError("missing constant scale")
    biases = None
    reciprocal = 1.0
    if has_bias:
        biasop = op.operands[1].owner
        if (
            not isinstance(biasop, Operation)
            or biasop.name != "arith.constant"
            or tuple(op.operands[1].type.get_shape()) != (dims.n,)
            or str(op.operands[1].type.get_element_type()) != "f32"
        ):
            raise ValueError("bias must be an immutable dense f32 channel constant")
        attr = biasop.properties.get("value", biasop.attributes.get("value"))
        if not hasattr(attr, "get_values"):
            raise ValueError("dense bias literal required")
        biases = list(attr.get_values())
        if not ops or ops[0].name != "arith.addf" or set(ops[0].operands) != {value, block.args[1]}:
            raise ValueError("expected bias after source scale chain")
        value = ops.pop(0).results[0]
        if ops and ops[0].name == "arith.mulf":
            mul = ops.pop(0)
            if value not in mul.operands:
                raise ValueError("output scale disconnected")
            if str(mul.properties.get("fastmath", "#arith.fastmath<none>")) != "#arith.fastmath<none>":
                raise ValueError("fastmath output scale unsupported")
            reciprocal = _constant(mul.operands[1] if mul.operands[0] is value else mul.operands[0])
            value = mul.results[0]
    if not ops or ops[0].name != "math.roundeven" or list(ops[0].operands) != [value]:
        raise ValueError("nearest-even rounding required")
    value = ops.pop(0).results[0]
    if len(ops) != 4 or [x.name for x in ops] != ["arith.maximumf", "arith.minimumf", "arith.fptosi", "linalg.yield"]:
        raise ValueError("expected exact signed i8 clamp; bias/residual/zero-point not supported")
    lower, upper, cast, yieldop = ops
    if value not in lower.operands:
        raise ValueError("clamp disconnected")
    lo = _constant(lower.operands[1] if lower.operands[0] is value else lower.operands[0])
    value = lower.results[0]
    if lo not in (-128.0, 0.0) or value not in upper.operands:
        raise ValueError("unsupported lower clamp")
    hi = _constant(upper.operands[1] if upper.operands[0] is value else upper.operands[0])
    if (
        hi != 127.0
        or list(cast.operands) != [upper.results[0]]
        or str(cast.results[0].type) != "i8"
        or list(yieldop.operands) != [cast.results[0]]
    ):
        raise ValueError("unsupported upper clamp or conversion")
    bound = dims.k * 128 * 128
    low, high = max(-(1 << 31), -bound), min((1 << 31) - 1, bound)
    proof = (
        prove_scale(scales, low, high, lo == 0)
        if biases is None
        else synthesize_bias(scales, biases, reciprocal, low, high, lo == 0)
    )
    if biases is not None and proof["accepted_channels"] != dims.n:
        raise ValueError(f"integer bias cannot preserve all channels: {proof['accepted_channels']}/{dims.n}")
    return RequantizedContraction(contraction, op, dims, lo == 0, proof, proof.get("integer_bias"))

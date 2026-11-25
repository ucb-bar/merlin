"""Recognize the declared BF16 inference batch norm body.

The body is PyTorch's CPU BF16 inference kernel in binary32: per channel,
``alpha = (1 / sqrt(var + eps)) * weight`` and ``beta = fma(-mean, alpha, bias)``;
per element, ``bf16(fma(float(x), alpha, beta))``. Each step rounds once to
nearest even. Operand roles come from the body's dataflow, not argument order.
Weight and bias are optional (``affine=False``); statistics may be BF16 or f32.

Temporary: this recognizer matches the fixed binary32 body chain that
model2MLIR's batch norm decomposition emits. Its general replacement is a
declared implementation family for per-channel affine maps, chosen by Merlin
cover selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .linalg_patterns import InvalidLinalgPattern

NUMERIC_SEMANTICS = "torch-cpu-bf16-batch-norm-inference-v1"
INPUT_DOMAIN = "finite-bf16"


@dataclass(frozen=True)
class BatchNorm:
    source: object
    channel: dict[str, object]  # role -> operand value for mean, var, and optional weight and bias
    epsilon_bits: int  # binary32 encoding of the added epsilon


def recognize_bf16_batch_norm(op) -> BatchNorm:
    """Prove maps, declaration, and the complete scalar dataflow."""
    from xdsl.dialects import arith, math
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        BFloat16Type,
        DenseArrayBase,
        Float32Type,
        FloatAttr,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(op) is not GenericOp or not 3 <= len(op.inputs) <= 5 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("batch norm needs an input, two to four channel operands, and one result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("batch norm has an unknown Linalg property or successor")
    if any(
        not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("batch norm has an unknown Linalg attribute")
    policy, domain = op.attributes.get("merlin.numeric_semantics"), op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr)
        or policy.data != NUMERIC_SEMANTICS
        or not isinstance(domain, StringAttr)
        or domain.data != INPUT_DOMAIN
    ):
        raise InvalidLinalgPattern("batch norm requires its exact numerical policy and input domain")
    segments = op.properties["operandSegmentSizes"]
    if not isinstance(segments, DenseArrayBase) or tuple(segments.iter_values()) != (len(op.inputs), 1):
        raise InvalidLinalgPattern("batch norm has an inconsistent operand segment")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr) for value in values):
        raise InvalidLinalgPattern("batch norm requires default-encoded tensors")
    shape = tuple(op.inputs[0].type.get_shape())
    rank = len(shape)
    if (
        rank < 2
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or any(tuple(value.type.get_shape()) != shape for value in (op.inputs[0], *op.outputs, *op.results))
        or any(not isinstance(value.type.element_type, BFloat16Type) for value in (op.inputs[0], *op.outputs))
        or any(tuple(value.type.get_shape()) != (shape[1],) for value in op.inputs[1:])
        or any(not isinstance(value.type.element_type, (BFloat16Type, Float32Type)) for value in op.inputs[1:])
    ):
        raise InvalidLinalgPattern("batch norm needs static BF16 activations and per-channel BF16 or f32 operands")
    dims = tuple(AffineDimExpr(i) for i in range(rank))
    identity, channel = AffineMap(rank, 0, dims), AffineMap(rank, 0, (dims[1],))
    maps = tuple(op.indexing_maps.data)
    iterators = op.iterator_types.data
    if (
        len(maps) != len(op.inputs) + 1
        or any(not isinstance(item, AffineMapAttr) for item in maps)
        or maps[0].data != identity
        or maps[-1].data != identity
        or any(item.data != channel for item in maps[1:-1])
        or any(not isinstance(item, IteratorTypeAttr) or item.data != IteratorType.PARALLEL for item in iterators)
        or len(iterators) != rank
    ):
        raise InvalidLinalgPattern("batch norm needs identity activation maps and dimension-1 channel maps")
    block = op.regions[0].block if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else None
    if block is None or len(block.args) != len(op.inputs) + 1:
        raise InvalidLinalgPattern("batch norm needs one scalar body")
    body = tuple(block.ops)
    if not body or type(body[-1]) is not YieldOp or len(body[-1].operands) != 1:
        raise InvalidLinalgPattern("batch norm body must yield one value")
    for inner in body:
        if inner.regions or inner.successors or any(not key.startswith("prov.") for key in inner.attributes):
            raise InvalidLinalgPattern("batch norm body has an unknown attribute, region, or successor")
        if set(inner.properties) - {"fastmath", "value"} or ("fastmath" in inner.properties and inner.fastmath.data):
            raise InvalidLinalgPattern("batch norm body needs strict IEEE operations")
    used = set()

    def producer(value, kind):
        owner = value.owner
        if type(owner) is not kind:
            raise InvalidLinalgPattern(f"batch norm body needs {kind.name} here")
        used.add(owner)
        return owner

    def literal(value):
        owner = producer(value, arith.ConstantOp)
        attr = owner.properties.get("value")
        if not isinstance(attr, FloatAttr) or not isinstance(attr.type, Float32Type):
            raise InvalidLinalgPattern("batch norm literal must be f32")
        return int.from_bytes(attr.type.pack((attr.value.data,)), "little")

    roles: dict[str, object] = {}

    def widened(value, role):
        argument = value
        if type(value.owner) is arith.ExtFOp:
            argument = producer(value, arith.ExtFOp).input
            if not isinstance(argument.type, BFloat16Type):
                raise InvalidLinalgPattern("batch norm widens only BF16 operands")
        elif not isinstance(value.type, Float32Type):
            raise InvalidLinalgPattern("batch norm operand must be f32 or a widened BF16 value")
        if argument not in block.args[:-1] or role in roles or argument in roles.values():
            raise InvalidLinalgPattern("batch norm operand roles must be distinct block arguments")
        roles[role] = argument
        return argument

    narrowed = producer(body[-1].operands[0], arith.TruncFOp)
    if not isinstance(narrowed.result.type, BFloat16Type):
        raise InvalidLinalgPattern("batch norm must narrow to BF16")
    element = producer(narrowed.input, math.FmaOp)
    widened(element.a, "x")
    alpha, beta = element.b, element.c
    shift = producer(beta, math.FmaOp)
    if shift.b is not alpha:
        raise InvalidLinalgPattern("batch norm shift must reuse the per-channel scale")
    widened(producer(shift.a, arith.NegfOp).operand, "mean")
    if type(shift.c.owner) is arith.ConstantOp:
        if literal(shift.c) != 0:
            raise InvalidLinalgPattern("batch norm without bias adds +0")
    else:
        widened(shift.c, "bias")
    inverse = alpha
    if type(alpha.owner) is arith.MulfOp:
        scale = producer(alpha, arith.MulfOp)
        inverse = scale.lhs
        widened(scale.rhs, "weight")
    reciprocal = producer(inverse, arith.DivfOp)
    if literal(reciprocal.lhs) != 0x3F800000:
        raise InvalidLinalgPattern("batch norm reciprocal must divide +1")
    total = producer(producer(reciprocal.rhs, math.SqrtOp).operand, arith.AddfOp)
    widened(total.lhs, "var")
    epsilon = literal(total.rhs)
    if epsilon & 0x80000000 or (epsilon & 0x7F800000) == 0x7F800000:
        raise InvalidLinalgPattern("batch norm epsilon must be a finite nonnegative f32")
    if used | {body[-1]} != set(body) or roles["x"] is not block.args[0] or len(roles) != len(op.inputs):
        raise InvalidLinalgPattern("batch norm body has unused or unaccounted scalar operations")
    if alpha.owner is not None and len(tuple(alpha.uses)) != 2:
        raise InvalidLinalgPattern("batch norm scale must feed only the shift and the element update")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("batch norm Linalg operation does not verify") from exc
    operands = dict(zip(block.args[:-1], op.inputs))
    return BatchNorm(op.inputs[0], {role: operands[arg] for role, arg in roles.items() if role != "x"}, epsilon)

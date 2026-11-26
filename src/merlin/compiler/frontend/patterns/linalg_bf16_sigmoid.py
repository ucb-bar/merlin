"""Recognize the declared BF16 sigmoid body.

The body is PyTorch's CPU BF16 kernel: widen exactly, ``1 / (1 + exp(-x))`` in
binary32 with each step rounded to nearest even, then one BF16 rounding. The
declaration names that order; this proves the Linalg body, maps, and types.

Temporary: this recognizer matches the fixed binary32 body chain that
model2MLIR's BF16 sigmoid decomposition emits. Its general replacement is a
declared implementation family for elementwise transcendental activations,
chosen by Merlin cover selection.
"""

from __future__ import annotations

from .linalg_patterns import InvalidLinalgPattern

NUMERIC_SEMANTICS = "torch-cpu-bf16-sigmoid-v1"
INPUT_DOMAIN = "non-nan-bf16"
SEMANTIC_OP = "bf16_sigmoid"


def recognize_bf16_sigmoid(op):
    """Return the input of an exact declared BF16 sigmoid ``linalg.generic``."""
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

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16 sigmoid needs one input and one tensor result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16 sigmoid has an unknown Linalg property or successor")
    if any(
        not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("BF16 sigmoid has an unknown Linalg attribute")
    policy, domain = op.attributes.get("merlin.numeric_semantics"), op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr)
        or policy.data != NUMERIC_SEMANTICS
        or not isinstance(domain, StringAttr)
        or domain.data != INPUT_DOMAIN
    ):
        raise InvalidLinalgPattern("BF16 sigmoid requires its exact numerical policy and input domain")
    segments = op.properties["operandSegmentSizes"]
    if not isinstance(segments, DenseArrayBase) or tuple(segments.iter_values()) != (1, 1):
        raise InvalidLinalgPattern("BF16 sigmoid has an inconsistent operand segment")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType)
        or not isinstance(value.type.element_type, BFloat16Type)
        or not isinstance(value.type.encoding, NoneAttr)
        for value in values
    ):
        raise InvalidLinalgPattern("BF16 sigmoid requires default-encoded BF16 tensors")
    shape = tuple(op.inputs[0].type.get_shape())
    rank = len(shape)
    if (
        not rank
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or any(tuple(value.type.get_shape()) != shape for value in values)
    ):
        raise InvalidLinalgPattern("BF16 sigmoid requires one positive static shape")
    identity = AffineMap(rank, 0, tuple(AffineDimExpr(dim) for dim in range(rank)))
    maps, iterators = tuple(op.indexing_maps.data), op.iterator_types.data
    if (
        len(maps) != 2
        or any(not isinstance(item, AffineMapAttr) or item.data != identity for item in maps)
        or len(iterators) != rank
        or any(not isinstance(item, IteratorTypeAttr) or item.data != IteratorType.PARALLEL for item in iterators)
    ):
        raise InvalidLinalgPattern("BF16 sigmoid requires parallel identity index maps")
    block = op.regions[0].block if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else None
    if block is None or len(block.args) != 2:
        raise InvalidLinalgPattern("BF16 sigmoid needs one scalar body")
    body = tuple(block.ops)
    kinds = (
        arith.ExtFOp,
        arith.ConstantOp,
        arith.NegfOp,
        math.ExpOp,
        arith.AddfOp,
        arith.DivfOp,
        arith.TruncFOp,
        YieldOp,
    )
    if tuple(type(inner) for inner in body) != kinds:
        raise InvalidLinalgPattern("BF16 sigmoid body must be extf, constant, negf, exp, addf, divf, truncf, yield")
    for inner in body:
        if inner.regions or inner.successors or any(not key.startswith("prov.") for key in inner.attributes):
            raise InvalidLinalgPattern("BF16 sigmoid body has an unknown attribute, region, or successor")
        if set(inner.properties) - {"fastmath", "value"} or ("fastmath" in inner.properties and inner.fastmath.data):
            raise InvalidLinalgPattern("BF16 sigmoid body needs strict IEEE operations")
    wide, one, neg, exp, add, div, narrow, yielded = body
    literal = one.properties.get("value")
    if (
        tuple(wide.operands) != (block.args[0],)
        or not isinstance(wide.results[0].type, Float32Type)
        or not isinstance(literal, FloatAttr)
        or not isinstance(literal.type, Float32Type)
        or literal.value.data != 1.0
        or tuple(neg.operands) != (wide.results[0],)
        or tuple(exp.operands) != (neg.results[0],)
        or tuple(add.operands) != (one.results[0], exp.results[0])
        or tuple(div.operands) != (one.results[0], add.results[0])
        or tuple(narrow.operands) != (div.results[0],)
        or not isinstance(narrow.results[0].type, BFloat16Type)
        or tuple(yielded.operands) != (narrow.results[0],)
    ):
        raise InvalidLinalgPattern("BF16 sigmoid does not yield bf16(1 / (1 + exp(-f32(x))))")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16 sigmoid Linalg operation does not verify") from exc
    return op.inputs[0]

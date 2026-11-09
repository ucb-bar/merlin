"""Recognize one exact BF16 Linalg ReLU source form with an explicit domain."""

from __future__ import annotations

from math import copysign

from .linalg_patterns import InvalidLinalgPattern


def recognize_bf16_sign_clamp_relu(op) -> tuple[object, object]:
    """Return input and destination for a parsed elementwise maximum with +0.

    The source declaration supplies a finite-normal-or-zero premise. This
    function proves the Linalg body and indexing geometry, not that runtime
    inputs satisfy the premise or that an accelerator implements the operation.
    """
    from xdsl.dialects.arith import ConstantOp, MaximumfOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        FloatAttr,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16 ReLU requires one input and one tensor result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16 ReLU has an unknown Linalg property or successor")
    if any(not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
           for key in op.attributes):
        raise InvalidLinalgPattern("BF16 ReLU has an unknown Linalg attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr) or policy.data != "bf16-sign-clamp-v1"
        or not isinstance(domain, StringAttr) or domain.data != "finite-normal-or-zero-bf16"
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires its exact numerical policy and input domain")
    segments = op.properties["operandSegmentSizes"]
    if not isinstance(segments, DenseArrayBase) or str(segments.elt_type) != "i32" or (
        tuple(segments.iter_values()) != (1, 1)
    ):
        raise InvalidLinalgPattern("BF16 ReLU has an inconsistent operand segment")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(not isinstance(value.type, TensorType) or not isinstance(value.type.element_type, BFloat16Type)
           or not isinstance(value.type.encoding, NoneAttr) for value in values):
        raise InvalidLinalgPattern("BF16 ReLU requires default-encoded BF16 tensors")
    shapes = tuple(tuple(value.type.get_shape()) for value in values)
    if len(shapes[0]) != 2 or any(shape != shapes[0] for shape in shapes) or (
        any(type(dim) is not int or dim <= 0 for dim in shapes[0])
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires one positive static rank-two shape")
    maps = op.get_indexing_maps()
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (
        len(maps) != 2 or any(not isinstance(item, AffineMapAttr) or item.data != identity for item in maps)
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(item, IteratorTypeAttr) for item in op.iterator_types)
        or tuple(item.data for item in op.iterator_types) != (IteratorType.PARALLEL,) * 2
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires parallel identity index maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("BF16 ReLU requires one scalar body")
    block = op.regions[0].block
    if len(block.args) != 2 or any(not isinstance(arg.type, BFloat16Type) for arg in block.args):
        raise InvalidLinalgPattern("BF16 ReLU scalar arguments differ from BF16 tensor elements")
    body = tuple(block.ops)
    if len(body) != 3 or tuple(type(inner) for inner in body) != (ConstantOp, MaximumfOp, YieldOp):
        raise InvalidLinalgPattern("BF16 ReLU body must be constant, maximumf, yield")
    zero, maximum, yield_op = body
    literal = zero.properties.get("value")
    if (
        zero.attributes or set(zero.properties) != {"value"}
        or not isinstance(literal, FloatAttr) or not isinstance(literal.type, BFloat16Type)
        or literal.value.data != 0.0 or copysign(1.0, literal.value.data) != 1.0
        or tuple(zero.results) != (maximum.operands[1],)
        or maximum.operands[0] is not block.args[0]
        or maximum.attributes or set(maximum.properties) != {"fastmath"}
        or maximum.fastmath.data
        or yield_op.attributes or yield_op.properties
        or tuple(yield_op.operands) != tuple(maximum.results)
    ):
        raise InvalidLinalgPattern("BF16 ReLU does not yield exact maximumf(input, +0)")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16 ReLU Linalg operation does not verify") from exc
    return op.inputs[0], op.outputs[0]

"""Recognize the exact integer residual-add subset implemented by Gemmini LOOP_WS.

The hardware residual path is not a generic floating-point add.  With identity
load/store scales it computes, per element::

  clamp_i8(add_i32(sign_extend_i8(a), sign_extend_i8(b)))

and may replace the lower clamp by ReLU.  The recognizer accepts only that
source spelling, identity indexing maps, and row-major i8 tensors.  In
particular it does not reassociate PT2E's floating-point branch scales or move
rounding across the addition.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr


@dataclass(frozen=True)
class IntegerResAdd:
    operation: Operation
    support_operations: tuple[Operation, ...]
    lhs: SSAValue
    rhs: SSAValue
    output: SSAValue
    shape: tuple[int, ...]
    relu: bool
    a_scale: float = 1.0
    b_scale: float = 1.0
    c_scale: float = 1.0

    @property
    def numeric_contract(self) -> str:
        return "clamp_i8(add_i32(sign_extend_i8(a),sign_extend_i8(b)))"

    @property
    def operations(self) -> tuple[Operation, ...]:
        return (*self.support_operations, self.operation)


def _shape(value: SSAValue) -> tuple[int, ...] | None:
    ty = value.type
    if not isinstance(ty, TensorType):
        return None
    return tuple(int(dim) for dim in ty.get_shape())


def _width(value: SSAValue) -> int | None:
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.element_type, IntegerType):
        return None
    return int(ty.element_type.width.data)


def _identity_maps(op: Operation, rank: int) -> bool:
    attr = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    maps = getattr(attr, "data", ())
    if len(maps) != 3:
        return False
    expected = tuple(range(rank))
    for item in maps:
        amap = getattr(item, "data", None)
        if amap is None or amap.num_symbols:
            return False
        row = tuple(expr.position for expr in amap.results
                    if isinstance(expr, AffineDimExpr))
        if len(row) != len(amap.results) or row != expected:
            return False
    return True


def _literal(operation: Operation) -> int | None:
    attr = operation.properties.get("value")
    if attr is None:
        attr = operation.attributes.get("value")
    raw = getattr(getattr(attr, "value", None), "data", None)
    if raw is None:
        raw = getattr(attr, "data", None)
    return int(raw) if isinstance(raw, int) else None


def recognize(op: Operation) -> IntegerResAdd | None:
    """Return a hardware-exact residual add, otherwise leave the op on the host."""
    if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.outputs) != 1:
        return None
    if len(op.results) != 1:
        return None
    shape = _shape(op.results[0])
    if (shape is None or len(shape) < 1 or any(dim <= 0 for dim in shape)
            or any(_shape(value) != shape for value in (*op.inputs, op.results[0]))
            or any(_width(value) != 8 for value in (*op.inputs, op.results[0]))
            or not _identity_maps(op, len(shape))):
        return None
    iterators = getattr(op.properties.get("iterator_types")
                        or op.attributes.get("iterator_types"), "data", ())
    if len(iterators) != len(shape) or any("parallel" not in str(item) for item in iterators):
        return None
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    block = op.regions[0].blocks[0]
    operations = tuple(block.ops)
    names = tuple(item.name for item in operations)
    if names != ("arith.extsi", "arith.extsi", "arith.addi", "arith.constant",
                 "arith.maxsi", "arith.constant", "arith.minsi", "arith.trunci",
                 "linalg.yield"):
        return None
    if len(block.args) != 3:
        return None
    xwide, ywide, added, lower, clamped_low, upper, clamped, narrowed, yielded = operations
    if (tuple(xwide.operands) != (block.args[0],)
            or tuple(ywide.operands) != (block.args[1],)
            or tuple(added.operands) != (xwide.results[0], ywide.results[0])
            or tuple(clamped_low.operands) != (added.results[0], lower.results[0])
            or tuple(clamped.operands) != (clamped_low.results[0], upper.results[0])
            or tuple(narrowed.operands) != (clamped.results[0],)
            or tuple(yielded.operands) != (narrowed.results[0],)
            or _literal(upper) != 127 or _literal(lower) not in (-128, 0)):
        return None
    init = op.outputs[0].owner if isinstance(op.outputs[0].owner, Operation) else None
    if (init is None or init.name != "tensor.empty"
            or any(use.operation is not op for use in init.results[0].uses)):
        return None
    return IntegerResAdd(
        op, (init,), op.inputs[0], op.inputs[1], op.results[0], shape,
        relu=_literal(lower) == 0)


def looks_like_candidate(op: Operation) -> bool:
    """Identify an intended integer residual add for an auditable refusal receipt."""
    tag = getattr(op.attributes.get("prov.op"), "data", "")
    return (op.name == "linalg.generic" and "add" in str(tag).lower()
            and len(op.inputs) == 2 and len(op.results) == 1
            and any(_width(value) == 8 for value in (*op.inputs, op.results[0])))

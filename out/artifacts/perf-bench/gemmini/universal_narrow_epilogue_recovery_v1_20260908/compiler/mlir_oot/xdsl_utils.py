"""Small structural xDSL queries used by the self-contained backend.

These used to be imported from Merlin's authoring tree.  Keeping the queries
here makes the emitted package independent of the harness and reference
implementation, as required by the backend integrity contract.
"""
from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr


def _integer_tensor(value: SSAValue) -> bool:
    ty = value.type
    return isinstance(ty, TensorType) and isinstance(ty.get_element_type(), IntegerType)


def is_integer_matmul(op: Operation) -> bool:
    """Recognize a canonical integer matmul structurally.

    The integer preparation pass deliberately emits a generic op when xDSL cannot build a
    mixed-input/result named matmul with the required sign-extension body.  Treat that exact
    ``C[m,n] += A[m,k] * B[k,n]`` form as the same operation; do not use provenance annotations to
    grant accelerator eligibility.
    """
    integer_signature = (len(op.operands) == 3 and len(op.results) == 1
                         and all(_integer_tensor(value) for value in op.operands)
                         and _integer_tensor(op.results[0]))
    if not integer_signature:
        return False
    if op.name == "linalg.matmul":
        return True
    if op.name != "linalg.generic":
        return False
    maps_attr = op.properties.get("indexing_maps")
    iters_attr = op.properties.get("iterator_types")
    if maps_attr is None or iters_attr is None:
        return False
    maps = [item.data for item in maps_attr.data]
    dims = []
    for affine_map in maps:
        row = []
        for expr in affine_map.results:
            if not isinstance(expr, AffineDimExpr):
                return False
            row.append(expr.position)
        dims.append(row)
    iters = [str(item) for item in iters_attr.data]
    body_names = [item.name for item in op.regions[0].blocks[0].ops]
    return (dims == [[0, 2], [2, 1], [0, 1]]
            and len(iters) == 3
            and all("parallel" in item for item in iters[:2])
            and "reduction" in iters[2]
            and body_names == ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi",
                               "linalg.yield"])


def _trunc_div(lhs: int, rhs: int) -> int:
    if rhs == 0:
        raise ZeroDivisionError
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) != (rhs < 0) else quotient


def constant_integer(value: SSAValue) -> int | None:
    """Evaluate a side-effect-free integer SSA expression when it is constant.

    This is deliberately a query, not an interpreter: unknown values and ops
    return ``None``.  It covers the LLVM integer operations emitted by
    :class:`FnBuilder`, which is enough to prove tensor-extract indexes without
    consulting any runtime or reference implementation.
    """
    seen: set[SSAValue] = set()

    def visit(item: SSAValue) -> int | None:
        if item in seen:
            return None
        seen.add(item)
        owner = item.owner
        if not isinstance(owner, Operation):
            return None
        if owner.name == "llvm.mlir.constant":
            attr = owner.properties.get("value")
            if attr is None:
                attr = owner.attributes.get("value")
            if isinstance(attr, IntegerAttr):
                return int(attr.value.data)
            raw = getattr(getattr(attr, "value", None), "data", None)
            return None if raw is None else int(raw)
        if len(owner.operands) != 2:
            return None
        lhs, rhs = (visit(operand) for operand in owner.operands)
        if lhs is None or rhs is None:
            return None
        name = owner.name
        if name == "llvm.add":
            return lhs + rhs
        if name == "llvm.sub":
            return lhs - rhs
        if name == "llvm.mul":
            return lhs * rhs
        if name == "llvm.and":
            return lhs & rhs
        if name == "llvm.or":
            return lhs | rhs
        if name == "llvm.xor":
            return lhs ^ rhs
        if name == "llvm.sdiv":
            return _trunc_div(lhs, rhs)
        if name == "llvm.udiv":
            return (lhs & ((1 << 64) - 1)) // (rhs & ((1 << 64) - 1))
        if name == "llvm.srem":
            return lhs - _trunc_div(lhs, rhs) * rhs
        if name == "llvm.urem":
            return (lhs & ((1 << 64) - 1)) % (rhs & ((1 << 64) - 1))
        if name == "llvm.shl":
            return lhs << rhs
        if name == "llvm.ashr":
            return lhs >> rhs
        if name == "llvm.lshr":
            return (lhs & ((1 << 64) - 1)) >> rhs
        return None

    return visit(value)

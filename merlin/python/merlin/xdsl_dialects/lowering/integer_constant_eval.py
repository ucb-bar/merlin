"""Conservative integer SSA folding for statically indexed tensor lowering.

Only pure, explicitly supported integer instructions are evaluated. Loads,
block arguments and unknown instructions stay unknown. Integer widths and
signedness come from the IR, never from a target or a model's identity.
"""
from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Operation, SSAValue


def constant_integer(value: SSAValue) -> int | None:
    """Return the signed integer constant carried by ``value``, if provable."""
    cache: dict[SSAValue, int | None] = {}

    def visit(v: SSAValue) -> int | None:
        if v in cache:
            return cache[v]
        cache[v] = None
        if not isinstance(v.type, IntegerType) or not isinstance(v.owner, Operation):
            return None
        width = int(v.type.width.data)
        mask = (1 << width) - 1
        op = v.owner
        overflow = op.properties.get("overflowFlags")
        if isinstance(overflow, IntegerAttr) and int(overflow.value.data):
            # A wrapping evaluation cannot prove an nsw/nuw operation non-poison.
            return None
        attr = op.properties.get("value", op.attributes.get("value"))
        if op.name in ("llvm.mlir.constant", "arith.constant") and isinstance(attr, IntegerAttr):
            result = int(attr.value.data)
        else:
            vals = [visit(operand) for operand in op.operands]
            if not vals or any(item is None for item in vals):
                return None
            a = vals[0]
            assert a is not None
            if op.name in ("llvm.sext", "llvm.trunc", "arith.extsi", "arith.trunci"):
                result = a
            elif op.name in ("llvm.zext", "arith.extui"):
                result = a & ((1 << int(op.operands[0].type.width.data)) - 1)
            elif len(vals) == 2:
                b = vals[1]
                assert b is not None
                if op.name in ("llvm.add", "arith.addi"):
                    result = a + b
                elif op.name in ("llvm.sub", "arith.subi"):
                    result = a - b
                elif op.name in ("llvm.mul", "arith.muli"):
                    result = a * b
                elif op.name in ("llvm.and", "arith.andi"):
                    result = a & b
                elif op.name in ("llvm.or", "arith.ori"):
                    result = a | b
                elif op.name in ("llvm.xor", "arith.xori"):
                    result = a ^ b
                elif op.name in ("llvm.shl", "llvm.ashr", "llvm.lshr") and 0 <= b < width:
                    result = a << b if op.name == "llvm.shl" else (
                        a >> b if op.name == "llvm.ashr" else (a & mask) >> b)
                else:
                    return None
            else:
                return None
        result &= mask
        if result >= 1 << (width - 1):
            result -= 1 << width
        cache[v] = result
        return result

    return visit(value)

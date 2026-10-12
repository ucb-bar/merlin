"""Bounded exact control trace when every branch depends only on constants.

This fallback observes no tensor values: loads, calls and opaque results remain
unknown. A branch consuming any unknown refuses the whole count. LLVM widths,
signed comparisons, casts and integer wrap are preserved; possible poison refuses.
"""

from collections import Counter

from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.llvm import ICmpPredicateFlag

from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

MAX_BLOCK_VISITS = 100_000
MAX_OPERATIONS = 1_000_000


def _width(value):
    ty = value.type
    if not isinstance(ty, IntegerType) or not 1 <= ty.width.data <= 128:
        raise ValueError("constant control requires a bounded integer width")
    return ty.width.data


def _signed(value, width):
    value %= 1 << width
    return value - (1 << width) if value >= 1 << (width - 1) else value


def integer_operation(op, args):
    """One exact integer instruction; unknown operands never become constants."""
    if op.name == "llvm.mlir.constant":
        return constant_integer(op.results[0])
    if op.name == "llvm.select":
        if args[0] is None:
            return None
        return args[1] if args[0] else args[2]
    if not args or any(type(v) is not int for v in args):
        return None
    width = _width(op.operands[0])
    unsigned = [v % (1 << width) for v in args]
    signed = [_signed(v, width) for v in args]
    name = op.name
    if name == "llvm.icmp":
        predicate = ICmpPredicateFlag.from_int(op.properties["predicate"].value.data).value
        a, b = signed if predicate.startswith("s") else unsigned
        comparison = {
            "eq": a == b,
            "ne": a != b,
            "slt": a < b,
            "ult": a < b,
            "sle": a <= b,
            "ule": a <= b,
            "sgt": a > b,
            "ugt": a > b,
            "sge": a >= b,
            "uge": a >= b,
        }
        return int(comparison[predicate])
    if name in {"llvm.sext", "llvm.zext", "llvm.trunc"}:
        result = signed[0] if name == "llvm.sext" else unsigned[0]
    else:
        a = unsigned[0]
        b = unsigned[1] if len(unsigned) > 1 else None
        if name == "llvm.add":
            result = a + b
        elif name == "llvm.sub":
            result = a - b
        elif name == "llvm.mul":
            result = a * b
        elif name == "llvm.and":
            result = a & b
        elif name == "llvm.or":
            result = a | b
        elif name == "llvm.xor":
            result = a ^ b
        elif name in {"llvm.udiv", "llvm.urem", "llvm.sdiv", "llvm.srem"}:
            if not b:
                raise ValueError("constant control integer division by zero")
            if name.startswith("llvm.s"):
                a, b = signed
                if a == -(1 << (width - 1)) and b == -1:
                    raise ValueError("constant control signed division overflow")
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                result = quotient if name == "llvm.sdiv" else a - quotient * b
            else:
                result = a // b if name == "llvm.udiv" else a % b
        elif name in {"llvm.shl", "llvm.lshr", "llvm.ashr"}:
            if b >= width:
                raise ValueError("constant control shift is poison")
            result = a << b if name == "llvm.shl" else a >> b if name == "llvm.lshr" else signed[0] >> b
        else:
            return None
        # No-wrap/exact assertions need a separate poison proof, never ignored.
        for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
            flags = op.properties.get(key)
            flag_value = getattr(getattr(flags, "value", None), "data", getattr(flags, "data", None))
            if flags is not None and flag_value not in ((), 0, False):
                raise ValueError("constant control arithmetic has unproved poison flags")
    return _signed(result, _width(op.results[0]))


def constant_control_counts(function, *, max_blocks=MAX_BLOCK_VISITS, max_operations=MAX_OPERATIONS):
    """Prove execution counts by terminating bounded constant-only CFG traversal."""
    blocks = tuple(function.body.blocks)
    if (
        not blocks
        or type(max_blocks) is not int
        or type(max_operations) is not int
        or min(max_blocks, max_operations) <= 0
    ):
        raise ValueError("constant control has no declared finite proof budget")
    if sum(len(tuple(b.ops)) for b in blocks) > max_operations:
        raise ValueError("constant control static operation budget exhausted")
    values = {arg: None for arg in blocks[0].args}
    counts = Counter()
    visits = operations = 0
    block, incoming = blocks[0], ()
    while block is not None:
        visits += 1
        if visits > max_blocks or block not in blocks:
            raise ValueError("constant control block budget exhausted or branch leaves function")
        counts[block] += 1
        if len(incoming) != len(block.args) and block is not blocks[0]:
            raise ValueError("constant control block argument mismatch")
        if incoming:
            values.update(zip(block.args, incoming, strict=True))
        successor = None
        for op in block.ops:
            operations += 1
            if operations > max_operations:
                raise ValueError("constant control operation budget exhausted")
            args = [values.get(v) for v in op.operands]
            if op.name == "llvm.br":
                successor, incoming = op.successors[0], tuple(args)
            elif op.name == "llvm.cond_br":
                if type(args[0]) is not int:
                    raise ValueError("constant control branch depends on tensor data or opaque result")
                sizes = list(op.properties["operandSegmentSizes"].get_values())
                taken = args[0] != 0
                successor = op.successors[0 if taken else 1]
                incoming = tuple(args[1 : 1 + sizes[1]] if taken else args[1 + sizes[1] :])
            elif op.name == "llvm.return":
                pass
            elif len(op.results) == 1:
                values[op.results[0]] = integer_operation(op, args)
            else:
                values.update((result, None) for result in op.results)
        if block.last_op.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            raise ValueError("constant control has unsupported terminator")
        block = successor
    return {b: counts[b] for b in blocks}

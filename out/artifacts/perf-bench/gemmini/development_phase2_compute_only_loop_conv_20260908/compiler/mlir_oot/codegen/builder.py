"""A small builder over the xDSL LLVM dialect for the compiler-generated CPU lane.

Two deliberate constraints, both established by measuring what the target's program oracle
actually executes (see docs/iteration_notes.md):

* the emitted `llvm.func` is **single-block** — the kernel is fully unrolled straight-line code;
* the CPU lane is **branch-free and select-free** — every min/max/clamp/predicate is built out of
  sign-mask integer arithmetic, so nothing depends on `llvm.icmp`/`llvm.select`.

Everything is constructed as real IR; no part of the artifact is assembled from strings.
"""
from __future__ import annotations

from typing import Callable

from xdsl.dialects import llvm
from xdsl.dialects.builtin import Attribute, IntegerAttr, IntegerType, i8, i16, i32, i64
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, result_def

PTR = llvm.LLVMPointerType()

INT_TYPES = {"i8": i8, "i16": i16, "i32": i32, "i64": i64}


@irdl_op_definition
class TypedConstantOp(IRDLOperation):
    """`llvm.mlir.constant` that always prints its value WITH its type.

    xDSL elides the type for an i64 constant (`llvm.mlir.constant(5) : i64`).  The emitted
    artifact is read back by a decoder that expects the canonical `llvm.mlir.constant(5 : i64)`
    spelling, so this op keeps the type in the literal.
    """

    name = "llvm.mlir.constant"
    result = result_def(Attribute)
    value = prop_def(IntegerAttr)

    def print(self, printer) -> None:
        printer.print_string("(")
        printer.print_attribute(self.value)
        printer.print_string(") : ")
        printer.print_attribute(self.result.type)


def iconst(value: int, ty: IntegerType = i64) -> TypedConstantOp:
    return TypedConstantOp(properties={"value": IntegerAttr(int(value), ty)}, result_types=[ty])


class FnBuilder:
    """Builds the body of one single-block `llvm.func`."""

    def __init__(self, arg_types: list[Attribute]):
        self.region = Region([])
        self.entry = Block(arg_types=list(arg_types))
        self.region.add_block(self.entry)
        self.blk = self.entry
        self._consts: dict[tuple[int, str], SSAValue] = {}

    # -- primitives --------------------------------------------------------------------------
    def add(self, op):
        self.blk.add_op(op)
        return op

    def const(self, value: int, ty: IntegerType = i64) -> SSAValue:
        key = (int(value), str(ty))
        hit = self._consts.get(key)
        if hit is not None:
            return hit
        op = iconst(value, ty)
        self.prologue(op)
        self._consts[key] = op.results[0]
        return op.results[0]

    def for_range(self, count: int, body: Callable, initial: SSAValue | None = None):
        """Emit the source loop as CFG, optionally carrying one exact scalar value."""
        if count <= 0:
            return initial
        if initial is None:
            header, loop, after = Block(arg_types=[i64]), Block(), Block()
        else:
            header = Block(arg_types=[i64, initial.type])
            loop = Block(arg_types=[initial.type])
            after = Block(arg_types=[initial.type])
        self.region.add_block(header)
        self.region.add_block(loop)
        self.region.add_block(after)
        entry_args = [self.const(0)] + ([] if initial is None else [initial])
        self.add(llvm.BrOp(header, *entry_args))
        self.blk = header
        iv = header.args[0]
        condition = self.add(llvm.ICmpOp(iv, self.const(count), IntegerAttr(2, i64))).results[0]
        carried = [] if initial is None else [header.args[1]]
        self.add(llvm.CondBrOp(condition, loop, carried, after, carried))
        self.blk = loop
        result = body(iv) if initial is None else body(iv, loop.args[0])
        next_args = [self.add_i(iv, self.const(1))]
        if initial is not None:
            if not isinstance(result, SSAValue) or result.type != initial.type:
                raise TypeError("loop-carried body must return one value of the initial type")
            next_args.append(result)
        self.add(llvm.BrOp(header, *next_args))
        self.blk = after
        return None if initial is None else after.args[0]

    def prologue(self, op):
        """Entry definitions dominate every loop; insert before its branch."""
        last = self.entry.last_op
        if last is not None and last.name in ("llvm.br", "llvm.cond_br", "llvm.return"):
            self.entry.insert_op_before(op, last)
        else:
            self.entry.add_op(op)
        return op

    # -- memory ------------------------------------------------------------------------------
    def gep(self, base: SSAValue, index: SSAValue, elem_ty: Attribute) -> SSAValue:
        op = self.add(llvm.GEPOp(base, [llvm.GEP_USE_SSA_VAL], elem_ty, ssa_indices=[index]))
        return op.results[0]

    def load_i64(self, base: SSAValue, index: SSAValue, dtype: str) -> SSAValue:
        """Load one element and sign-extend it to i64 (the CPU lane's working width)."""
        ty = INT_TYPES[dtype]
        val = self.add(llvm.LoadOp(self.gep(base, index, ty), ty)).results[0]
        if ty is i64:
            return val
        return self.add(llvm.SExtOp(val, i64)).results[0]

    def store_i64(self, value: SSAValue, base: SSAValue, index: SSAValue, dtype: str) -> None:
        ty = INT_TYPES[dtype]
        out = value if ty is i64 else self.add(llvm.TruncOp(value, ty)).results[0]
        self.add(llvm.StoreOp(out, self.gep(base, index, ty)))

    # -- select-free integer helpers ---------------------------------------------------------
    def add_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.AddOp(a, b)).results[0]

    def sub_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.SubOp(a, b)).results[0]

    def mul_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.MulOp(a, b)).results[0]

    def and_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.AndOp(a, b)).results[0]

    def or_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.OrOp(a, b)).results[0]

    def ashr_i(self, a: SSAValue, bits: int) -> SSAValue:
        return self.add(llvm.AShrOp(a, self.const(bits))).results[0]

    def shl_i(self, a: SSAValue, bits: int) -> SSAValue:
        return self.add(llvm.ShlOp(a, self.const(bits))).results[0]

    def xor_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.XOrOp(a, b)).results[0]

    def sdiv_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.SDivOp(a, b)).results[0]

    def udiv_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.UDivOp(a, b)).results[0]

    def srem_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.SRemOp(a, b)).results[0]

    def urem_i(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.add(llvm.URemOp(a, b)).results[0]

    # Variable-amount shifts.  The constant-amount helpers above cover the fixed field
    # extractions; an f32 <-> integer conversion shifts by a value it computed, so the amount
    # has to be an SSA operand.
    def shl_v(self, a: SSAValue, n: SSAValue) -> SSAValue:
        return self.add(llvm.ShlOp(a, n)).results[0]

    def lshr_v(self, a: SSAValue, n: SSAValue) -> SSAValue:
        return self.add(llvm.LShrOp(a, n)).results[0]

    def ashr_v(self, a: SSAValue, n: SSAValue) -> SSAValue:
        return self.add(llvm.AShrOp(a, n)).results[0]

    def wrap_int(self, value: SSAValue, bits: int) -> SSAValue:
        """Reduce an i64 to the value an integer of `bits` width holds (two's-complement wrap).

        linalg integer arithmetic is modular in the declared width, so a narrower element type
        has to wrap here rather than at the store: an intermediate that never reaches memory
        would otherwise keep a range its declared type cannot hold.
        """
        if bits >= 64:
            return value
        return self.ashr_i(self.shl_i(value, 64 - bits), 64 - bits)

    def notmask(self, m: SSAValue) -> SSAValue:
        return self.add(llvm.XOrOp(m, self.const(-1))).results[0]

    def mask_nonneg(self, v: SSAValue) -> SSAValue:
        """All-ones when `v >= 0`, zero otherwise (i64 sign mask, complemented)."""
        return self.notmask(self.ashr_i(v, 63))

    def mask_neg(self, v: SSAValue) -> SSAValue:
        """All-ones when `v < 0`, zero otherwise."""
        return self.ashr_i(v, 63)

    def max0(self, v: SSAValue) -> SSAValue:
        """`max(v, 0)` without a select."""
        return self.and_i(v, self.mask_nonneg(v))

    def smax(self, a: SSAValue, b: SSAValue) -> SSAValue:
        """`max(a, b) = b + max(a - b, 0)` — exact in i64 for accumulator-width operands."""
        return self.add_i(b, self.max0(self.sub_i(a, b)))

    def smin(self, a: SSAValue, b: SSAValue) -> SSAValue:
        """`min(a, b) = a - max(a - b, 0)`."""
        return self.sub_i(a, self.max0(self.sub_i(a, b)))

    def blend(self, mask: SSAValue, when_set: SSAValue, when_clear: SSAValue) -> SSAValue:
        """`mask ? when_set : when_clear` for an all-ones / all-zero mask."""
        return self.or_i(self.and_i(when_set, mask),
                         self.and_i(when_clear, self.notmask(mask)))

    def clamp(self, value: SSAValue, lo: int, hi: int) -> SSAValue:
        return self.smin(self.smax(value, self.const(lo)), self.const(hi))

    def finish(self) -> Region:
        self.blk.add_op(llvm.ReturnOp())
        return self.region

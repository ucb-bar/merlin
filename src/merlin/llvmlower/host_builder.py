"""A small builder over the xDSL LLVM dialect for the compiler-generated CPU lane.

The straight-line CPU lowering uses two explicit representation constraints:

* the emitted `llvm.func` is **single-block** — the kernel is fully unrolled straight-line code;
* the CPU lane is **branch-free and select-free** — every min/max/clamp/predicate is built out of
  sign-mask integer arithmetic, so nothing depends on `llvm.icmp`/`llvm.select`.

The golden kernel path also uses this builder's `for_loop` to emit ordinary CPU CFG loops.
Everything is constructed as real IR; no part of the artifact is assembled from strings.
"""

from __future__ import annotations

from collections.abc import Callable

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
    """Builds one `llvm.func`, with optional CPU loops for the golden path."""

    def __init__(self, arg_types: list[Attribute]):
        self.region = Region([])
        self.entry = Block(arg_types=list(arg_types))
        self.region.add_block(self.entry)
        self.blk = self.entry
        self._consts: dict[tuple[int, str], SSAValue] = {}

    @classmethod
    def for_logical_kernel(cls, command_buffer, *, abi=None):
        """Build pointer arguments in the selected logical ABI, addressed by source name.

        Semantic roles such as lhs/rhs do not determine pointer position. Use the
        returned name map when lowering operands; its insertion order is the
        actual function argument order. The ordinary host shim uses the same
        resolver. An explicit whole-program ABI remains the original caller's
        selection, never an inference from this builder.
        """
        from merlin.compile.logical_abi import kernel_arg_order

        names = kernel_arg_order(command_buffer, abi)
        builder = cls([PTR] * len(names))
        return builder, dict(zip(names, builder.entry.args, strict=True))

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
        # Keep constants in the entry block so they dominate CPU loops.  A loop may already
        # have installed the entry terminator when its body asks for a new constant.
        terminator = self.entry.last_op
        if terminator is not None and terminator.has_trait(llvm.IsTerminator):
            self.entry.insert_op_before(op, terminator)
        else:
            self.entry.add_op(op)
        self._consts[key] = op.results[0]
        return op.results[0]

    def for_range(self, count: int, body: Callable[[SSAValue], None]) -> None:
        """`for iv in [0, count)`, fully unrolled — the kernel stays single-block."""
        for i in range(count):
            body(self.const(i))

    def for_loop(
        self, start: int, stop: int, step: int, body: Callable[[SSAValue], None], *, retain_loop: bool = False
    ) -> None:
        """Emit an ordinary CPU loop in LLVM CFG; using scalar CPU control flow."""
        if type(retain_loop) is not bool:
            raise ValueError("retaining an ordinary CPU loop requires explicit boolean")
        if step <= 0 or start < 0 or stop < start:
            raise ValueError("invalid CPU loop bounds")
        if start == stop:
            return
        header = Block(arg_types=[i64])
        loop_body = Block()
        exit_block = Block()
        self.region.add_block(header)
        self.region.add_block(loop_body)
        self.region.add_block(exit_block)
        self.add(llvm.BrOp(header, self.const(start)))
        self.blk = header
        iv = header.args[0]
        cond = self.add(
            llvm.ICmpOp(iv, self.const(stop), IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
        ).results[0]
        self.add(llvm.CondBrOp(cond, loop_body, [], exit_block, []))
        self.blk = loop_body
        body(iv)
        nxt = self.add_i(iv, self.const(step))
        latch = self.add(llvm.BrOp(header, nxt))
        if retain_loop:
            from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll

            disable_loop_unroll(latch)
        self.blk = exit_block

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
        return self.or_i(self.and_i(when_set, mask), self.and_i(when_clear, self.notmask(mask)))

    def clamp(self, value: SSAValue, lo: int, hi: int) -> SSAValue:
        return self.smin(self.smax(value, self.const(lo)), self.const(hi))

    def finish(self) -> Region:
        self.blk.add_op(llvm.ReturnOp())
        return self.region


def append_void_region(builder: FnBuilder, function: llvm.FuncOp, arguments) -> dict:
    """Append a closed void CFG without calls, reordering or removing any effects.

    This is mechanical region composition, not an alias or async-completion
    proof. The caller must establish actual argument ownership and ordering.
    All exits join a fresh continuation; internal branches and fences survive.
    The original function is never changed. External calls/globals, nested
    regions and captured SSA values refuse rather than gaining new authority.
    """
    arguments = tuple(arguments)
    function.verify()
    if any(not key.startswith("prov.") for key in function.attributes) or set(function.properties) - {"sym_name", "function_type", "CConv", "linkage", "visibility_"}:
        raise ValueError("region has unsupported function-level semantics")
    blocks = tuple(function.body.blocks)
    if not blocks or len(blocks[0].args) != len(arguments):
        raise ValueError("closed region argument arity differs")
    if any(a.type != b.type for a, b in zip(blocks[0].args, arguments, strict=True)):
        raise ValueError("closed region argument types differ")
    if builder.blk.last_op is not None and builder.blk.last_op.has_trait(llvm.IsTerminator):
        raise ValueError("region append requires an open continuation")
    operations = tuple(op for block in blocks for op in block.ops)
    defined = {arg for block in blocks for arg in block.args} | {r for op in operations for r in op.results}
    returns = [op for op in operations if isinstance(op, llvm.ReturnOp)]
    if not returns or any(op.operands for op in returns):
        raise ValueError("region composition requires explicit void returns")
    if any(op.regions or isinstance(op, (llvm.CallOp, llvm.AddressOfOp)) for op in operations):
        raise ValueError("region composition refuses nested regions, calls and global references")
    if any(value not in defined for op in operations for value in op.operands):
        raise ValueError("region contains captured SSA values")
    if any(successor is blocks[0] for op in operations for successor in op.successors):
        raise ValueError("region cannot branch to its argument entry")
    if any(successor not in blocks for op in operations for successor in op.successors):
        raise ValueError("region branches outside its source function")
    if any(not isinstance(block.last_op, (llvm.BrOp, llvm.CondBrOp, llvm.ReturnOp)) for block in blocks):
        raise ValueError("region has an unsupported exit")
    clone = function.clone()
    copied = tuple(clone.body.blocks)
    first = copied[0]
    for old, new in zip(tuple(first.args), arguments, strict=True):
        old.replace_all_uses_with(new)
    for old in reversed(tuple(first.args)):
        first.erase_arg(old)
    continuation = Block()
    builder.add(llvm.BrOp(first))
    for block in copied:
        clone.body.detach_block(block)
        builder.region.add_block(block)
        last = block.last_op
        if isinstance(last, llvm.ReturnOp):
            block.erase_op(last)
            block.add_op(llvm.BrOp(continuation))
    builder.region.add_block(continuation)
    builder.blk = continuation
    return {"blocks": len(blocks), "returns": len(returns), "operations": len(operations),
            "effects": "all original operations retained in source order", "calls_added": 0}

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
from dataclasses import dataclass
from hashlib import sha256
from math import prod

from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    Attribute,
    Float32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    i8,
    i16,
    i32,
    i64,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, result_def

from merlin.compiler.dialects.xdsl_context import text
from merlin.compiler.lower.host.writers import PrivateWorkspaceContract, validate_private_workspaces
from merlin.compiler.schedule.panels.loaded_sums import DenseLoadedSumComposition

# ---- host_builder ------------------------------------------------------------------------------

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
        from merlin.compiler.ir.logical_abi import kernel_arg_order

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
    if any(not key.startswith("prov.") for key in function.attributes) or set(function.properties) - {
        "sym_name",
        "function_type",
        "CConv",
        "linkage",
        "visibility_",
    }:
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
    return {
        "blocks": len(blocks),
        "returns": len(returns),
        "operations": len(operations),
        "effects": "all original operations retained in source order",
        "calls_added": 0,
    }


# ---- llvm_loop_metadata ------------------------------------------------------------------------
# Explicit upstream LLVM loop choices for xDSL-produced CPU control flow.


def disable_loop_unroll(latch):
    """Retain a caller-selected ordinary CPU loop through upstream LLVM.

    The caller identifies the actual latch. No arithmetic, dependence, alias or
    numerical permission is inferred. Existing loop metadata refuses to avoid
    discarding another pass's choices. Upstream translation validates the LLVM
    attribute, which this xDSL version retains as an unregistered attribute.
    """
    if not isinstance(latch, (llvm.BrOp, llvm.CondBrOp)):
        raise ValueError("loop metadata requires an LLVM latch branch")
    if "loop_annotation" in latch.attributes or "loop_annotation" in latch.properties:
        raise ValueError("existing loop metadata requires explicit composition")
    annotation = Context(allow_unregistered=True).get_attr("llvm.loop_annotation")
    latch.attributes["loop_annotation"] = annotation("llvm.loop_annotation", False, False, "unroll = <disable = true>")


# ---- generated_private_workspace ---------------------------------------------------------------
# Bind one generated private byte allocation to a checked staged Q1 source.
#
# An independently regenerated complete module fixes every instruction and edge.
# This reader proves only the allocation and its pointer-use closure. Original
# primitive DMA bounds/noescape/completion and stock compiler custody remain
# separate caller obligations; no host store or ordinary call is admitted.


def _digest(module):
    return sha256(text(module, generic=True).encode()).hexdigest()


def _normalized(module):
    # Ordinary parsing may move inherent loop annotations from attributes to
    # properties. Normalize both actual and independently regenerated modules.
    from xdsl.parser import Parser

    from merlin.compiler.ir.llvm_context import make_llvm_context

    return text(Parser(make_llvm_context(), text(module)).parse_module(), generic=True)


@dataclass(frozen=True)
class PrivateWorkspaceOperand:
    """One actual original target operand, qualified by its target owner.

    Supplying this value does not establish the primitive's memory semantics.
    The reader checks current operation membership and the complete use closure.
    """

    operation: object
    operand: int
    direction: str
    qualification_identity: str


@dataclass(frozen=True)
class GeneratedPrivateWorkspace:
    module: ModuleOp
    reference: ModuleOp
    composition: DenseLoadedSumComposition
    workspace: PrivateWorkspaceContract
    entry_symbol: str
    target_operands: tuple[PrivateWorkspaceOperand, ...]
    source_sha256: str
    reference_sha256: str
    matched_uses: int

    def revalidate(self):
        if _digest(self.module) != self.source_sha256 or _digest(self.reference) != self.reference_sha256:
            raise ValueError("generated quantized workspace source changed")
        current = recognize_quantized_workspace(
            self.module,
            self.reference,
            self.composition,
            self.workspace,
            entry_symbol=self.entry_symbol,
            target_operands=self.target_operands,
        )
        if current.record() != self.record():
            raise ValueError("generated quantized workspace binding changed")

    def record(self):
        return dict(
            schema="merlin.generated_quantized_workspace.v1",
            source_sha256=self.source_sha256,
            reference_sha256=self.reference_sha256,
            composition_identity=self.composition.identity,
            original_source_identity=self.composition.source.identity,
            entry_symbol=self.entry_symbol,
            private_bytes=self.workspace.bytes,
            alignment=self.workspace.alignment,
            target_pointer_uses=self.matched_uses,
            bound="complete ordinary regenerated instructions and CFG; one invocation-private byte allocation",
            source_lifetime="original Q1 cover, real handoff drain and original loaded-sum phase drains",
            scope="allocation/use closure only; original target effects, pointer bounds, compiler preservation and final object remain independently owned",
        )


def recognize_quantized_workspace(module, reference, composition, workspace, *, entry_symbol, target_operands):
    """Recognize a regenerated staged-Q1 image without permitting host writes.

    ``reference`` is the original owner's independent normal source emission,
    never a saved verdict or executable substitute. Its exact equality is
    necessary in addition to the current source/consumer and allocation checks.
    Every opaque target pointer operand still requires original qualification.
    """
    if (
        type(module) is not ModuleOp
        or type(reference) is not ModuleOp
        or type(composition) is not DenseLoadedSumComposition
        or type(workspace) is not PrivateWorkspaceContract
    ):
        raise ValueError("current original modules, checked staged source and private workspace required")
    composition.revalidate()
    validate_private_workspaces((workspace,))
    if (
        workspace.bytes != prod(composition.kernel.tensor("quantized_workspace").shape)
        or workspace.effect_witness_sha256 != composition.identity
    ):
        raise ValueError("private workspace differs from the complete original Q1 source lifetime")
    module.verify()
    reference.verify()
    if _normalized(module) != _normalized(reference):
        raise ValueError("actual quantized workspace image differs from the independent ordinary emission")
    functions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == entry_symbol]
    if len(functions) != 1 or not functions[0].body.blocks:
        raise ValueError("one current original quantized entry required")
    function = functions[0]
    entry = function.body.blocks.first
    if function.function_type.output != llvm.LLVMVoidType() or any(a.type != PTR for a in entry.args):
        raise ValueError("private workspace requires the original pointer-only void entry")
    operations = tuple(function.walk())
    allocations = [op for op in operations if isinstance(op, llvm.AllocaOp)]
    if len(allocations) != 1:
        raise ValueError("exactly one generated Q1 allocation required")
    allocation = allocations[0]
    owner = allocation.size.owner
    if (
        allocation.parent is not entry
        or owner.name != "llvm.mlir.constant"
        or owner.value.value.data != workspace.bytes
        or allocation.elem_type != i8
        or allocation.alignment.value.data < workspace.alignment
        or allocation.alignment.value.data % workspace.alignment
    ):
        raise ValueError("generated Q1 allocation size, position, byte type or alignment changed")
    for op in operations:
        if op.name in {
            "llvm.store",
            "llvm.call",
            "llvm.call_intrinsic",
            "llvm.invoke",
            "llvm.callbr",
        } or op.name.startswith("llvm.intr."):
            raise ValueError("quantized workspace admits no host store or ordinary call")
    if type(target_operands) is not tuple or not target_operands:
        raise ValueError("complete original primitive pointer-operand roster required")
    declared = {}
    for use in target_operands:
        if (
            type(use) is not PrivateWorkspaceOperand
            or use.operation not in operations
            or not isinstance(use.operation, llvm.InlineAsmOp)
            or use.operation.results
            or type(use.operand) is not int
            or not 0 <= use.operand < len(use.operation.operands)
            or use.direction not in ("read", "write")
            or not isinstance(use.qualification_identity, str)
            or not use.qualification_identity.strip()
            or (use.operation, use.operand) in declared
        ):
            raise ValueError("private workspace needs exact current qualified primitive operands")
        declared[use.operation, use.operand] = use.direction
    seen, pending, actual = set(), [allocation.res], set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        for use in value.uses:
            op, index = use.operation, use.index
            if isinstance(op, llvm.GEPOp) and index == 0 and value.type == PTR and op.elem_type == i8:
                pending.extend(op.results)
            elif isinstance(op, llvm.PtrToIntOp) and index == 0 and value.type == PTR and op.output.type == i64:
                pending.extend(op.results)
            elif (op, index) in declared:
                actual.add((op, index))
            else:
                raise ValueError("generated private Q1 pointer has an unaccounted escape")
    if actual != set(declared) or set(declared.values()) != {"read", "write"}:
        raise ValueError("private Q1 pointer roster is incomplete or names another allocation")
    composition.revalidate()
    return GeneratedPrivateWorkspace(
        module,
        reference,
        composition,
        workspace,
        entry_symbol,
        target_operands,
        _digest(module),
        _digest(reference),
        len(actual),
    )


# ---- host_scalar_builder -----------------------------------------------------------------------
# Branch-free f32 primitives for the compiler-generated CPU lane.
#
# The straight-line lowering uses a single basic block with no `select` or branch,
# so every predicate here is built from the IEEE-754 bit
# pattern: a comparison becomes an arithmetic difference whose SIGN BIT, broadcast by an
# arithmetic shift, is the blend mask.  The transcendentals are likewise *generated* — a
# range-reduced polynomial written out as ordinary `llvm.f*` ops — rather than a call into a
# math library. These approximations require the caller's original numerical policy;
# this module does not select or authorize them automatically.
#
# Nothing in this module is keyed on a capsule: it is the scalar instruction set the linalg
# shared tensor lowering targets. No accelerator instructions are emitted here.

F32 = Float32Type()

#: exp() range reduction.  `MAGIC` is 1.5 * 2**23: adding it to a float smaller than 2**22
#: forces the fractional bits out through round-to-nearest-even, so the mantissa's low bits
#: hold the rounded integer.  That is how the exponent is extracted without an fp->int op.
_LOG2E = 1.4426950408889634
_MAGIC = 12582912.0
_LN2_HI = 0.693359375
_LN2_LO = -2.1219444005469057e-4
#: exp(r) on the reduced interval |r| <= ln2/2: the Taylor series to r**5 (< 1 ulp there).
_EXP_POLY = (1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0)
#: Abramowitz & Stegun 7.1.26 -- erf to 1.5e-7 absolute, in terms of exp() and one divide.
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
#: The classic inverse-square-root seed; three Newton steps take it below 1e-7 relative.
_RSQRT_SEED = 0x5F3759DF


class FpBuilder:
    """f32 scalar arithmetic emitted into one `FnBuilder`'s single block."""

    def __init__(self, fb):
        self.fb = fb
        self._fconsts: dict[float, SSAValue] = {}

    # -- constants ---------------------------------------------------------------------------
    def fconst(self, value: float) -> SSAValue:
        key = float(value)
        hit = self._fconsts.get(key)
        if hit is not None:
            return hit
        op = llvm.ConstantOp(FloatAttr(key, F32), F32)
        self.fb.entry.add_op(op)
        self._fconsts[key] = op.results[0]
        return op.results[0]

    def i32c(self, value: int) -> SSAValue:
        return self.fb.const(int(value), i32)

    # -- arithmetic --------------------------------------------------------------------------
    def fadd(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FAddOp(a, b)).results[0]

    def fsub(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FSubOp(a, b)).results[0]

    def fmul(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FMulOp(a, b)).results[0]

    def fdiv(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FDivOp(a, b)).results[0]

    def fneg(self, a: SSAValue) -> SSAValue:
        return self.fsub(self.fconst(0.0), a)

    # -- bit-level ---------------------------------------------------------------------------
    def bits(self, a: SSAValue) -> SSAValue:
        return self.fb.add(llvm.BitcastOp(a, i32)).results[0]

    def unbits(self, a: SSAValue) -> SSAValue:
        return self.fb.add(llvm.BitcastOp(a, F32)).results[0]

    def and32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.AndOp(a, b)).results[0]

    def or32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.OrOp(a, b)).results[0]

    def xor32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.XOrOp(a, b)).results[0]

    def add32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.AddOp(a, b)).results[0]

    def sub32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.SubOp(a, b)).results[0]

    def shl32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.ShlOp(a, self.i32c(n))).results[0]

    def ashr32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.AShrOp(a, self.i32c(n))).results[0]

    def lshr32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.LShrOp(a, self.i32c(n))).results[0]

    def not32(self, a: SSAValue) -> SSAValue:
        return self.xor32(a, self.i32c(-1))

    def blend32(self, mask: SSAValue, when_set: SSAValue, when_clear: SSAValue) -> SSAValue:
        return self.or32(self.and32(when_set, mask), self.and32(when_clear, self.not32(mask)))

    def smax32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        d = self.sub32(a, b)
        return self.add32(b, self.and32(d, self.not32(self.ashr32(d, 31))))

    def smin32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        d = self.sub32(a, b)
        return self.sub32(a, self.and32(d, self.not32(self.ashr32(d, 31))))

    # -- select-free float predicates --------------------------------------------------------
    def fmax(self, a: SSAValue, b: SSAValue) -> SSAValue:
        """`max(a, b)` for finite (or -inf) operands, with no compare and no select.

        `a - b` is negative exactly when `a < b`, so its sign bit -- broadcast by an
        arithmetic shift -- selects between the two operands' bit patterns.
        """
        mask = self.ashr32(self.bits(self.fsub(a, b)), 31)
        return self.unbits(self.blend32(mask, self.bits(b), self.bits(a)))

    def fmin(self, a: SSAValue, b: SSAValue) -> SSAValue:
        mask = self.ashr32(self.bits(self.fsub(a, b)), 31)
        return self.unbits(self.blend32(mask, self.bits(a), self.bits(b)))

    def fabs(self, a: SSAValue) -> SSAValue:
        return self.unbits(self.and32(self.bits(a), self.i32c(0x7FFFFFFF)))

    def copysign(self, magnitude: SSAValue, sign_of: SSAValue) -> SSAValue:
        sign = self.and32(self.bits(sign_of), self.i32c(-0x80000000))
        return self.unbits(self.or32(self.and32(self.bits(magnitude), self.i32c(0x7FFFFFFF)), sign))

    def fclamp(self, a: SSAValue, lo: float, hi: float) -> SSAValue:
        return self.fmin(self.fmax(a, self.fconst(lo)), self.fconst(hi))

    # -- transcendentals ---------------------------------------------------------------------
    def exp(self, x: SSAValue) -> SSAValue:
        """`exp(x)` as a generated 2**k * poly(r) sequence (no libm call)."""
        x = self.fclamp(x, -87.0, 88.0)
        y = self.fadd(self.fmul(x, self.fconst(_LOG2E)), self.fconst(_MAGIC))
        kbits = self.sub32(self.and32(self.bits(y), self.i32c(0x7FFFFF)), self.i32c(0x400000))
        kbits = self.smin32(self.smax32(kbits, self.i32c(-126)), self.i32c(127))
        kf = self.fsub(y, self.fconst(_MAGIC))
        r = self.fsub(self.fsub(x, self.fmul(kf, self.fconst(_LN2_HI))), self.fmul(kf, self.fconst(_LN2_LO)))
        acc = self.fconst(_EXP_POLY[-1])
        for c in reversed(_EXP_POLY[:-1]):
            acc = self.fadd(self.fmul(acc, r), self.fconst(c))
        scale = self.unbits(self.shl32(self.add32(kbits, self.i32c(127)), 23))
        return self.fmul(acc, scale)

    def erf(self, x: SSAValue) -> SSAValue:
        ax = self.fabs(x)
        t = self.fdiv(self.fconst(1.0), self.fadd(self.fconst(1.0), self.fmul(self.fconst(_ERF_P), ax)))
        acc = self.fconst(_ERF_A[-1])
        for c in reversed(_ERF_A[:-1]):
            acc = self.fadd(self.fmul(acc, t), self.fconst(c))
        poly = self.fmul(t, acc)
        e = self.exp(self.fneg(self.fmul(ax, ax)))
        y = self.fsub(self.fconst(1.0), self.fmul(poly, e))
        return self.copysign(y, x)

    def rsqrt(self, x: SSAValue) -> SSAValue:
        """`1/sqrt(x)` by the reciprocal-sqrt seed plus three Newton steps."""
        half = self.fmul(self.fconst(0.5), x)
        y = self.unbits(self.sub32(self.i32c(_RSQRT_SEED), self.ashr32(self.bits(x), 1)))
        for _ in range(3):
            y = self.fmul(y, self.fsub(self.fconst(1.5), self.fmul(half, self.fmul(y, y))))
        return y

    def sqrt(self, x: SSAValue) -> SSAValue:
        return self.fmul(x, self.rsqrt(self.fadd(x, self.fconst(0.0))))

    def log(self, x: SSAValue) -> SSAValue:
        """`log(x)` for x > 0: split the exponent off the bit pattern, then an atanh series.

        `m = x / 2**e` lands in [1, 2), so `s = (m-1)/(m+1)` is in [0, 1/3] and the odd series
        `2*(s + s^3/3 + ... + s^9/9)` is accurate to ~1e-7 there.
        """
        u = self.bits(x)
        e = self.sub32(self.and32(self.lshr32(u, 23), self.i32c(0xFF)), self.i32c(127))
        mantissa = self.unbits(self.or32(self.and32(u, self.i32c(0x807FFFFF)), self.i32c(0x3F800000)))
        s = self.fdiv(self.fsub(mantissa, self.fconst(1.0)), self.fadd(mantissa, self.fconst(1.0)))
        s2 = self.fmul(s, s)
        acc = self.fconst(1.0 / 9.0)
        for c in (1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0):
            acc = self.fadd(self.fmul(acc, s2), self.fconst(c))
        log_m = self.fmul(self.fconst(2.0), self.fmul(s, acc))
        ef = self.unbits(self.shl32(self.add32(e, self.i32c(127)), 23))
        # 2**e as a float, then log(2**e) = e * ln2 -- e is recovered from that float rather than
        # converted, so no fp<->int instruction is needed.
        e_as_float = self.fmul(self.fconst(1.0), self.log2_of_pow2(ef))
        return self.fadd(self.fmul(e_as_float, self.fconst(0.6931471805599453)), log_m)

    def log2_of_pow2(self, p: SSAValue) -> SSAValue:
        """`log2(p)` where p is exactly a power of two: recover the exponent as a float.

        `(bits(p) >> 23) - 127` is the integer exponent; adding it to the 1.5*2**23 magic and
        subtracting the magic back turns it into a float without an int->fp instruction.
        """
        e = self.sub32(self.lshr32(self.bits(p), 23), self.i32c(127))
        y = self.unbits(self.add32(self.bits(self.fconst(_MAGIC)), e))
        return self.fsub(y, self.fconst(_MAGIC))

    def powf(self, x: SSAValue, y: SSAValue) -> SSAValue:
        """`x ** y` for x > 0, as exp(y * log(x)) -- the only form a generated lane can take."""
        return self.exp(self.fmul(y, self.log(x)))

    def tanh(self, x: SSAValue) -> SSAValue:
        z = self.exp(self.fmul(self.fclamp(x, -15.0, 15.0), self.fconst(2.0)))
        return self.fsub(self.fconst(1.0), self.fdiv(self.fconst(2.0), self.fadd(z, self.fconst(1.0))))

    def recip(self, x: SSAValue) -> SSAValue:
        return self.fdiv(self.fconst(1.0), x)

    # -- integer bridging --------------------------------------------------------------------
    def sitofp(self, v: SSAValue) -> SSAValue:
        """The f32 nearest to the i64 integer value `v` (the IEEE conversion)."""
        return self.fb.add(llvm.SIToFPOp(v, F32)).results[0]

    def fptosi(self, x: SSAValue) -> SSAValue:
        """`(i64) x`, truncated TOWARD ZERO, built from the IEEE-754 fields of `x`.

        There is no `fptosi` in the LLVM dialect this package builds with, and the kernel is
        branch-free, so the conversion is arithmetic on the bit pattern:
        `|x| = mantissa * 2**(exponent - 23)`, and a truncation is the shift that drops the
        fractional bits.  Exact for every value an i64 can hold.
        """
        fb = self.fb
        wide = fb.add(llvm.SExtOp(self.bits(x), i64)).results[0]
        sign = fb.ashr_i(wide, 63)  # 0 or -1
        exp = fb.sub_i(fb.and_i(fb.ashr_i(wide, 23), fb.const(0xFF)), fb.const(127))
        mant = fb.or_i(fb.and_i(wide, fb.const(0x7FFFFF)), fb.const(0x800000))
        # left shift when the exponent is past the mantissa's 23 fractional bits, right shift
        # otherwise; both amounts are clamped so no shift is wider than the i64 they act on.
        left = fb.smin(fb.max0(fb.sub_i(exp, fb.const(23))), fb.const(39))
        right = fb.smin(fb.max0(fb.sub_i(fb.const(23), exp)), fb.const(63))
        magnitude = fb.lshr_v(fb.shl_v(mant, left), right)
        return fb.sub_i(fb.xor_i(magnitude, sign), sign)  # negate iff sign == -1

    def round_near_even(self, x: SSAValue) -> SSAValue:
        """`x` rounded to the nearest integer, ties to EVEN, still an f32.

        This is the ISA header's own `ROUND_NEAR_EVEN` (`gemmini_params.h`), which is the
        rounding the accumulator's scale unit applies before the readout clamps -- so the
        generated CPU-lane readout and the hardware store path agree element for element.

        Branch-free, per R1.4: adding and then subtracting 2**23 (SIGNED LIKE `x`, so the sum
        never crosses zero) forces the IEEE round-to-nearest-even at the ulp of 1.0.  The
        magnitude matters -- with `|x| < 2**23` the sum lands in `[2**23, 2**24)`, where the f32
        ulp is EXACTLY 1, which is what makes the add round to an integer.  (The more familiar
        1.5 * 2**23 constant only holds for `|x| < 2**22`; checked against the header's own
        ROUND_NEAR_EVEN over 184k values, it disagrees on ~3% of them and this does not.)
        Every f32 with `|x| >= 2**23` is ALREADY an integer, so the two cases are BLENDED on the
        exponent (an integer compare of the IEEE bit patterns, whose order matches numeric order
        for non-negative floats) rather than branched on.
        """
        magic = self.copysign(self.fconst(8388608.0), x)  # 2**23, signed like x
        rounded = self.fsub(self.fadd(x, magic), magic)
        small = self.ashr32(self.sub32(self.bits(self.fabs(x)), self.i32c(0x4B000000)), 31)
        return self.unbits(self.blend32(small, self.bits(rounded), self.bits(x)))

    # -- storage formats ---------------------------------------------------------------------
    def round_bf16(self, x: SSAValue) -> SSAValue:
        """Round an f32 to bf16 precision (round-half-to-even) and keep it in f32."""
        u = self.bits(x)
        bias = self.add32(self.and32(self.lshr32(u, 16), self.i32c(1)), self.i32c(0x7FFF))
        return self.unbits(self.and32(self.add32(u, bias), self.i32c(-0x10000)))

    def load(self, ptr: SSAValue, index: SSAValue, dtype: str) -> SSAValue:
        """Load one element of `dtype` from `ptr[index]`, widened to f32."""
        if dtype == "f32":
            return self.fb.add(llvm.LoadOp(self.fb.gep(ptr, index, F32), F32)).results[0]
        if dtype == "bf16":
            raw = self.fb.add(llvm.LoadOp(self.fb.gep(ptr, index, i16), i16)).results[0]
            wide = self.fb.add(llvm.ZExtOp(raw, i32)).results[0]
            return self.unbits(self.shl32(wide, 16))
        raise ValueError(f"no CPU-lane load for element type {dtype!r}")

    def store(self, value: SSAValue, ptr: SSAValue, index: SSAValue, dtype: str) -> None:
        if dtype == "f32":
            self.fb.add(llvm.StoreOp(value, self.fb.gep(ptr, index, F32)))
            return
        if dtype == "bf16":
            u = self.bits(value)
            bias = self.add32(self.and32(self.lshr32(u, 16), self.i32c(1)), self.i32c(0x7FFF))
            packed = self.lshr32(self.add32(u, bias), 16)
            half = self.fb.add(llvm.TruncOp(packed, i16)).results[0]
            self.fb.add(llvm.StoreOp(half, self.fb.gep(ptr, index, i16)))
            return
        raise ValueError(f"no CPU-lane store for element type {dtype!r}")


# ---- dense_layout_loop -------------------------------------------------------------------------
# Emit a bounded dense transpose through an existing scalar loop builder.
#
# The caller owns immutable source storage and a disjoint private destination for
# the complete synchronous copy. This helper grants no device or alias authority.


def emit_dense_transpose(builder, source, destination, rows, cols, *, max_elements):
    """Copy dense i8 ``source[rows, cols]`` to ``destination[cols, rows]``.

    Validate the complete static work bound before emitting any operation. The
    builder supplies ordinary host loops, address arithmetic and byte accesses;
    no input value, numerical recipe or accelerator command is inspected.
    """
    if any(type(value) is not int or value <= 0 for value in (rows, cols, max_elements)):
        raise ValueError("dense transpose needs positive integer extents and work bound")
    if rows * cols > max_elements or rows * cols >= 1 << 63:
        raise ValueError("dense transpose exceeds its selected work/address bound")
    if source == destination:
        raise ValueError("dense transpose requires distinct private destination storage")

    def row(ri):
        def column(ci):
            src = builder.op("add", builder.op("mul", ri, cols), ci)
            dst = builder.op("add", builder.op("mul", ci, rows), ri)
            value = builder.load(builder.gep(source, src), "i8")
            builder.store(value, builder.gep(destination, dst), "i8")

        builder.loop(0, cols, 1, column)

    builder.loop(0, rows, 1, row)

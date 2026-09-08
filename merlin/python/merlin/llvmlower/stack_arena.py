"""Move a whole-model entrypoint's STATIC STACK temporaries into one module-level arena.

THE DEFECT, MEASURED. A target declares a maximum static frame for its kernel entrypoint, and the
compiler's host lane hoists one ``alloca`` per intermediate into that frame with no reuse. On
ResNet-50 the frame is 816 bytes and nobody noticed; on SmolVLA's ``flow_denoise`` the same shape
asks for **99,897,984 bytes against a 65,536-byte budget** -- 1,534 allocas whose declared sizes
account for 100.0% of the demand. The frame scales with the model, so the largest workloads are
exactly the ones it stops.

WHY BYTE REUSE IS NOT THE FIX, established rather than assumed. The obvious remedy is to colour the
allocations and share bytes, as :mod:`merlin.llvmlower.arena_bind` does for the heap. It cannot
work here for two independent reasons:

1. **The largest single allocation on SmolVLA is 489,000 bytes -- 7.5x the entire budget.** Perfect
   reuse still does not fit, so no colouring however good makes the frame legal.
2. **An ``alloca`` has no ``free``.** ``arena_bind``'s soundness rests on the dynamic
   ``[malloc, free)`` window being a live range it never has to recover from the IR's shape. A stack
   slot's live range runs to the end of the function unless a lifetime intrinsic says otherwise, and
   this IR has none (measured: 0 ``llvm.lifetime``, 0 ``llvm.stacksave``). So no two ranges are
   PROVABLY disjoint, and sharing bytes would be a guess.

So this pass **never shares bytes**. Every bound allocation gets its own, and the arena is the sum
of what it binds. That is not a compromise: the win being claimed is moving the storage from a
64 KiB stack to ``.bss`` on a target with 16 GiB of DRAM, and refusing to share is what makes the
transform sound with no liveness analysis at all -- there is no lifetime mistake available to make.
Reuse remains a separate, later, and independently measurable optimization.

WHAT IT REFUSES, each for a stated reason. An allocation failing any check is LEFT ON THE STACK and
counted in the report with its reason, exactly as ``arena_bind`` does:

* a non-constant element count -- there is no static size to reserve;
* an allocation whose block lies on a CFG cycle -- two dynamic instances of one site would then
  coexist and alias the single slot, which is the one way this transform can produce wrong numbers;
* an allocation in a block the entry cannot reach -- reserving bytes nothing writes;
* an entrypoint that can call itself -- a static arena is shared by every activation, so recursion
  would have the inner call scribble on the outer's temporaries.

THE ARENA MAKES THE ENTRYPOINT NON-REENTRANT, and that is a real change to its contract rather than
a footnote. Two concurrent activations share the arena. The whole-model harness invokes the kernel
sequentially on one core, which is why this is admissible there; a caller that does otherwise must
not use it, and :func:`bind_stack_arena` records the property in its report so a caller cannot be
unaware of it.

WHY THIS SEAM. The transform runs on the emitted LLVM IR, the last point the measured path still
passes through, and leaves the MLIR untouched. That matters for more than convenience:
``merlin.perf.host_cfg_activity`` keys its allocation accounting on the ``llvm.alloca`` operation and
its ``merlin.host_storage_bytes`` attribute. Rewriting the MLIR would move the storage out from
under that analysis, which would then silently report zero host allocation -- a number that looks
like an improvement and is an instrument failure. At this seam the MLIR still says exactly what the
program asks for.

NOTHING HERE NAMES A TARGET. The entry symbol and the frame budget are the target's own declaration
(``harness_build_recipe(target).require_kernel_stack_frame()``), threaded in by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .arena_bind import (ArenaBindError, _blocks_of, _cyclic_blocks, _reachable,
                         _split_function)
from ..xdsl_dialects.lowering.arena_plan import ARENA_ALIGN, _align_up

__all__ = ["bind_stack_arena", "StackArenaError", "StackArenaReport", "STACK_ARENA_SYMBOL",
           "element_bytes_of"]

#: The symbol the stack arena is emitted under. Internal linkage and zero-initialized, so it lands
#: in ``.bss`` and costs image size only on a loader that materializes ``.bss``. Deliberately
#: distinct from :data:`merlin.llvmlower.arena_bind.ARENA_SYMBOL`: the two passes may both run, they
#: bind different storage under different arguments, and one symbol would silently overlay them.
STACK_ARENA_SYMBOL = "merlin_stack_arena"

#: Byte widths for the LLVM first-class types an ``alloca`` in this pipeline is emitted with. An
#: unlisted type is REFUSED rather than guessed: a wrong width reserves the wrong number of bytes,
#: which is the one error that produces plausible output instead of a crash.
_LLVM_TYPE_BYTES: dict[str, int] = {
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "i128": 16,
    "half": 2, "bfloat": 2, "float": 4, "double": 8, "fp128": 16,
    "ptr": 8,
}


class StackArenaError(RuntimeError):
    """The module cannot be read well enough to transform it safely, so nothing is rewritten."""


def element_bytes_of(llvm_type: str) -> int:
    """Bytes for one element of ``llvm_type``, or raise. Never defaulted."""
    width = _LLVM_TYPE_BYTES.get(llvm_type.strip())
    if width is None:
        raise StackArenaError(
            f"alloca element type {llvm_type.strip()!r} is not one this pass can size "
            f"(it knows {sorted(_LLVM_TYPE_BYTES)}); reserving a guessed width would under- or "
            f"over-allocate silently")
    return width


@dataclass
class _Slot:
    """One ``alloca`` site considered for the arena."""

    name: str            # SSA name, e.g. "%1234"
    line: int            # index into the function's line list
    block: int           # block index
    elem_type: str
    count: int
    align: int
    bytes: int


@dataclass
class StackArenaReport:
    """What moved, what did not, and what the transform now requires of its caller."""

    entry_symbol: str = ""
    symbol: str = STACK_ARENA_SYMBOL
    #: Frame bytes the bound allocations were asking of the stack.
    moved_bytes: int = 0
    #: Arena bytes reserved. Equal to ``moved_bytes`` plus per-slot alignment padding, because this
    #: pass never shares bytes.
    arena_bytes: int = 0
    n_bound: int = 0
    n_refused: int = 0
    #: Frame bytes still demanded by the allocations left on the stack.
    remaining_stack_bytes: int = 0
    refusals: list[dict[str, Any]] = field(default_factory=list)
    #: Stated, not implied: the arena is one shared object per module.
    reentrancy: str = ("the entrypoint is NOT reentrant after this transform -- two concurrent or "
                       "nested activations share the arena. The whole-model harness invokes the "
                       "kernel sequentially on one core, which is the condition under which this "
                       "is admissible")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_stack_arena_bind_v1", "entry_symbol": self.entry_symbol,
                "symbol": self.symbol, "moved_bytes": self.moved_bytes,
                "arena_bytes": self.arena_bytes, "n_bound": self.n_bound,
                "n_refused": self.n_refused,
                "remaining_stack_bytes": self.remaining_stack_bytes,
                "shares_bytes": False,
                "why_no_sharing": ("an alloca has no free, so no two live ranges are provably "
                                   "disjoint; and on the measured model the largest single "
                                   "allocation alone exceeds the whole frame budget, so no "
                                   "colouring would make it fit"),
                "reentrancy": self.reentrancy,
                "refusals": [dict(r) for r in self.refusals]}


def _parse_alloca(inst: str) -> tuple[str, str, int, int] | None:
    """``(ssa_name, elem_type, count, align)`` for a constant-count alloca, else ``None``.

    Parsed structurally by splitting on the punctuation LLVM's own printer emits -- never by
    pattern-matching a spelling. An alloca this cannot read returns ``None`` and is refused, which
    leaves it on the stack; it is never treated as if it had been understood.
    """
    text = inst.strip()
    name, sep, rest = text.partition(" = ")
    if not sep or not name.startswith("%"):
        return None
    rest = rest.strip()
    keyword, sep, operands = rest.partition(" ")
    if sep is None or keyword != "alloca":
        return None
    # `alloca [inalloca] <ty> [, <ty> <NumElements>] [, align <n>] [, addrspace(<n>)]`
    parts = [piece.strip() for piece in operands.split(",")]
    if not parts or not parts[0]:
        return None
    elem_type = parts[0]
    count = 1
    align = 1
    for piece in parts[1:]:
        head, _, tail = piece.partition(" ")
        if head == "align":
            if not tail.strip().isdigit():
                return None
            align = int(tail.strip())
        elif head == "addrspace":
            # A non-default address space is not a plain stack slot; refuse it.
            return None
        else:
            # The element-count operand: `<ty> <NumElements>`.
            if not tail.strip().isdigit():
                return None            # a dynamic count has no static size
            count = int(tail.strip())
    return name, elem_type, count, align


def bind_stack_arena(ll_text: str, *, entry_symbol: str,
                     symbol: str = STACK_ARENA_SYMBOL) -> tuple[str, StackArenaReport]:
    """Rewrite ``@entry_symbol``'s provable static allocas into one arena. ``(ll, report)``.

    With nothing bindable the text is returned UNCHANGED -- byte-identical, not merely equivalent --
    and no arena global is emitted, so a module this cannot help compiles exactly as before. That
    property is what lets the caller apply this only to a program whose frame has been MEASURED to
    exceed its budget, leaving every already-fitting build untouched.
    """
    if not entry_symbol:
        raise StackArenaError("the entrypoint symbol is required; it is the target's declaration")
    lines = ll_text.split("\n")
    try:
        start, end = _split_function(lines, entry_symbol)
    except ArenaBindError as exc:
        raise StackArenaError(str(exc)) from exc

    report = StackArenaReport(entry_symbol=entry_symbol, symbol=symbol)

    # A static arena is one object per module, so a self-call would let an inner activation
    # overwrite an outer one's temporaries.
    body = "\n".join(lines[start + 1:end])
    if f"@{entry_symbol}(" in body:
        raise StackArenaError(
            f"@{entry_symbol} contains a call to itself; a static arena is shared by every "
            f"activation, so a recursive entrypoint would scribble on its own caller's temporaries")

    try:
        blocks = _blocks_of(lines, start, end)
    except ArenaBindError as exc:
        raise StackArenaError(str(exc)) from exc
    by_label = {block.label: index for index, block in enumerate(blocks)}
    succs: list[list[int]] = []
    for block in blocks:
        resolved = []
        for label in block.succs:
            if label not in by_label:
                raise StackArenaError(
                    f"block {block.label} branches to {label}, which is not a block of "
                    f"@{entry_symbol}; a CFG with a missing edge yields a cycle set that is wrong "
                    f"in the permissive direction")
            resolved.append(by_label[label])
        succs.append(resolved)
    cyclic = _cyclic_blocks(succs)
    reachable = _reachable(succs)

    slots: list[_Slot] = []
    for index, block in enumerate(blocks):
        for line_index in block.insts:
            inst = lines[line_index]
            if " = alloca " not in inst:
                continue
            parsed = _parse_alloca(inst)
            if parsed is None:
                report.n_refused += 1
                report.refusals.append({"line": line_index, "reason": (
                    "the alloca's element count, alignment or address space could not be read as a "
                    "static size, so no arena bytes can be reserved for it"),
                    "instruction": inst.strip()[:200]})
                continue
            name, elem_type, count, align = parsed
            try:
                width = element_bytes_of(elem_type)
            except StackArenaError as exc:
                report.n_refused += 1
                report.refusals.append({"line": line_index, "name": name,
                                        "reason": str(exc)})
                continue
            size = width * count
            if index in cyclic:
                report.n_refused += 1
                report.remaining_stack_bytes += size
                report.refusals.append({"line": line_index, "name": name, "bytes": size,
                                        "reason": ("its block lies on a CFG cycle, so two dynamic "
                                                   "instances of this one site could coexist and "
                                                   "would alias a single arena slot")})
                continue
            if index not in reachable:
                report.n_refused += 1
                report.refusals.append({"line": line_index, "name": name, "bytes": size,
                                        "reason": ("its block is unreachable from the entry, so "
                                                   "arena bytes reserved for it would never be "
                                                   "written")})
                continue
            slots.append(_Slot(name=name, line=line_index, block=index, elem_type=elem_type,
                               count=count, align=align, bytes=size))

    if not slots:
        return ll_text, report

    # NO SHARING. Each slot gets its own bytes, at least as aligned as it asked for.
    cursor = 0
    offsets: dict[str, int] = {}
    for slot in slots:
        cursor = _align_up(cursor, max(int(slot.align), 1))
        offsets[slot.name] = cursor
        cursor += slot.bytes
    arena_bytes = _align_up(cursor, ARENA_ALIGN)

    out = list(lines)
    for slot in slots:
        out[slot.line] = (
            f"  {slot.name} = getelementptr inbounds i8, ptr @{symbol}, "
            f"i64 {offsets[slot.name]}  ; was `alloca {slot.elem_type}, i64 {slot.count}, "
            f"align {slot.align}` ({slot.bytes} bytes)")

    alignment = max([ARENA_ALIGN, *(int(slot.align) for slot in slots)])
    out.insert(start, f"@{symbol} = internal global [{arena_bytes} x i8] zeroinitializer, "
                      f"align {alignment}")

    report.n_bound = len(slots)
    report.moved_bytes = sum(slot.bytes for slot in slots)
    report.arena_bytes = arena_bytes
    _assert_slots_disjoint(offsets, {slot.name: slot.bytes for slot in slots}, arena_bytes)
    return "\n".join(out), report


def _assert_slots_disjoint(offsets: dict[str, int], sizes: dict[str, int],
                           arena_bytes: int) -> None:
    """Prove the placement this pass just computed: no two slots overlap, none runs off the end.

    A self-check rather than a test, because the failure it guards is silent. This pass promises no
    sharing; an off-by-one in the cursor would share bytes anyway and produce wrong numbers with no
    crash, and the promise is exactly the thing the soundness argument rests on.
    """
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    for (name, offset), (next_name, next_offset) in zip(ordered, ordered[1:]):
        if offset + sizes[name] > next_offset:
            raise StackArenaError(
                f"placement bug: {name} at {offset} spans {sizes[name]} bytes and would overlap "
                f"{next_name} at {next_offset}; this pass promises no sharing")
    for name, offset in ordered:
        if offset + sizes[name] > arena_bytes:
            raise StackArenaError(
                f"placement bug: {name} at {offset} spans {sizes[name]} bytes, past the "
                f"{arena_bytes}-byte arena")

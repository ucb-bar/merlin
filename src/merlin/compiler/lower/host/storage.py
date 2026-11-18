"""Move a whole-model entrypoint's STATIC STACK temporaries into one module-level arena.

THE DEFECT, MEASURED. A target declares a maximum static frame for its kernel entrypoint, and the
compiler's host lane hoists one ``alloca`` per intermediate into that frame with no reuse. On
ResNet-50 the frame is 816 bytes and nobody noticed; on SmolVLA's ``flow_denoise`` the same shape
asks for **99,897,984 bytes against a 65,536-byte budget** -- 1,534 allocas whose declared sizes
account for 100.0% of the demand. The frame scales with the model, so the largest workloads are
exactly the ones it stops.

WHY BYTE REUSE IS NOT THE FIX, established rather than assumed. The obvious remedy is to colour the
allocations and share bytes, as :mod:`merlin.compiler.lower.host.arena` does for the heap. It cannot
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
``merlin.analysis.programs.cfg_activity`` keys its allocation accounting on the ``llvm.alloca`` operation and
its ``merlin.host_storage_bytes`` attribute. Rewriting the MLIR would move the storage out from
under that analysis, which would then silently report zero host allocation -- a number that looks
like an improvement and is an instrument failure. At this seam the MLIR still says exactly what the
program asks for.

NOTHING HERE NAMES A TARGET. The entry symbol and the frame budget are the target's own declaration
(``harness_build_recipe(target).require_kernel_stack_frame()``), threaded in by the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from math import prod
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file
from merlin.compiler.lower.host.arena import (
    ArenaBindError,
    _blocks_of,
    _cyclic_blocks,
    _reachable,
    _split_function,
)
from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
from merlin.compiler.lower.host.writers import (
    FreshTensorWriterContract,
    PrivateWorkspaceContract,
    pool_private_writer_workspaces,
    rewrite_fresh_tensor_writers,
    validate_private_workspaces,
)
from merlin.compiler.partition.memory_plan import ARENA_ALIGN, _align_up
from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan, SpatialProducerPlan

# ---- stack_arena -------------------------------------------------------------------------------

#: The symbol the stack arena is emitted under. Internal linkage and zero-initialized, so it lands
#: in ``.bss`` and costs image size only on a loader that materializes ``.bss``. Deliberately
#: distinct from :data:`merlin.compiler.lower.host.arena.ARENA_SYMBOL`: the two passes may both run, they
#: bind different storage under different arguments, and one symbol would silently overlay them.
STACK_ARENA_SYMBOL = "merlin_stack_arena"

#: Byte widths for the LLVM first-class types an ``alloca`` in this pipeline is emitted with. An
#: unlisted type is REFUSED rather than guessed: a wrong width reserves the wrong number of bytes,
#: which is the one error that produces plausible output instead of a crash.
_LLVM_TYPE_BYTES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "i128": 16,
    "half": 2,
    "bfloat": 2,
    "float": 4,
    "double": 8,
    "fp128": 16,
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
            f"over-allocate silently"
        )
    return width


@dataclass
class _Slot:
    """One ``alloca`` site considered for the arena."""

    name: str  # SSA name, e.g. "%1234"
    line: int  # index into the function's line list
    block: int  # block index
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
    reentrancy: str = (
        "the entrypoint is NOT reentrant after this transform -- two concurrent or "
        "nested activations share the arena. The whole-model harness invokes the "
        "kernel sequentially on one core, which is the condition under which this "
        "is admissible"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "merlin_stack_arena_bind_v1",
            "entry_symbol": self.entry_symbol,
            "symbol": self.symbol,
            "moved_bytes": self.moved_bytes,
            "arena_bytes": self.arena_bytes,
            "n_bound": self.n_bound,
            "n_refused": self.n_refused,
            "remaining_stack_bytes": self.remaining_stack_bytes,
            "shares_bytes": False,
            "why_no_sharing": (
                "an alloca has no free, so no two live ranges are provably "
                "disjoint; and on the measured model the largest single "
                "allocation alone exceeds the whole frame budget, so no "
                "colouring would make it fit"
            ),
            "reentrancy": self.reentrancy,
            "refusals": [dict(r) for r in self.refusals],
        }


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
                return None  # a dynamic count has no static size
            count = int(tail.strip())
    return name, elem_type, count, align


def bind_stack_arena(
    ll_text: str, *, entry_symbol: str, symbol: str = STACK_ARENA_SYMBOL
) -> tuple[str, StackArenaReport]:
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
    body = "\n".join(lines[start + 1 : end])
    if f"@{entry_symbol}(" in body:
        raise StackArenaError(
            f"@{entry_symbol} contains a call to itself; a static arena is shared by every "
            f"activation, so a recursive entrypoint would scribble on its own caller's temporaries"
        )

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
                    f"in the permissive direction"
                )
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
                report.refusals.append(
                    {
                        "line": line_index,
                        "reason": (
                            "the alloca's element count, alignment or address space could not be read as a "
                            "static size, so no arena bytes can be reserved for it"
                        ),
                        "instruction": inst.strip()[:200],
                    }
                )
                continue
            name, elem_type, count, align = parsed
            try:
                width = element_bytes_of(elem_type)
            except StackArenaError as exc:
                report.n_refused += 1
                report.refusals.append({"line": line_index, "name": name, "reason": str(exc)})
                continue
            size = width * count
            if index in cyclic:
                report.n_refused += 1
                report.remaining_stack_bytes += size
                report.refusals.append(
                    {
                        "line": line_index,
                        "name": name,
                        "bytes": size,
                        "reason": (
                            "its block lies on a CFG cycle, so two dynamic "
                            "instances of this one site could coexist and "
                            "would alias a single arena slot"
                        ),
                    }
                )
                continue
            if index not in reachable:
                report.n_refused += 1
                report.refusals.append(
                    {
                        "line": line_index,
                        "name": name,
                        "bytes": size,
                        "reason": (
                            "its block is unreachable from the entry, so "
                            "arena bytes reserved for it would never be "
                            "written"
                        ),
                    }
                )
                continue
            slots.append(
                _Slot(
                    name=name, line=line_index, block=index, elem_type=elem_type, count=count, align=align, bytes=size
                )
            )

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
            f"align {slot.align}` ({slot.bytes} bytes)"
        )

    alignment = max([ARENA_ALIGN, *(int(slot.align) for slot in slots)])
    out.insert(start, f"@{symbol} = internal global [{arena_bytes} x i8] zeroinitializer, align {alignment}")

    report.n_bound = len(slots)
    report.moved_bytes = sum(slot.bytes for slot in slots)
    report.arena_bytes = arena_bytes
    _assert_slots_disjoint(offsets, {slot.name: slot.bytes for slot in slots}, arena_bytes)
    return "\n".join(out), report


def _assert_slots_disjoint(offsets: dict[str, int], sizes: dict[str, int], arena_bytes: int) -> None:
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
                f"{next_name} at {next_offset}; this pass promises no sharing"
            )
    for name, offset in ordered:
        if offset + sizes[name] > arena_bytes:
            raise StackArenaError(
                f"placement bug: {name} at {offset} spans {sizes[name]} bytes, past the {arena_bytes}-byte arena"
            )


# ---- spatial_scratch_abi -----------------------------------------------------------------------
# Private caller-owned scratch for a source-bound staged spatial producer.
#
# Reuse the ordinary fresh writer and workspace pool. Public tensor signatures
# remain unchanged; only a private borrowed helper receives the byte descriptor.
# The supplied source/effect contracts remain independently owned premises, not
# qualifications inferred from pointer shape, source hashes or generated code.


@dataclass(frozen=True)
class SpatialScratchArtifact:
    module: object
    bridge_source: str
    writer_contract: FreshTensorWriterContract
    pointer_alignments: tuple[int, ...]
    owner_symbol: str
    selected_source_sha256: str
    plan_sha256: str
    receipt: dict


def _symbol(value):
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (c.isalnum() or c == "_") for c in value)
    ):
        raise ValueError("spatial scratch requires a plain explicit C symbol")
    return value


def spatial_logical_buffers(plan):
    """Exact original logical pointer shapes; private scratch is excluded."""
    if type(plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan):
        raise ValueError("spatial scratch requires an exact producer plan")
    plan.__post_init__()
    if type(plan) is PackedSpatialProducerPlan:
        plan = plan.source
    p, s = plan.producer, plan.spatial
    batch, _, _, ci = p.window.input_shape
    kh, kw = p.window.kernel
    oh, ow = s.output_hw
    buffers = [("input", p.window.input_shape, "i8"), ("weight", (kh * kw * ci, p.output_channels), "i8")]
    if p.bias:
        buffers.append(("bias", (p.output_channels,), "i32"))
    buffers.append(("output", (batch * oh * ow, p.output_channels), "i8"))
    return tuple(buffers)


def _transport_preamble(buffers, *, kernel_symbol=None):
    ranks = sorted({1, *(len(shape) for _, shape, _ in buffers)})
    descriptors = "\n".join(
        f"typedef struct {{ void *allocated,*aligned; int64_t offset,size[{rank}],stride[{rank}]; }} M{rank};"
        for rank in ranks
    )
    external = (
        f"extern void {_symbol(kernel_symbol)}({','.join('void*' for _ in range(len(buffers) + 1))});\n"
        if kernel_symbol is not None
        else ""
    )
    return f"""#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(uintptr_t)==8 && sizeof(intptr_t)==8,"selected ranked caller requires 64-bit pointers");
{descriptors}
{external}static int spatial_span(void *aligned,int64_t offset,uintptr_t width,uintptr_t elements,
                        uintptr_t *address,uintptr_t *bytes) {{
  if (!aligned || offset<0 || !width || !elements) return 0;
  uintptr_t limit=(uintptr_t)INTPTR_MAX/width, off=(uintptr_t)offset, base=(uintptr_t)aligned;
  if (elements>limit || off>limit-elements) return 0;
  uintptr_t delta=off*width, size=elements*width;
  if (delta>UINTPTR_MAX-base) return 0;
  uintptr_t first=base+delta;
  if (size>UINTPTR_MAX-first) return 0;
  *address=first;*bytes=size;return 1;
}}
static int spatial_disjoint(uintptr_t a,uintptr_t an,uintptr_t b,uintptr_t bn) {{
  if (an>UINTPTR_MAX-a || bn>UINTPTR_MAX-b) return 0;
  return a+an<=b || b+bn<=a;
}}
"""


def _bridge(plan, contract, workspace, kernel_symbol, alignments):
    buffers = spatial_logical_buffers(plan)
    parameters = [f"const M{len(shape)} *d{index}" for index, (_, shape, _) in enumerate(buffers)]
    parameters.append("const M1 *scratch")
    lines = [f"void _mlir_ciface_{_symbol(contract.borrowed_symbol)}({', '.join(parameters)}) {{"]
    count = len(buffers) + 1
    lines.extend(
        (f"  uintptr_t address[{count}], bytes[{count}];", f"  uintptr_t metadata[{count}], metadata_bytes[{count}];")
    )
    for index, (_, shape, dtype) in enumerate(buffers):
        name = f"d{index}"
        row_stride = 1
        checks = [f"!{name}"]
        for axis in reversed(range(len(shape))):
            checks.extend(
                (f"{name}->size[{axis}] != INT64_C({shape[axis]})", f"{name}->stride[{axis}] != INT64_C({row_stride})")
            )
            row_stride *= shape[axis]
        width = int(dtype[1:]) // 8
        lines.extend(
            (
                f"  if ({' || '.join(checks)}) __builtin_trap();",
                f"  if (!spatial_span({name}->aligned,{name}->offset,{width},UINT64_C({prod(shape)}),",
                f"       &address[{index}],&bytes[{index}]) ||",
                f"       address[{index}] % UINT64_C({alignments[index]})) __builtin_trap();",
                f"  metadata[{index}]=(uintptr_t){name}; metadata_bytes[{index}]=sizeof(*{name});",
            )
        )
    index = len(buffers)
    lines.extend(
        (
            "  if (!scratch || scratch->stride[0] != 1 ||",
            f"      scratch->size[0] < INT64_C({workspace.bytes}) ||",
            "      !spatial_span(scratch->aligned,scratch->offset,1,(uintptr_t)scratch->size[0],",
            f"          &address[{index}],&bytes[{index}]) ||",
            f"      address[{index}] % UINT64_C({workspace.alignment})) __builtin_trap();",
            f"  metadata[{index}]=(uintptr_t)scratch; metadata_bytes[{index}]=sizeof(*scratch);",
            f"  for (unsigned i=0;i<{count};++i) {{",
            "    if (metadata_bytes[i] > UINTPTR_MAX-metadata[i]) __builtin_trap();",
            f"    for (unsigned j=0;j<{count};++j) {{",
            "      if (!spatial_disjoint(address[i],bytes[i],metadata[j],metadata_bytes[j])) __builtin_trap();",
            "      if (i<j && !spatial_disjoint(address[i],bytes[i],address[j],bytes[j])) __builtin_trap();",
            "    }",
            "  }",
            f"  {_symbol(kernel_symbol)}({','.join(f'(void*)address[{n}]' for n in range(count))});",
            "}",
        )
    )
    preamble = _transport_preamble(buffers, kernel_symbol=kernel_symbol)
    return preamble + "\n".join(lines) + "\n"


def prepare_spatial_scratch_abi(
    module,
    plan,
    contract,
    workspace,
    *,
    kernel_symbol,
    pointer_alignments,
    source_binding_sha256,
    stack_budget_bytes=None,
):
    """Return a detached normal writer/pool preparation and its descriptor bridge.

    The original caller separately binds the exact source pair, numerical DAG,
    device image and effects. ``workspace`` carries that caller's explicit
    noescape/initialized-before-read/completion premises. Capacity validation
    and safe generated pointer transport cannot establish those premises.
    Accessible allocations and target DMA visibility remain runtime obligations.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp, TensorType, i8, i32

    from merlin.compiler.dialects.xdsl_context import text

    if type(module) is not ModuleOp or type(plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan):
        raise ValueError("spatial scratch requires current typed source and producer plan")
    plan.__post_init__()
    source_plan = plan.source if type(plan) is PackedSpatialProducerPlan else plan
    if source_plan.readout_mode not in ("staged_readouts", "single_readout"):
        raise ValueError("spatial scratch ABI requires a checked original readout realization")
    if (
        type(contract) is not FreshTensorWriterContract
        or contract.private_workspaces
        or type(contract.borrowed_effects) is not BorrowedBufferEffects
        or contract.declaration_abi != "ranked_c"
    ):
        raise ValueError("spatial writer needs a separate ranked borrowed contract and complete effects")
    contract.borrowed_effects.__post_init__()
    validate_private_workspaces((workspace,))
    if type(plan) is PackedSpatialProducerPlan:
        required = plan.workspace.start + plan.required_bytes
    elif source_plan.readout_mode == "staged_readouts":
        required = plan.intermediate.start + plan.temporary_rows * plan.producer.column_tile
    else:
        # Keep the existing private pointer ABI even when a complete numerical
        # proof eliminates the byte stripe. The caller still supplies checked
        # positive storage (which may be a one-byte bounded stack workspace).
        required = 0
    if type(workspace) is not PrivateWorkspaceContract or workspace.bytes < required:
        raise ValueError("private workspace does not cover the complete staged stripe and byte origin")
    if type(plan) is PackedSpatialProducerPlan and workspace.bytes < plan.workspace.start + plan.workspace.size:
        raise ValueError("private owner does not cover the complete assigned packed workspace view")
    if type(plan) is PackedSpatialProducerPlan and workspace.alignment % plan.panel_alignment:
        raise ValueError("private workspace does not guarantee the selected packed panel alignment")
    if (
        type(source_binding_sha256) is not str
        or len(source_binding_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_binding_sha256)
    ):
        raise ValueError("exact source-binding custody identity required")
    buffers = spatial_logical_buffers(plan)
    if (
        type(pointer_alignments) is not tuple
        or len(pointer_alignments) != len(buffers)
        or any(type(n) is not int or not 0 < n < 1 << 63 or n & (n - 1) for n in pointer_alignments)
    ):
        raise ValueError("every original logical pointer requires an explicit power-of-two alignment")
    _symbol(kernel_symbol)
    if contract.result_argument != len(buffers) - 1 or contract.fully_written_arguments != (len(buffers) - 1,):
        raise ValueError("spatial writer must retain its sole original fully written output")
    declaration = next(
        (f for f in module.body.block.ops if isinstance(f, func.FuncOp) and f.sym_name.data == contract.symbol), None
    )
    if declaration is None or tuple(declaration.function_type.inputs) != tuple(
        TensorType({"i8": i8, "i32": i32}[dtype], shape) for _, shape, dtype in buffers
    ):
        raise ValueError("spatial logical signature differs from original source extents and formats")
    before = text(module, generic=True)
    public_types = {
        f.sym_name.data: str(f.function_type)
        for f in module.body.block.ops
        if isinstance(f, func.FuncOp) and getattr(f.sym_visibility, "data", None) != "private"
    }
    selected = module.clone()
    selected_contract = replace(contract, private_workspaces=(workspace,))
    writer = rewrite_fresh_tensor_writers(selected, (selected_contract,))
    pool = pool_private_writer_workspaces(selected, writer, stack_budget_bytes=stack_budget_bytes)
    after_types = {
        f.sym_name.data: str(f.function_type)
        for f in selected.body.block.ops
        if isinstance(f, func.FuncOp) and getattr(f.sym_visibility, "data", None) != "private"
    }
    if public_types != after_types or text(module, generic=True) != before:
        raise ValueError("spatial scratch preparation changed the original public source signature")
    bridge = _bridge(plan, selected_contract, workspace, kernel_symbol, pointer_alignments)
    return SpatialScratchArtifact(
        selected,
        bridge,
        selected_contract,
        pointer_alignments,
        pool["owner_symbol"],
        hashlib.sha256(text(selected, generic=True).encode()).hexdigest(),
        plan.identity,
        dict(
            schema="merlin.spatial_scratch_abi.v1",
            plan_sha256=plan.identity,
            original_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
            selected_source_sha256=hashlib.sha256(text(selected, generic=True).encode()).hexdigest(),
            source_binding_sha256=source_binding_sha256,
            public_signatures=public_types,
            private_workspace=asdict(workspace),
            required_stripe_bytes=required,
            logical_buffers=buffers,
            pointer_alignments=pointer_alignments,
            writer=writer,
            pool=pool,
            bridge_sha256=hashlib.sha256(bridge.encode()).hexdigest(),
            raw_private_helper=kernel_symbol,
            release="normal upstream owned deallocation; arena reclamation not inferred"
            if stack_budget_bytes is None
            else pool["release"],
            target_execution="not_established",
            source_and_effect_qualification="caller supplied; not inferred",
        ),
    )


def emit_spatial_raw_entry(artifact, plan, *, entry_symbol="kernel_entry", standalone=False):
    """Adapt the original flat logical pointers to the normal ranked public owner.

    By default concatenate with this artifact's descriptor bridge. Explicit
    ``standalone=True`` emits transport declarations for a separate object;
    the existing private bridge remains in the normal device shim. The caller must
    lower its actual detached owner through ordinary output-parameter conversion
    and link that exact C-interface definition. No private pointer is added to
    the public signature; allocation/release stay inside the compiled owner.
    """
    from xdsl.dialects import func

    from merlin.compiler.dialects.xdsl_context import text

    if (
        type(artifact) is not SpatialScratchArtifact
        or artifact.plan_sha256 != plan.identity
        or hashlib.sha256(text(artifact.module, generic=True).encode()).hexdigest() != artifact.selected_source_sha256
    ):
        raise ValueError("raw spatial entry requires its exact prepared owner and source plan")
    if type(standalone) is not bool:
        raise ValueError("standalone spatial transport requires an explicit Boolean")
    buffers = spatial_logical_buffers(plan)
    owner_symbol = artifact.owner_symbol
    owner = next(
        op for op in artifact.module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == owner_symbol
    )
    if (
        len(owner.function_type.inputs) != len(buffers) - 1
        or len(owner.function_type.outputs) != 1
        or tuple(owner.function_type.outputs)[0].get_shape() != buffers[-1][1]
    ):
        raise ValueError("raw spatial entry requires the complete original input and sole result owner")
    _symbol(entry_symbol)
    descriptor_types = [f"M{len(shape)}" for _, shape, _ in buffers]
    lines = [
        f"extern void _mlir_ciface_{_symbol(owner_symbol)}({','.join(t + '*' for t in descriptor_types)});",
        f"void {entry_symbol}({','.join('void*p' + str(i) for i in range(len(buffers)))}) {{",
        f"  uintptr_t address[{len(buffers)}],bytes[{len(buffers)}];",
    ]
    for index, (_, shape, dtype) in enumerate(buffers):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        sizes = ",".join(str(n) for n in shape)
        steps = ",".join(str(n) for n in strides)
        lines.extend(
            (
                f"  {descriptor_types[index]} d{index}={{p{index},p{index},0,{{{sizes}}},{{{steps}}}}};",
                f"  if (!spatial_span(p{index},0,{int(dtype[1:]) // 8},UINT64_C({prod(shape)}),",
                f"      &address[{index}],&bytes[{index}]) ||",
                f"      address[{index}] % UINT64_C({artifact.pointer_alignments[index]})) __builtin_trap();",
            )
        )
    count = len(buffers)
    lines.extend(
        (
            f"  for(unsigned i=0;i<{count};++i) for(unsigned j=i+1;j<{count};++j)",
            "    if(!spatial_disjoint(address[i],bytes[i],address[j],bytes[j])) __builtin_trap();",
        )
    )
    for index in range(count):
        lines.extend(
            (
                f"  for(unsigned i=0;i<{count};++i)",
                f"    if(!spatial_disjoint(address[i],bytes[i],(uintptr_t)&d{index},sizeof(d{index})))",
                "      __builtin_trap();",
            )
        )
    lines.extend((f"  _mlir_ciface_{owner_symbol}({','.join('&d' + str(i) for i in range(count))});", "}"))
    return (_transport_preamble(buffers) if standalone else "") + "\n".join(lines) + "\n"


# ---- host_bitcode_transport --------------------------------------------------------------------
# Explicit LLVM assembly transport for large constant-bearing host modules.
#
# Textual LLVM can exceed Clang's source-location capacity even when the module's
# executable code is small. The selected installation's llvm-as serializes the
# whole unchanged module to bitcode, without optimization or constant removal.
# The ordinary compiler then consumes that observed product with its original
# target/ABI/optimization flags. This is transport custody, not numerical proof.


def assemble_host_bitcode(source, work, *, compiler, compilation, runner):
    """Use an explicitly selected compiler's paired assembler; no fallback."""
    source = Path(source).resolve(strict=True)
    compiler = Path(compiler).resolve(strict=True)
    assembler = compiler.with_name("llvm-as").resolve(strict=True)
    if not source.is_file() or not compiler.is_file() or not assembler.is_file():
        raise ValueError("host bitcode requires ordinary selected source and tool files")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    bitcode = work / "model.bc"
    compilation.run(
        [assembler, source, "-o", bitcode],
        runner=runner,
        inputs=[source],
        output=bitcode,
    )
    command = compilation.record["commands"][-1]
    if sha256_file(source) != command["inputs"][0]["sha256"]:
        raise ValueError("host LLVM source changed during bitcode assembly")
    return bitcode, {
        "schema": "merlin.host_bitcode_transport.v1",
        "source": command["inputs"][0],
        "bitcode": command["output"],
        "assembler": command["executable"],
        "scope": "whole-module LLVM serialization; no optimization or execution qualification",
    }


__all__ = [
    "STACK_ARENA_SYMBOL",
    "StackArenaError",
    "StackArenaReport",
    "bind_stack_arena",
    "element_bytes_of",
]

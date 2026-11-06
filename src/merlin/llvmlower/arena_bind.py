"""Bind a whole-model function's heap allocations to ONE statically planned arena.

The measured whole-model path is monolithic: ``one-shot-bufferize`` turns every ``tensor.empty``
into a ``memref.alloc``, the ownership-based deallocation passes pair each with a ``memref.dealloc``,
and ``finalize-memref-to-llvm`` emits them as ``call ptr @malloc(i64 N)`` / ``call void @free(ptr)``
inside ``@forward``. So a single inference performs one heap allocation per intermediate --
measured on the int8 builds: deepjscc 112, small_llama 121, lstmnetvit 175 -- while ExecuTorch's
runner (``EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON``) plans one workspace at delegate init and
allocates nothing in the timed loop.

This module closes that gap AT THE LAST SEAM THE MEASURED PATH STILL PASSES THROUGH: the emitted
LLVM IR. It reads ``@forward``, proves which allocations it may seat in a shared arena, colours them
with :func:`~merlin.xdsl_dialects.lowering.arena_plan.pack_disjoint` (the same placement core the
DispatchProgram planner uses), and rewrites each proven ``malloc`` into a ``getelementptr`` into one
module-level arena, deleting its ``free``. Nothing else in the module is touched.

Why not the dispatch-program planner directly: ``arena_plan.plan_arena`` plans a
:class:`~merlin.xdsl_dialects.lowering.dispatch_program.DispatchProgram`, and the K1 build does not
produce one -- it lowers the monolithic ``@forward``. ``docs/design/static_arena_wiring.md`` records
what routing the build through the outlined lane would take (a per-kernel emission path and a C
replay engine that does not exist). This binder needs neither: it plans the buffers the build
actually emits.

WHY THIS IS SAFE, stated as the argument it is -- an arena that reuses bytes turns a lifetime
mistake into wrong numbers, not a crash, so the reasoning has to be checkable:

1. The input program is already correct under ``malloc``/``free``. So no access to a buffer happens
   outside its dynamic ``[malloc, free)`` window; that window is a sound over-approximation of the
   live range, and we never have to recover liveness from the IR's shape.
2. A bound site executes AT MOST ONCE per call. Both its ``malloc`` block and its ``free`` block are
   required to lie in no CFG cycle, so there is exactly one window per call and two dynamic
   instances of the same site can never coexist.
3. Two bound buffers share bytes only when one's ``free`` DOMINATES the other's ``malloc`` -- every
   path reaching the second allocation has already executed the first free. With (2) that makes the
   windows disjoint, which is exactly the property the arena needs.
4. Nothing outlives the call and nothing else frees the buffer: the HANDLE ``malloc`` returned is
   followed through every value that may carry it (the descriptor it is inserted into, the local
   ``alloca`` the descriptor is spilled to, values loaded back from there) and each must stay local
   and reach only callees that cannot keep or free it; the function returns void.

Every allocation that fails ANY of those is LEFT ALONE -- still a ``malloc``/``free`` pair, still
correct, counted in the report as ``refused`` with a reason. The pass never guesses; a program it
cannot read at all raises :class:`ArenaBindError` rather than emitting a partially rewritten module.

Default OFF. ``lower_model`` calls this only when asked (``static_arena=True`` or
``MERLIN_STATIC_ARENA=1``), so an unflagged build is byte-identical to the frozen baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..xdsl_dialects.lowering.arena_plan import ARENA_ALIGN, _align_up, pack_disjoint

#: The symbol the arena is emitted under. Internal linkage, zero-initialized, so it lands in .bss
#: and costs image size only on a target whose loader materializes .bss (it does not on Linux).
ARENA_SYMBOL = "merlin_arena"

#: LLVM identifier characters that may follow a ``%`` in an unquoted SSA name.
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")

#: Opcodes allowed to consume the RAW (unaligned) malloc pointer.
#:
#: ``ptrtoint`` is the alignment round-up the memref lowering always emits. ``insertvalue`` is the
#: memref descriptor being assembled -- field 0 of ``{ptr, ptr, i64, [N x i64], [N x i64]}`` is the
#: ALLOCATED pointer -- which the ``memref.copy`` lowering builds, spills to an ``alloca`` and hands
#: to ``@memrefCopy``. Measured on the deepjscc int8 build: 17 of 112 allocations reach a descriptor
#: this way and nothing else, and refusing them costs 15% of the allocations this pass can remove.
#:
#: Admitting ``insertvalue`` is only sound while the descriptor cannot be kept or freed by anyone, so
#: :func:`_handle_escapes` follows it to every use for EACH allocation. That used to be decided for the
#: whole function -- one unrecognised callee anywhere switched the relaxation off -- and the ownership
#: passes' own ``@dealloc_helper`` is such a callee: on the spectformer int8 build it left 123
#: allocations (32 MiB) on the heap although none of their descriptors reaches it. Anything else
#: consuming the raw pointer -- a store to caller-visible memory, a second free, a return -- refuses
#: the allocation.
_ALLOWED_RAW_USES = ("ptrtoint", "insertvalue")

#: Callees that neither free nor retain a pointer argument past the call. These are the LLVM
#: intrinsics and the MLIR C-runner entry points the memref lowering emits; they are not facts about
#: any target. An unknown callee is not assumed harmless: a handle that reaches one is refused.
_NON_CAPTURING_CALLEES = frozenset(
    {
        "@free",
        "@malloc",
        "@memrefCopy",
        "@llvm.memcpy.p0.p0.i64",
        "@llvm.memset.p0.i64",
        "@llvm.memmove.p0.p0.i64",
        "@llvm.stacksave.p0",
        "@llvm.stackrestore.p0",
    }
)

#: Pure libm calls: value in, value out, no pointer argument. Listed by EXACT name on purpose --
#: prefix matching would admit ``@sincosf`` under ``@sin`` and ``@frexpf`` under ``@f``, and both
#: write through a pointer out-parameter, which is exactly the capture this check exists to catch.
_PURE_LIBM_CALLEES = frozenset(
    {
        "@fabs",
        "@fabsf",
        "@sqrt",
        "@sqrtf",
        "@exp",
        "@expf",
        "@exp2",
        "@exp2f",
        "@expm1",
        "@expm1f",
        "@log",
        "@logf",
        "@log2",
        "@log2f",
        "@log10",
        "@log10f",
        "@log1p",
        "@log1pf",
        "@pow",
        "@powf",
        "@sin",
        "@sinf",
        "@cos",
        "@cosf",
        "@tan",
        "@tanf",
        "@asin",
        "@asinf",
        "@acos",
        "@acosf",
        "@atan",
        "@atanf",
        "@atan2",
        "@atan2f",
        "@sinh",
        "@sinhf",
        "@cosh",
        "@coshf",
        "@tanh",
        "@tanhf",
        "@erf",
        "@erff",
        "@erfc",
        "@erfcf",
        "@cbrt",
        "@cbrtf",
        "@hypot",
        "@hypotf",
        "@fmod",
        "@fmodf",
        "@remainder",
        "@remainderf",
        "@copysign",
        "@copysignf",
        "@ceil",
        "@ceilf",
        "@floor",
        "@floorf",
        "@trunc",
        "@truncf",
        "@round",
        "@roundf",
        "@roundeven",
        "@roundevenf",
        "@rint",
        "@rintf",
        "@nearbyint",
        "@nearbyintf",
        "@fmax",
        "@fmaxf",
        "@fmin",
        "@fminf",
        "@fma",
        "@fmaf",
    }
)

#: ``@llvm.*`` intrinsic families that compute on values only. Matched by prefix because each family
#: is spelled with its overload suffix (``@llvm.maximum.f32``), and every member of a listed family
#: has the same value-only signature.
_PURE_CALLEE_PREFIXES = (
    "@llvm.fabs.",
    "@llvm.sqrt.",
    "@llvm.exp.",
    "@llvm.exp2.",
    "@llvm.log.",
    "@llvm.log2.",
    "@llvm.log10.",
    "@llvm.pow.",
    "@llvm.powi.",
    "@llvm.sin.",
    "@llvm.cos.",
    "@llvm.tan.",
    "@llvm.fma.",
    "@llvm.fmuladd.",
    "@llvm.maximum.",
    "@llvm.minimum.",
    "@llvm.maxnum.",
    "@llvm.minnum.",
    "@llvm.smax.",
    "@llvm.smin.",
    "@llvm.umax.",
    "@llvm.umin.",
    "@llvm.abs.",
    "@llvm.floor.",
    "@llvm.ceil.",
    "@llvm.trunc.",
    "@llvm.rint.",
    "@llvm.nearbyint.",
    "@llvm.round.",
    "@llvm.roundeven.",
    "@llvm.copysign.",
    "@llvm.fshl.",
    "@llvm.fshr.",
    "@llvm.ctlz.",
    "@llvm.cttz.",
    "@llvm.ctpop.",
    "@llvm.bswap.",
    "@llvm.sadd.sat.",
    "@llvm.ssub.sat.",
    "@llvm.uadd.sat.",
    "@llvm.usub.sat.",
    "@llvm.assume",
    "@llvm.expect.",
)


def _is_non_capturing(callee: str, harmless: frozenset[str] = frozenset()) -> bool:
    return (
        callee in _NON_CAPTURING_CALLEES
        or callee in _PURE_LIBM_CALLEES
        or callee.startswith(_PURE_CALLEE_PREFIXES)
        or callee in harmless
    )


#: Type punctuation blanked before a parameter is split into words, so ``<4 x ptr>`` and ``{ ptr, i64 }``
#: both yield the word ``ptr``.
_TYPE_PUNCTUATION = str.maketrans({c: " " for c in "{}[]<>(),"})


def _parameter_list(header: str) -> str | None:
    """The text between the parentheses of a ``declare``/``define`` header's parameter list.

    Read with a depth count rather than to the first ``)``: an attribute such as ``captures(none)``
    closes a parenthesis INSIDE the list, and stopping there would drop the parameters after it -- the
    direction that would let a pointer parameter go unseen.
    """
    at = header.find("@")
    open_at = header.find("(", at) if at >= 0 else -1
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(header)):
        if header[i] == "(":
            depth += 1
        elif header[i] == ")":
            depth -= 1
            if depth == 0:
                return header[open_at + 1 : i]
    return None


def _top_level_items(params: str) -> list[str]:
    """``params`` split on the commas that separate parameters, not those inside an aggregate type."""
    out: list[str] = []
    depth, cur = 0, []
    for ch in params:
        depth += (ch in "({[<") - (ch in ")}]>")
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _harmless_callees(lines: list[str]) -> frozenset[str]:
    """Callees whose OWN signature in this module shows they cannot keep or free a pointer they are given.

    Derived from the module rather than listed, so a new math helper or vector intrinsic needs no entry:

    * a function with no pointer-carrying parameter never receives the pointer at all -- the portable
      transcendentals (``float @merlin_erff(float)``) and ``@rsqrtf`` are of this shape;
    * an ``@llvm.*`` intrinsic every one of whose pointer parameters LLVM itself marks ``captures(none)``
      (or the older ``nocapture``) reads or writes through the pointer and keeps nothing -- the masked
      vector loads and stores. Only intrinsics: on an ordinary function ``captures(none)`` does not
      exclude FREEING the pointer (``@free`` may be declared that way).
    """
    out: set[str] = set()
    for line in lines:
        if not line.startswith(("declare ", "define ")):
            continue
        at = line.find("@")
        end = at + 1
        while end < len(line) and line[end] in _NAME_CHARS:
            end += 1
        params = _parameter_list(line)
        if at < 0 or params is None:
            continue
        name = line[at:end]
        # Any parameter whose type mentions `ptr` -- a bare pointer or an aggregate holding one.
        carrying = [p for p in _top_level_items(params) if "ptr" in p.translate(_TYPE_PUNCTUATION).split()]
        if not carrying:
            out.add(name)
        elif name.startswith("@llvm.") and all("captures(none)" in p or "nocapture" in p.split() for p in carrying):
            out.add(name)
    return frozenset(out)


class ArenaBindError(RuntimeError):
    """The module could not be READ well enough to transform it safely.

    Distinct from an allocation being refused: a refusal is a normal, reported outcome that leaves
    that allocation on the heap. This is raised only when the parse itself is untrustworthy (an
    unterminated function, a branch whose successors cannot be resolved), because a partially
    understood CFG produces a dominance relation that is wrong in a direction nothing would notice.
    """


@dataclass
class _Block:
    label: str
    insts: list[int] = field(default_factory=list)  # indices into the function's line list
    succs: list[str] = field(default_factory=list)


@dataclass
class _Alloc:
    name: str  # raw SSA name, e.g. "%555"
    size: int  # the constant byte count the malloc asks for
    malloc_line: int  # index into the function's line list
    malloc_block: int  # block index
    malloc_pos: int  # position within the block
    free_line: int
    free_block: int
    free_pos: int


def _ssa_names(text: str) -> set[str]:
    """Every ``%name`` token in one instruction, read structurally (no pattern matching).

    A quoted name (``%"a b"``) is read to its closing quote, so an identifier containing a space or
    a comma is not split into two tokens that then match the wrong value.
    """
    out: set[str] = set()
    i, n = 0, len(text)
    while i < n:
        if text[i] != "%":
            i += 1
            continue
        j = i + 1
        if j < n and text[j] == '"':
            j += 1
            while j < n and text[j] != '"':
                j += 1
            j = min(j + 1, n)
        else:
            while j < n and text[j] in _NAME_CHARS:
                j += 1
        if j > i + 1:
            out.add(text[i:j])
        i = max(j, i + 1)
    return out


def _label_targets(terminator: str) -> list[str]:
    """Successor labels of a terminator, from its ``label %X`` operands.

    Covers ``br``/``switch``/``invoke`` uniformly: every successor in LLVM's textual form is spelled
    ``label %X``, so splitting on that token reads all of them without knowing the opcode. An opcode
    that names successors some other way would silently yield none, so the caller checks that a
    non-terminating block got at least one.
    """
    out: list[str] = []
    parts = terminator.split("label %")
    for chunk in parts[1:]:
        end = 0
        while end < len(chunk) and chunk[end] in _NAME_CHARS:
            end += 1
        if end:
            out.append("%" + chunk[:end])
    # A branch on a literal condition has ONE successor, and saying otherwise is not conservative
    # here -- it is the difference between a usable analysis and a vacuous one. The ownership-based
    # deallocation passes emit every dealloc behind `br i1 true, label %free, label %skip`
    # (measured: 112 of 112 on the deepjscc int8 build). Keeping the dead `%skip` edge makes every
    # free look optional, so no buffer is ever provably dead and the arena degenerates to one slot
    # per allocation -- 1.00x reuse, the same footprint as per-op malloc.
    if terminator.startswith("br i1 true,") and len(out) == 2:
        return out[:1]
    if terminator.startswith("br i1 false,") and len(out) == 2:
        return out[1:]
    return out


def _is_terminator(inst: str) -> bool:
    head = inst.split(" ", 1)[0]
    return head in ("br", "ret", "switch", "unreachable", "resume", "indirectbr", "callbr")


def _split_function(lines: list[str], symbol: str) -> tuple[int, int]:
    """``(first_body_line, closing_brace_line)`` for ``define ... @symbol(``."""
    needle = "@" + symbol + "("
    start = None
    for i, line in enumerate(lines):
        if line.startswith("define ") and needle in line and line.rstrip().endswith("{"):
            start = i
            break
    if start is None:
        raise ArenaBindError(f"no `define ... @{symbol}(` in the module")
    for j in range(start + 1, len(lines)):
        if lines[j] == "}":
            return start, j
    raise ArenaBindError(f"@{symbol} has no closing brace")


def _blocks_of(lines: list[str], start: int, end: int) -> list[_Block]:
    """Split a function body into basic blocks with their successors.

    A block label is an unindented line whose text before the first ``:`` is the label name; an
    entry block without one is named ``%0entry`` (not a legal LLVM name, so it can never collide
    with a real one).
    """
    blocks = [_Block(label="%0entry")]
    for i in range(start + 1, end):
        line = lines[i]
        if not line or line.startswith((" ", "\t")):
            if line.strip():
                blocks[-1].insts.append(i)
            continue
        head, sep, _ = line.partition(":")
        if not sep or not head:
            raise ArenaBindError(f"line {i} in the function body is neither indented nor a label: {line!r}")
        blocks.append(_Block(label="%" + head))
    # A body that opens with a label names its own entry block -- `mlir-translate` spells an outlined
    # OpenMP region that way (`omp.par.entry:`) -- so the unnamed one before it never existed.
    if not blocks[0].insts and len(blocks) > 1:
        blocks = blocks[1:]
    for blk in blocks:
        if not blk.insts:
            raise ArenaBindError(f"block {blk.label} is empty")
        term = lines[blk.insts[-1]].strip()
        if not _is_terminator(term):
            raise ArenaBindError(f"block {blk.label} does not end in a terminator: {term!r}")
        if term.split(" ", 1)[0] in ("ret", "unreachable", "resume"):
            continue
        blk.succs = _label_targets(term)
        if not blk.succs:
            raise ArenaBindError(
                f"block {blk.label} branches but no successor could be read from "
                f"{term!r}; a CFG with a missing edge yields a dominance relation "
                "that is wrong in the direction nothing checks"
            )
    return blocks


def _reverse_postorder(succs: list[list[int]], entry: int = 0) -> list[int]:
    order, seen, stack = [], set(), [(entry, iter(succs[entry]))]
    seen.add(entry)
    while stack:
        node, it = stack[-1]
        for nxt in it:
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(succs[nxt])))
                break
        else:
            order.append(stack.pop()[0])
    order.reverse()
    return order


def _reachable(succs: list[list[int]], entry: int = 0) -> set[int]:
    """Blocks the entry can reach. An allocation outside this set never runs, so binding it would
    reserve arena bytes nothing ever writes -- and, worse, make the window analysis reason about a
    program point that has no executions to reason about."""
    return set(_reverse_postorder(succs, entry))


def _cyclic_blocks(succs: list[list[int]]) -> set[int]:
    """Blocks that lie on a cycle: a self-edge, or a non-trivial strongly connected component.

    Iterative Tarjan -- the function bodies here run to ~1900 blocks, and a recursive walk on a deep
    chain would hit Python's recursion limit and report an exception where the answer is "this
    allocation is in a loop, refuse it".
    """
    n = len(succs)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    result: set[int] = set()
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work: list[tuple[int, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            for i in range(pi, len(succs[v])):
                w = succs[v][i]
                if w == v:
                    result.add(v)
                if index[w] == -1:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    result.update(comp)
    return result


def _parse_malloc(inst: str) -> tuple[str, int] | None:
    """``(raw_ssa_name, byte_size)`` for ``%x = call ptr @malloc(i64 N)``, else None.

    Only a CONSTANT size is accepted. A computed size belongs to a dynamically shaped buffer, which
    a static arena cannot seat; refusing it here is the same fail-closed rule ``arena_plan`` applies
    to a dynamic extent.
    """
    lhs, sep, rhs = inst.partition(" = ")
    if not sep or not lhs.startswith("%"):
        return None
    body = rhs.strip()
    marker = "@malloc(i64 "
    head, sep2, tail = body.partition(marker)
    if not sep2 or "call" not in head.split("@")[0]:
        return None
    arg, sep3, rest = tail.partition(")")
    if not sep3 or rest.strip():
        return None
    try:
        size = int(arg.strip())
    except ValueError:
        return None  # a non-literal size: dynamic, refuse
    return lhs.strip(), size


def _parse_free(inst: str) -> str | None:
    """The freed SSA name for ``call void @free(ptr %x)``, else None."""
    body = inst.strip()
    marker = "@free(ptr "
    head, sep, tail = body.partition(marker)
    if not sep or not head.startswith("call ") or "=" in head:
        return None
    arg, sep2, rest = tail.partition(")")
    if not sep2 or rest.strip():
        return None
    arg = arg.strip()
    return arg if arg.startswith("%") else None


def _window_blocks(a: _Alloc, succs: list[list[int]]) -> set[int]:
    """Blocks that can be entered between ``a``'s malloc and its free.

    A forward walk from the malloc's block that does NOT expand the free's block: once control
    reaches the free, the buffer is dead, so nothing past it belongs to the window. The malloc's
    block is never re-entered because a bound allocation is required to sit outside every CFG cycle.
    """
    if a.malloc_block == a.free_block and a.free_pos > a.malloc_pos:
        return set()
    seen: set[int] = set()
    stack = [s for s in succs[a.malloc_block]]
    while stack:
        blk = stack.pop()
        if blk in seen:
            continue
        seen.add(blk)
        if blk == a.free_block:
            continue  # dead from here on: do not walk past the free
        stack.extend(succs[blk])
    return seen


def _store_destination(inst: str) -> str | None:
    """The pointer a ``store`` writes THROUGH, or None if it cannot be read.

    ``store <ty> <value>, ptr <dest>[, align N]``. Splitting on ``", ptr "`` finds the destination
    without parsing the type: an aggregate type spells its members ``", ptr,"`` (comma, no space
    before the next member's name), so only the real destination operand matches.
    """
    body = inst.strip()
    if not body.startswith("store "):
        return None
    head, sep, tail = body.rpartition(", ptr ")
    if not sep:
        return None
    end = 0
    while end < len(tail) and tail[end] in _NAME_CHARS or (end == 0 and tail[:1] == "%"):
        end += 1
    name = tail[:end].strip().rstrip(",")
    return name if name.startswith("%") else None


#: Opcodes whose result is a comparison: an ``i1`` cannot carry a pointer, so the walk stops there.
_NO_CARRY_OPS = ("icmp ", "fcmp ")

#: Opcodes whose result may carry whatever their operand carries (pointer and integer arithmetic, and
#: the merges of control flow).
_CARRY_OPS = frozenset(
    "getelementptr bitcast addrspacecast ptrtoint inttoptr phi select freeze "
    "add sub and or xor urem udiv mul shl lshr".split()
)


def _operand_name(item: str) -> str:
    """The value of one ``<type> <value>`` operand: its last word."""
    words = item.split()
    return words[-1] if words else ""


def _handle_escapes(
    raw: str,
    free_line: int,
    lines: list[str],
    uses: dict[str, list[int]],
    defs: dict[str, int],
    harmless: frozenset[str],
) -> str | None:
    """Can the HANDLE ``raw`` -- the pointer ``free`` accepts -- reach anything but its one ``free``?

    Returns the refusal reason, or None when every value that may carry the handle stays inside the
    call and reaches only callees that cannot keep or free it.

    WHAT IS FOLLOWED, AND WHY ONLY THIS. Binding moves the buffer into the arena and deletes its
    ``free``. Byte REUSE is safe because the input program is correct under ``malloc``/``free``
    (module docstring, point 1): no access to the buffer's bytes happens outside its window, whoever
    performs it, so a callee that reads or writes through the DATA pointer -- the aligned address the
    lowering derives by ``ptrtoint`` arithmetic, which is what an OpenMP region or a matrix kernel is
    handed -- cannot observe the reuse. What binding CHANGES is the handle: deleting its ``free`` is
    only right if nothing else frees it, and an arena address handed to ``free`` corrupts the heap.
    So this follows the handle and everything that may contain it: the memref descriptor it is
    inserted into (field-sensitively -- extracting the ALIGNED field of a descriptor does not yield the
    handle), any local ``alloca`` the descriptor is spilled to and every value loaded back from one,
    and pointer or integer arithmetic on any of those.

    Refused when a carrier is stored anywhere but a local ``alloca`` (``descriptor_escapes``: it would
    outlive the call), is handed to a callee not shown harmless or to a second ``free``
    (``handle_escapes_to_callee``), or is returned or used by an instruction this reader does not know
    (``descriptor_escapes``).
    """
    taint: dict[str, frozenset] = {raw: frozenset({"*"})}  # value -> fields that may hold the handle
    local: set[str] = set()  # allocas holding a carrier, and pointers into them
    work = [raw]

    def is_alloca(name: str) -> bool:
        src = defs.get(name)
        return src is not None and lines[src].strip().partition(" = ")[2].startswith("alloca ")

    def mark(name: str, fields: frozenset, is_local: bool = False) -> None:
        old = taint.get(name)
        grew = old is None or not fields <= old
        if is_local and name not in local:
            local.add(name)
            grew = True  # re-read its uses: loads from it now carry
        if grew:
            taint[name] = (old or frozenset()) | fields
            work.append(name)

    while work:
        v = work.pop()
        for li in uses.get(v, []):
            if li == defs.get(v):
                continue
            inst = lines[li].strip()
            lhs, sep, rhs = inst.partition(" = ")
            body = rhs if sep else inst
            res = lhs.strip() if sep else ""
            if v == raw and (li == free_line or body.startswith("ptrtoint ")):
                continue  # its own free, and the alignment arithmetic that yields the DATA pointer
            if body.startswith(_NO_CARRY_OPS):
                continue
            if body.startswith("store "):
                dest = _store_destination(inst)
                if dest is None:
                    return "descriptor_escapes"
                if dest == v and v not in _ssa_names(inst.rpartition(", ptr ")[0]):
                    continue  # writing THROUGH the carrier; the value stored is something else
                if dest in local or is_alloca(dest):
                    mark(dest, frozenset({"*"}), is_local=True)
                    continue
                return "descriptor_escapes"
            if body.startswith("load "):
                if v in local and res:
                    mark(res, frozenset({"*"}))
                continue  # a load through a data pointer reads data, not the handle
            if body.startswith(("call ", "tail call ", "musttail call ", "notail call ")):
                callee = "@" + (_call_target(inst) or "<indirect>")
                if callee == "@free" or not _is_non_capturing(callee, harmless):
                    return "handle_escapes_to_callee"
                continue
            if body.startswith("insertvalue "):
                items = _top_level_items(body[len("insertvalue ") :])
                if len(items) < 3 or not res:
                    return "descriptor_escapes"
                agg, elt, field = _operand_name(items[0]), _operand_name(items[1]), items[2].strip()
                held = taint.get(agg, frozenset())
                fields = {f for f in held if f != field or f == "*"}
                if elt in taint:
                    fields.add(field)
                if fields:
                    mark(res, frozenset(fields), is_local=agg in local)
                continue
            if body.startswith("extractvalue "):
                items = _top_level_items(body[len("extractvalue ") :])
                if len(items) < 2 or not res:
                    return "descriptor_escapes"
                held = taint.get(_operand_name(items[0]), frozenset())
                if "*" in held or items[1].strip() in held:
                    mark(res, frozenset({"*"}))
                continue
            if res and body.split(" ", 1)[0] in _CARRY_OPS:
                mark(res, frozenset({"*"}), is_local=v in local and body.startswith("getelementptr "))
                continue
            return "descriptor_escapes"  # returned, or an instruction this reader does not know
    return None


@dataclass
class BindReport:
    arena_bytes: int
    bound: int
    refused: int
    refusals: dict[str, int]
    naive_total_bytes: int
    reuse_factor: float
    symbol: str
    #: The most bytes the bound buffers hold at any one program point: a LOWER bound on any arena that
    #: seats them. ``arena_bytes`` minus this is what the first-fit placement lost to fragmentation.
    peak_live_bytes: int = 0
    #: Bytes the refused constant-size allocations ask for, once per site. They stay on the heap.
    refused_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arena_bytes": self.arena_bytes,
            "bound": self.bound,
            "refused": self.refused,
            "refusals": dict(self.refusals),
            "naive_total_bytes": self.naive_total_bytes,
            "reuse_factor": self.reuse_factor,
            "symbol": self.symbol,
            "peak_live_bytes": self.peak_live_bytes,
            "refused_bytes": self.refused_bytes,
        }


def bind_arena(ll_text: str, *, symbol: str = ARENA_SYMBOL, entry: str = "forward") -> tuple[str, BindReport]:
    """Rewrite the provable heap allocations of ``@entry`` into one arena. Returns ``(ll, report)``.

    With no bindable allocation the text is returned UNCHANGED (not merely equivalent), so a module
    the analysis cannot help is byte-identical to the input and no arena global is emitted.
    """
    lines = ll_text.split("\n")
    start, end = _split_function(lines, entry)
    define = lines[start]
    if not define.startswith("define void "):
        raise ArenaBindError(
            f"@{entry} does not return void ({define.split('@')[0].strip()!r}); a returned value "
            "could carry an arena pointer out of the call, past the point the plan says those bytes "
            "belong to another buffer"
        )

    blocks = _blocks_of(lines, start, end)
    by_label = {b.label: i for i, b in enumerate(blocks)}
    succs: list[list[int]] = []
    for b in blocks:
        resolved = []
        for lab in b.succs:
            if lab not in by_label:
                raise ArenaBindError(f"block {b.label} branches to unknown label {lab}")
            resolved.append(by_label[lab])
        succs.append(resolved)
    # A recursive entry would give two activations one arena. Nothing here is recursive today; the
    # check is cheap and the failure it prevents is silent.
    for b in blocks:
        for li in b.insts:
            if "@" + entry + "(" in lines[li] and "call" in lines[li]:
                raise ArenaBindError(f"@{entry} calls itself; one arena cannot serve two activations")

    line_block: dict[int, tuple[int, int]] = {}
    for bi, b in enumerate(blocks):
        for pos, li in enumerate(b.insts):
            line_block[li] = (bi, pos)

    # ---- collect candidates -----------------------------------------------------------------
    mallocs: dict[str, tuple[int, int]] = {}  # name -> (line, size)
    frees: dict[str, list[int]] = {}
    uses: dict[str, list[int]] = {}
    for bi, b in enumerate(blocks):
        for li in b.insts:
            inst = lines[li].strip()
            m = _parse_malloc(inst)
            if m is not None:
                if m[0] in mallocs:
                    raise ArenaBindError(f"SSA name {m[0]} defined by two mallocs")
                mallocs[m[0]] = (li, m[1])
                continue
            f = _parse_free(inst)
            if f is not None:
                frees.setdefault(f, []).append(li)
            for nm in _ssa_names(inst):
                uses.setdefault(nm, []).append(li)

    refusals: dict[str, int] = {}

    def refuse(reason: str) -> None:
        refusals[reason] = refusals.get(reason, 0) + 1

    # Where the HANDLE (the pointer malloc returned, the one thing `free` accepts) can travel is decided
    # per allocation by following it -- see `_handle_escapes`. A callee is trusted with a value that
    # carries it only when its kind or its own signature shows it cannot keep or free it.
    harmless = _harmless_callees(lines)
    defs: dict[str, int] = {}
    for b in blocks:
        for li in b.insts:
            lhs, sep, _ = lines[li].strip().partition(" = ")
            if sep and lhs.startswith("%"):
                defs[lhs.strip()] = li

    cyclic = _cyclic_blocks(succs)
    allocs: list[_Alloc] = []
    for name, (mline, size) in sorted(mallocs.items(), key=lambda kv: kv[1][0]):
        fl = frees.get(name, [])
        if len(fl) != 1:
            refuse("no_single_free" if not fl else "multiple_frees")
            continue
        fline = fl[0]
        bad = [
            li
            for li in uses.get(name, [])
            if li != fline
            and not lines[li].strip().partition(" = ")[2].startswith(tuple(op + " " for op in _ALLOWED_RAW_USES))
        ]
        if bad:
            refuse("raw_pointer_escapes")
            continue
        why = _handle_escapes(name, fline, lines, uses, defs, harmless)
        if why is not None:
            refuse(why)
            continue
        mb, mp = line_block[mline]
        fb, fp = line_block[fline]
        if mb in cyclic or fb in cyclic:
            refuse("in_loop")
            continue
        allocs.append(
            _Alloc(
                name=name,
                size=size,
                malloc_line=mline,
                malloc_block=mb,
                malloc_pos=mp,
                free_line=fline,
                free_block=fb,
                free_pos=fp,
            )
        )

    if not allocs:
        return ll_text, BindReport(
            arena_bytes=0,
            bound=0,
            refused=sum(refusals.values()),
            refusals=refusals,
            naive_total_bytes=0,
            reuse_factor=0.0,
            symbol=symbol,
        )

    reachable = _reachable(succs)
    kept = [a for a in allocs if a.malloc_block in reachable and a.free_block in reachable]
    for _ in range(len(allocs) - len(kept)):
        refuse("unreachable_block")
    allocs = kept
    if not allocs:
        return ll_text, BindReport(
            arena_bytes=0,
            bound=0,
            refused=sum(refusals.values()),
            refusals=refusals,
            naive_total_bytes=0,
            reuse_factor=0.0,
            symbol=symbol,
        )

    window = {a.name: _window_blocks(a, succs) for a in allocs}

    def in_window(a: _Alloc, blocks_in: set[int], blk: int, pos: int) -> bool:
        """Is program point ``(blk, pos)`` inside ``a``'s ``[malloc, free)`` window?"""
        if blk == a.malloc_block and blk == a.free_block:
            return a.malloc_pos < pos < a.free_pos
        if blk == a.malloc_block:
            return pos > a.malloc_pos
        if blk == a.free_block:
            return pos < a.free_pos
        return blk in blocks_in

    # ---- interference ------------------------------------------------------------------------
    # Two windows can intersect only if one allocation happens INSIDE the other's window, and each
    # site runs at most once (the loop check above), so the whole relation is two reachability
    # questions. This is stated as reachability rather than dominance on purpose: dominance asks
    # "does every path go through the free", which a genuine diamond answers "no" even when the two
    # buffers are on mutually exclusive arms and can never coexist.
    conflicts: dict[str, set[str]] = {a.name: set() for a in allocs}
    for i, a in enumerate(allocs):
        for b in allocs[i + 1 :]:
            if in_window(a, window[a.name], b.malloc_block, b.malloc_pos) or in_window(
                b, window[b.name], a.malloc_block, a.malloc_pos
            ):
                conflicts[a.name].add(b.name)
                conflicts[b.name].add(a.name)

    sizes = {a.name: _align_up(a.size) for a in allocs}
    offsets, arena_bytes = _best_packing(allocs, sizes, conflicts)

    # ---- the property the pass exists for, re-checked on the placement it is about to emit ----
    _assert_no_conflicting_overlap(offsets, sizes, conflicts)

    # ---- rewrite ------------------------------------------------------------------------------
    out = list(lines)
    for a in allocs:
        out[a.malloc_line] = f"  {a.name} = getelementptr inbounds i8, ptr @{symbol}, i64 {offsets[a.name]}"
        out[a.free_line] = None  # type: ignore[call-overload]
    out = [ln for ln in out if ln is not None]
    first_define = next(i for i, ln in enumerate(out) if ln.startswith("define "))
    out.insert(first_define, "")
    out.insert(first_define, f"@{symbol} = internal global [{arena_bytes} x i8] zeroinitializer, align {ARENA_ALIGN}")

    naive = sum(sizes.values())
    # What is live at each allocation: the buffer itself plus every conflicting one whose window holds
    # this malloc. The heaviest such set is the peak; an arena cannot be smaller than it.
    by_name = {a.name: a for a in allocs}
    peak = max(
        sizes[b.name]
        + sum(sizes[o] for o in conflicts[b.name] if in_window(by_name[o], window[o], b.malloc_block, b.malloc_pos))
        for b in allocs
    )
    bound_names = {a.name for a in allocs}
    report = BindReport(
        arena_bytes=arena_bytes,
        bound=len(allocs),
        refused=sum(refusals.values()),
        refusals=refusals,
        naive_total_bytes=naive,
        reuse_factor=round(naive / arena_bytes, 2) if arena_bytes else 0.0,
        symbol=symbol,
        peak_live_bytes=peak,
        refused_bytes=sum(size for name, (_line, size) in mallocs.items() if name not in bound_names),
    )
    return "\n".join(out), report


def _best_packing(
    allocs: list[_Alloc], sizes: dict[str, int], conflicts: dict[str, set[str]]
) -> tuple[dict[str, int], int]:
    """First-fit placement under several orders; the smallest arena wins.

    No single greedy order is best for every program, and the difference is the board's margin: on a
    2-layer TinyLlama section at 512 tokens, largest-first left the arena 11.5 MiB over its peak, the
    image 2.2 MiB over Kodiak's DRAM. Every order is placed by the same `pack_disjoint` against the same
    conflict relation, so each is as sound as the other; only the size differs. Ties keep the
    largest-first order the binder always used, so a program that gains nothing is placed as before.
    Program position (the text line of the malloc and of its free) serves only as an ORDER key here --
    it says nothing about liveness, which the conflict relation alone decides.
    """
    span = {a.name: max(1, a.free_line - a.malloc_line) for a in allocs}
    keys = (
        lambda a: (-sizes[a.name], a.name),  # largest first
        lambda a: (a.malloc_line, a.name),  # in allocation order
        lambda a: (-span[a.name], -sizes[a.name], a.name),  # longest-lived first
        lambda a: (-sizes[a.name] * span[a.name], a.name),  # largest size x lifetime first
    )
    best: tuple[dict[str, int], int] | None = None
    for key in keys:
        order = [(a.name, sizes[a.name]) for a in sorted(allocs, key=key)]
        placed = pack_disjoint(order, conflicts)
        if best is None or placed[1] < best[1]:
            best = placed
    assert best is not None
    return best


def _assert_no_conflicting_overlap(
    offsets: dict[str, int], sizes: dict[str, int], conflicts: dict[str, set[str]]
) -> None:
    """Refuse to emit a placement in which two conflicting buffers share a byte.

    This is the one invariant whose violation is invisible: the module still verifies, still links,
    still runs, and returns numbers that are wrong only where the two buffers happened to be live at
    once. Checking it against the offsets actually about to be written -- rather than trusting the
    placement routine -- costs O(conflicting pairs) and is the difference between a bug that is
    caught here and one that is caught by a cosine on the board, if at all.
    """
    for a, others in conflicts.items():
        oa, sa = offsets[a], sizes[a]
        for b in others:
            ob, sb = offsets[b], sizes[b]
            if oa < ob + sb and ob < oa + sa:
                raise ArenaBindError(
                    f"placement seats conflicting buffers {a} [{oa},{oa + sa}) and {b} "
                    f"[{ob},{ob + sb}) on overlapping bytes"
                )


# ---- How many bytes one call of a program allocates, read off its emitted LLVM IR ---------------


@dataclass(frozen=True)
class HeapDemand:
    """What every ``malloc`` the program's host code makes in one run adds up to.

    For an allocator that never frees within a run (the bare-metal bump arena), ``bytes`` IS the arena
    the run needs, less per-allocation alignment. Each site counts once per call of its function, and
    a function counts once per call site of each caller, where a function nothing in the modules
    calls (``forward``, a group body the C dispatch calls) is called once. ``unbounded`` lists the
    sites the total cannot include, by reason: a computed size, a site in a loop, a function called
    from a loop or recursively. A caller decides what an unbounded site means; it is never guessed.
    """

    bytes: int
    allocations: int
    largest: int
    unbounded: dict[str, int]
    functions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "allocations": self.allocations,
            "largest": self.largest,
            "unbounded": dict(self.unbounded),
            "functions": self.functions,
        }


def _defines(lines: list[str]) -> list[tuple[str, int, int]]:
    """``(name, define_line, closing_line)`` of every function defined in a module."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("define ") and line.rstrip().endswith("{"):
            at = line.find("@")
            name = line[at + 1 :].partition("(")[0].strip('"') if at >= 0 else ""
            j = i + 1
            while j < len(lines) and lines[j] != "}":
                j += 1
            out.append((name, i, j))
            i = j
        i += 1
    return out


def _call_target(inst: str) -> str | None:
    """The ``@symbol`` a direct call names (without the ``@``), else None."""
    body = inst.partition(" = ")[2] or inst
    if not body.startswith(("call ", "tail call ", "musttail call ", "notail call ", "invoke ")):
        return None
    at = body.find("@")
    if at < 0:
        return None
    end = at + 1
    while end < len(body) and body[end] in _NAME_CHARS:
        end += 1
    return body[at + 1 : end]


def _malloc_size(inst: str) -> int | None:
    """The literal byte count a ``malloc`` call asks for, in any spelling (``i64 N``, ``i64 noundef N``,
    a trailing attribute group): the size is the argument's LAST token. ``None`` for a computed size."""
    args, close, _ = inst.partition("@malloc(")[2].partition(")")
    tokens = args.split()
    if not close or len(tokens) < 2 or not tokens[0].startswith("i") or not tokens[-1].isdigit():
        return None
    return int(tokens[-1])


def heap_demand(ll_texts: list[str]) -> HeapDemand:
    """The bytes one run of the program made of ``ll_texts`` (its LLVM modules) allocates."""
    sites: dict[str, list[tuple[int | None, bool]]] = {}  # fn -> [(size or None, in a cycle)]
    calls: dict[str, list[tuple[str, bool]]] = {}  # callee -> [(caller, in a cycle)]
    for text in ll_texts:
        lines = text.split("\n")
        for name, start, end in _defines(lines):
            blocks = _blocks_of(lines, start, end)
            by_label = {b.label: k for k, b in enumerate(blocks)}
            cyclic = _cyclic_blocks([[by_label[s] for s in b.succs] for b in blocks])
            mine = sites.setdefault(name, [])
            for k, b in enumerate(blocks):
                for li in b.insts:
                    inst = lines[li].strip()
                    target = _call_target(inst)
                    if target is None:
                        continue
                    if target == "malloc":
                        mine.append((_malloc_size(inst), k in cyclic))
                    else:
                        calls.setdefault(target, []).append((name, k in cyclic))

    counts: dict[str, int | None] = {}

    def count(fn: str, path: frozenset[str]) -> int | None:
        if fn in counts:
            return counts[fn]
        callers = [(c, loop) for c, loop in calls.get(fn, []) if c in sites]
        if not callers:
            n: int | None = 1
        elif any(loop for _c, loop in callers) or any(c in path for c, _loop in callers):
            n = None
        else:
            n = 0
            for caller, _loop in callers:
                m = count(caller, path | {fn})
                if m is None:
                    n = None
                    break
                n += m
        counts[fn] = n
        return n

    total = allocations = largest = 0
    unbounded: dict[str, int] = {}
    for fn, found in sites.items():
        n = count(fn, frozenset()) if found else 1
        for size, loop in found:
            # A loop multiplies whatever the site asks for, so it outranks a computed size as the reason.
            reason = "in_loop" if loop else "called_in_loop" if n is None else "dynamic_size" if size is None else None
            if reason:
                unbounded[reason] = unbounded.get(reason, 0) + 1
                continue
            total += size * n
            allocations += n
            largest = max(largest, size)
    return HeapDemand(total, allocations, largest, unbounded, len(sites))

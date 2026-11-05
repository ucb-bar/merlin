"""``merlin-arena-plan`` — compile-time static MEMORY PLAN for a dispatch program.

The current whole-model runtime materializes every intermediate with its own
``tensor.empty`` -> ``memref.alloc`` (~4391 mallocs on smolvla), the dominant share of the
dispatch overhead and the reason large models hit the RAM wall. This pass replaces that with
ONE arena: it computes each intermediate buffer's live range over the (topo-ordered) dispatch
DAG and greedy-colors non-overlapping buffers onto shared byte offsets, so the runtime binds
``arena_base() + offset`` with ZERO per-op allocation and bounded, reused memory.

Pure analysis over :class:`~.dispatch_program.DispatchProgram` (no IR mutation). What is wired, and
what is not, precisely:

* :func:`pack_disjoint` -- the placement core -- IS on the measured path. The LLVM-IR binder
  (:mod:`merlin.llvmlower.arena_bind`, reached from ``lower_model(static_arena=True)``) uses it to
  seat the allocations the whole-model build actually emits.
* :func:`plan_arena` itself is still analysis only. It plans a ``DispatchProgram``, and the K1 build
  lowers the monolithic ``@forward`` instead; ``merlin_program.c``, the replay engine two other
  docstrings name as its consumer, does not exist in the tree. See
  ``docs/design/static_arena_wiring.md``.

The two planners also measure DIFFERENT liveness, and the difference is the whole gap between their
numbers. :func:`plan_arena` ends a buffer's range at its last USE; the emitted binary frees every
intermediate at the END of ``@forward`` (measured: all 112 deallocations of the deepjscc int8 build
sit after the last compute), so the binder -- which may only trust the program's own ``free`` -- gets
1.00x reuse where this planner reports 5.77x. The reuse figures here are what last-use deallocation
placement would be worth, not what the binary does.

Every branch that cannot size a buffer raises :class:`ArenaPlanError` rather than substituting a
plausible number, and offsets are aligned to :data:`ARENA_ALIGN` (derived from the allocator the
runtime actually uses), so a caller gets an honest refusal instead of a short arena.

Args/weights are bound from the model arg table; results are bound to the caller's output buffer; both
stay OUT of the arena.
Algorithm: linear-scan / greedy first-fit interval allocation (the classic offline planner, à la
TFLite-Micro's memory planner) — non-optimal but bounded and reuse-maximizing vs per-op malloc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dispatch_program import DispatchProgram

# bytes per element for the dtypes the models use (MLIR element-type spellings).
_DTYPE_BYTES = {
    "f64": 8,
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "f8E4M3FN": 1,
    "fp8_e4m3": 1,
    "f8E5M2": 1,
    "i64": 8,
    "i32": 4,
    "i16": 2,
    "i8": 1,
    "i1": 1,
    "index": 8,
    "ui8": 1,
    "ui32": 4,
}

#: Alignment every offset in the plan must satisfy, DERIVED from the allocator the runtime actually
#: uses: ``runtime/baremetal/spike/merlin_malloc.c`` bumps with ``bump(n, 64)`` for ``malloc``/``calloc``
#: and clamps ``aligned_alloc`` up to 64, so every buffer the per-op path hands out today is 64-byte
#: aligned. A byte-tight arena plan silently WEAKENS that: an RVV whole-register load/store on an
#: under-aligned base is the class of bug that shows up as wrong numbers on silicon and nowhere in
#: simulation. Aligning costs padding and nothing else.
ARENA_ALIGN = 64


class ArenaPlanError(ValueError):
    """A buffer the plan cannot size, or a program it cannot represent.

    Every branch that raises this used to FAIL OPEN with a plausible number -- a dynamic dimension
    sized as 1 element, an unrecognised dtype sized as f32. Both produce a plan that looks complete and
    an arena that is too small, and the failure surfaces at run time as heap corruption on the board,
    arbitrarily far from the cause. Nothing consumes ``plan_arena`` yet, so raising here costs nothing
    today and is the only reading that will still be safe once something does.
    """


def _elem_bytes(dtype: str) -> int:
    """Width of one element, or raise. Never a default.

    The retired behavior returned 4 for anything unrecognised. That is not a tolerant fallback, it is a
    wrong answer for every dtype that is not 4 bytes wide -- and it silently halves the arena for an
    f64/i64 buffer.
    """
    d = (dtype or "").strip()
    if d in _DTYPE_BYTES:
        return _DTYPE_BYTES[d]
    # A prefixed spelling (e.g. "si8", "!f32") still resolves by SUFFIX, longest key first so "bf16"
    # is not read as "f16" and "ui32" not as "i32". A wrapped spelling ("vector<4xf32>") does NOT
    # resolve -- it ends in ">" -- and that is correct: its element count is part of the type, so
    # treating it as one scalar would under-size the buffer by that factor.
    for k in sorted(_DTYPE_BYTES, key=len, reverse=True):
        if d.endswith(k):
            return _DTYPE_BYTES[k]
    raise ArenaPlanError(
        f"cannot size element type {dtype!r}: not one of {sorted(_DTYPE_BYTES)}. Sizing it as f32 (the "
        "retired default) would under-size the arena for every wider type and over-size it for every "
        "narrower one, with no way to notice."
    )


def _buf_bytes(shape: list[int], dtype: str) -> int:
    """Byte size of a statically-shaped buffer, or raise.

    A DYNAMIC dimension arrives as MLIR's ``ShapedType::kDynamic`` sentinel (INT64_MIN) or a negative
    extent. The retired behavior mapped any non-positive dim to 1, so a `tensor<?x768xf32>` was planned
    as 768 elements: the plan reports success and the arena is short by whatever the real extent turns
    out to be. A static planner genuinely cannot size a dynamic buffer -- so say so.
    """
    n = 1
    for s in shape:
        try:
            d = int(s)
        except (TypeError, ValueError) as exc:
            raise ArenaPlanError(f"non-integer extent {s!r} in shape {shape!r}") from exc
        if d < 0:
            raise ArenaPlanError(
                f"dynamic extent {d} in shape {shape!r}: a static arena plan cannot size a buffer "
                "whose extent is only known at run time. Sizing it as 1 element (the retired default) "
                "produces a plan that looks complete and an arena that is too small."
            )
        if d == 0:
            return ARENA_ALIGN  # a legal empty buffer still needs a distinct, aligned address
        n *= d
    return max(n * _elem_bytes(dtype), 1)


def _align_up(n: int, a: int = ARENA_ALIGN) -> int:
    return ((int(n) + a - 1) // a) * a


def pack_disjoint(order: list[tuple[str, int]], conflicts: dict[str, set[str]]) -> tuple[dict[str, int], int]:
    """Greedy first-fit placement of ``(id, aligned_size)`` blocks into one arena.

    ``conflicts[x]`` is the set of ids that MAY be live at the same time as ``x`` and therefore may
    not share a byte with it. The relation must be symmetric; it is read symmetrically here so a
    caller that fills only one direction still gets a correct placement.

    This is the shared core of both planners. The DAG planner (:func:`plan_arena`) derives conflicts
    from live-range intervals over a topologically ordered dispatch program; the LLVM-IR binder
    (:mod:`merlin.llvmlower.arena_bind`) derives them from CFG dominance, where "intervals" do not
    exist because the order of two blocks in the text says nothing about the order they run in.
    Sharing the placement means the property that matters -- two conflicting blocks never overlap --
    is established once, and is checked once, for both.

    Blocks are placed in the given ``order``; each takes the LOWEST offset that overlaps no already
    placed block it conflicts with, rounded up to :data:`ARENA_ALIGN`. Returns
    ``(offsets, arena_bytes)``.
    """
    offsets: dict[str, int] = {}
    total = 0
    for bid, size in order:
        if size <= 0:
            raise ArenaPlanError(f"block {bid!r} has non-positive size {size}")
        against = conflicts.get(bid, set())
        # every already-placed block this one may not share bytes with
        busy = sorted(
            (offsets[o], offsets[o] + sz)
            for o, sz in ((o, s) for o, s in order if o in offsets)
            if o in against or bid in conflicts.get(o, set())
        )
        off = 0
        for lo, hi in busy:
            if off + size <= lo:
                break  # fits in the gap before this block
            if hi > off:
                off = _align_up(hi)  # push past it, to the next aligned boundary
        offsets[bid] = off
        total = max(total, off + size)
    return offsets, _align_up(total)


@dataclass
class MemoryPlan:
    arena_bytes: int  # total arena the runtime must provide
    offsets: dict[str, int]  # intermediate buffer id -> byte offset in arena
    sizes: dict[str, int]  # buffer id -> byte size
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arena_bytes": self.arena_bytes,
            "offsets": dict(self.offsets),
            "sizes": dict(self.sizes),
            "stats": dict(self.stats),
        }


def _assert_reads_are_complete(prog: DispatchProgram) -> None:
    """Refuse a program whose nodes do not record everything they READ.

    Liveness here is computed from ``node.inputs`` alone, and that is only the truth when ``inputs``
    is the node's complete read set. It is NOT complete for a region-carrying op built from
    ``op.operands``: an ``scf.for``'s operands are lb/ub/step/iter_args, so a tensor its body reads out
    of the enclosing scope appears nowhere. Measured on a three-dispatch driver whose loop body reads
    an earlier result: that buffer's last use was recorded as node 3 instead of node 6, and the planner
    then seated the NEXT buffer at the same offset -- two live buffers sharing bytes, no error, wrong
    numbers only when the sizes collide.

    :func:`~.dispatch_program.build_dispatch_program` now folds the region captures into ``inputs`` and
    records them in ``Node.captures``. ``captures is None`` means nobody established them (an older
    writer, a hand-built or deserialized program), and a static planner genuinely cannot recover them
    from the flat program -- the captured value IS defined earlier, it is simply never listed, so
    nothing about the DAG looks wrong. Guessing there is the exact failure mode this module refuses
    everywhere else, so it refuses here too.
    """
    for i, node in enumerate(prog.nodes):
        regions = int(getattr(node, "regions", 0) or 0)
        captures = getattr(node, "captures", None)
        if regions and captures is None:
            raise ArenaPlanError(
                f"node {i} ({node.op}) carries {regions} region(s) but no capture set was computed "
                "(Node.captures is None). Its region body may read buffers that node.inputs does not "
                "list, which understates their live ranges and lets the plan hand those bytes to "
                "another buffer while they are still being read. Rebuild the program with "
                "build_dispatch_program (which walks the regions), or drop the region-carrying node."
            )
        if captures:
            missing = [b for b in captures if b not in node.inputs]
            if missing:
                raise ArenaPlanError(
                    f"node {i} ({node.op}) records region captures {missing} that are absent from its "
                    "inputs. Liveness is computed from inputs, so a capture kept out of them is a read "
                    "the plan cannot see."
                )


def _live_ranges(prog: DispatchProgram) -> tuple[dict[str, int], dict[str, int]]:
    """first-def node index and last-use node index per buffer.

    A buffer is live across [def, last_use]. Args/consts are defined "before" node 0 (def=-1).
    Result buffers (and anything still consumed at the end) get last_use = len(nodes) so they
    survive the whole program (they are bound to the output buffer, not the arena, but we keep
    the conservative range so arena packing never reuses bytes that a result still needs).
    """
    _assert_reads_are_complete(prog)
    n = len(prog.nodes)
    first_def: dict[str, int] = {}
    last_use: dict[str, int] = {}
    for b in prog.buffers.values():
        if b.kind in ("arg", "const"):
            first_def[b.id] = -1
    for i, node in enumerate(prog.nodes):
        for bid in node.inputs:
            last_use[bid] = i
        for bid in node.outputs:
            first_def.setdefault(bid, i)
    for bid in prog.results:
        last_use[bid] = n  # lives to the end
    # a produced-but-never-used buffer dies immediately after its def
    for bid, d in first_def.items():
        last_use.setdefault(bid, d)
    return first_def, last_use


def plan_arena(prog: DispatchProgram) -> MemoryPlan:
    """Greedy first-fit static allocation of the program's INTERMEDIATE buffers into one arena.

    Excludes args/weights (bound from the arg table), consts (emitted as constants), and result
    buffers (bound to the caller's output). Returns a :class:`MemoryPlan` with per-buffer offsets
    and the total arena size.
    """
    first_def, last_use = _live_ranges(prog)
    results = set(prog.results)
    # buffers that need arena space: live intermediates that are not program results.
    arena_bufs = [b for b in prog.buffers.values() if b.kind == "intermediate" and b.id not in results]
    # Sizes are ALIGNED UP: the offsets are what the runtime adds to the arena base, so padding each
    # block is what keeps every buffer 64-byte aligned no matter what precedes it (see ARENA_ALIGN).
    sizes = {b.id: _align_up(_buf_bytes(b.shape, b.dtype)) for b in prog.buffers.values()}

    # process in definition order; tie-break larger-first so big buffers seat low and stable.
    order = sorted(arena_bufs, key=lambda b: (first_def.get(b.id, 0), -sizes[b.id]))

    # Two buffers conflict when their [def, last_use] intervals meet. Handing the relation to
    # `pack_disjoint` rather than open-coding the sweep is what makes this planner and the LLVM-IR
    # binder share ONE placement routine -- and one place where "conflicting blocks never overlap"
    # is established.
    ranges = {b.id: (first_def.get(b.id, 0), last_use.get(b.id, 0)) for b in arena_bufs}
    conflicts: dict[str, set[str]] = {b.id: set() for b in arena_bufs}
    for i, b in enumerate(order):
        db, ub = ranges[b.id]
        for c in order[i + 1 :]:
            dc, uc = ranges[c.id]
            if db <= uc and dc <= ub:
                conflicts[b.id].add(c.id)
                conflicts[c.id].add(b.id)
    offsets, arena_bytes = pack_disjoint([(b.id, sizes[b.id]) for b in order], conflicts)
    for bid, off in offsets.items():
        assert off % ARENA_ALIGN == 0, (bid, off)  # the invariant the C replay depends on
    naive_total = sum(sizes[b.id] for b in arena_bufs)
    stats = {
        "n_intermediate_buffers": len(arena_bufs),
        "naive_total_bytes": naive_total,  # what per-op malloc would use (peak-naive)
        "arena_bytes": arena_bytes,  # what the static plan uses
        "reuse_factor": round(naive_total / arena_bytes, 2) if arena_bytes else 0.0,
        "n_result_buffers": len(results),
        "align": ARENA_ALIGN,
    }
    return MemoryPlan(arena_bytes=arena_bytes, offsets=offsets, sizes=sizes, stats=stats)


@dataclass(frozen=True)
class ArenaRequirements:
    """Explicit graph-level storage and completion requirements.

    Events are ordered application events, not guessed accelerator cycles.
    Aliases are caller-provided physical views; this planner checks bounds and
    lifetime retention but does not prove the source view's numerical meaning.
    Persistent buffers remain reserved throughout one invocation. The caller
    separately owns initialization and state retention across invocations.
    """

    capacity_bytes: int
    launch_events: tuple[int, ...]
    completion_events: tuple[int, ...]
    alignment: int = ARENA_ALIGN
    aliases: tuple[tuple[str, str, int], ...] = ()
    persistent: tuple[str, ...] = ()
    reserved: tuple[tuple[int, int], ...] = ()
    dispatch_digest: str = ""
    physical_sizes: tuple[tuple[str, int], ...] = ()


def _bounded_arena_inputs(prog: DispatchProgram, requirements: ArenaRequirements):
    """Recover storage and lifetime obligations from the current graph."""
    from merlin.semantic_compiler.allocate import Reservation, StorageBank
    from merlin.semantic_compiler.buffer_allocation import Buffer

    from .dispatch_program import verify_program
    from .global_plan_emission import dispatch_digest

    if requirements.dispatch_digest != dispatch_digest(prog):
        raise ArenaPlanError("memory requirements do not match the emitted dispatch graph")
    problems = verify_program(prog)
    if problems:
        raise ArenaPlanError("invalid dispatch graph: " + "; ".join(problems))
    _assert_reads_are_complete(prog)
    if (
        type(requirements.capacity_bytes) is not int
        or requirements.capacity_bytes <= 0
        or type(requirements.alignment) is not int
        or requirements.alignment <= 0
    ):
        raise ArenaPlanError("arena needs positive capacity and alignment")
    launches, completions = requirements.launch_events, requirements.completion_events
    if (
        len(launches) != len(prog.nodes)
        or len(completions) != len(prog.nodes)
        or any(type(n) is not int or n < 0 for n in (*launches, *completions))
        or any(a > b for a, b in zip(launches, launches[1:]))
        or any(start > end for start, end in zip(launches, completions))
    ):
        raise ArenaPlanError("each dispatch needs valid ordered launch and completion events")
    sizes = {name: _buf_bytes(buffer.shape, buffer.dtype) for name, buffer in prog.buffers.items()}
    physical_sizes = dict(requirements.physical_sizes)
    if (
        len(physical_sizes) != len(requirements.physical_sizes)
        or not set(physical_sizes) <= set(sizes)
        or any(type(size) is not int or size < sizes[name] for name, size in physical_sizes.items())
    ):
        raise ArenaPlanError("physical sizes must name buffers and cover their logical storage")
    sizes.update(physical_sizes)
    external = {
        name for name, buffer in prog.buffers.items() if buffer.kind in ("arg", "const") or name in prog.results
    }
    owned = set(prog.buffers) - external
    aliases = {name: (parent, offset) for name, parent, offset in requirements.aliases}
    if (
        len(aliases) != len(requirements.aliases)
        or not set(aliases) <= owned
        or any(parent not in owned for parent, _ in aliases.values())
        or not set(requirements.persistent) <= owned
    ):
        raise ArenaPlanError("arena aliases and persistent buffers must name owned intermediates")
    first, last, ready, producers = {}, {}, {}, {}
    for index, node in enumerate(prog.nodes):
        for name in node.inputs:
            if name in ready and ready[name] > launches[index]:
                raise ArenaPlanError("consumer launches before its input producer completes")
            last[name] = max(last.get(name, 0), completions[index])
        for name in node.outputs:
            if name in first or prog.buffers[name].kind == "arg":
                raise ArenaPlanError("dispatch buffer has more than one definition")
            first[name], ready[name] = launches[index], completions[index]
            producers[name] = node
            last[name] = max(last.get(name, 0), completions[index])
    for name, (parent, _) in aliases.items():
        producer = producers.get(name)
        if producer is None or producer.kind != "view" or parent not in producer.inputs:
            raise ArenaPlanError("declared alias must be a view of its source input")
    requests = []
    for name, buffer in prog.buffers.items():
        if name not in owned:
            continue
        if name not in first:
            raise ArenaPlanError("intermediate has no producing dispatch")
        parent, offset = aliases.get(name, (None, 0))
        requests.append(
            Buffer(
                name,
                "arena",
                sizes[name],
                first[name],
                last[name],
                alignment=requirements.alignment if parent is None else 1,
                view_of=parent,
                view_offset=offset,
                persistent=name in requirements.persistent,
            )
        )
    return (
        sizes,
        tuple(requests),
        (StorageBank("arena", "arena", requirements.capacity_bytes, "byte"),),
        tuple(Reservation("arena", start, extent) for start, extent in requirements.reserved),
    )


def _bounded_arena_stats(prog, requirements, requests, arena_bytes):
    return {
        "n_intermediate_buffers": len(requests),
        "naive_total_bytes": sum(item.extent for item in requests),
        "arena_bytes": arena_bytes,
        "reuse_factor": round(sum(item.extent for item in requests) / arena_bytes, 2) if arena_bytes else 0.0,
        "n_result_buffers": len(prog.results),
        "align": requirements.alignment,
        "lifetime_basis": "explicit launch/completion events",
        "launch_events": list(requirements.launch_events),
        "completion_events": list(requirements.completion_events),
        "aliases": [list(row) for row in requirements.aliases],
        "persistent": list(requirements.persistent),
        "reserved": [list(row) for row in requirements.reserved],
        "capacity_bytes": requirements.capacity_bytes,
        "dispatch_digest": requirements.dispatch_digest,
        "physical_sizes": dict(requirements.physical_sizes),
        "execution_scope": "placement only; target lowering and runtime binding remain required",
    }


def check_bounded_arena(prog: DispatchProgram, requirements: ArenaRequirements, memory: MemoryPlan) -> None:
    """Check an emitted placement directly; do not invoke the allocator."""
    from merlin.semantic_compiler.buffer_allocation import check_buffer_allocation

    sizes, requests, banks, reservations = _bounded_arena_inputs(prog, requirements)
    if memory.sizes != sizes:
        raise ArenaPlanError("memory plan sizes differ from the current graph and physical footprints")
    if type(memory.arena_bytes) is not int or not 0 <= memory.arena_bytes <= requirements.capacity_bytes:
        raise ArenaPlanError("memory plan exceeds its arena capacity")
    check_buffer_allocation(requests, banks, memory.offsets, reservations=reservations)
    high_water = max(
        [memory.offsets[item.name] + item.extent for item in requests]
        + [item.start + item.extent for item in reservations],
        default=0,
    )
    if high_water > memory.arena_bytes:
        raise ArenaPlanError("memory plan offsets exceed the emitted arena extent")
    if memory.stats != _bounded_arena_stats(prog, requirements, requests, memory.arena_bytes):
        raise ArenaPlanError("memory plan statistics differ from the current requirements")


def plan_bounded_arena(prog: DispatchProgram, requirements: ArenaRequirements) -> MemoryPlan:
    """Place completion-aware temporaries; leave external buffers caller-owned."""
    from merlin.semantic_compiler.buffer_allocation import BufferAllocation, allocate_buffers, check_buffer_allocation

    sizes, requests, banks, reservations = _bounded_arena_inputs(prog, requirements)
    result = None
    # Reuse the existing deterministic planner for its supported case. Check
    # its witness against the completion-aware constraints before accepting it.
    # A failed heuristic is inconclusive; the finite-domain solver still runs.
    if not reservations and all(
        item.view_of is None and item.fixed_start is None and ARENA_ALIGN % item.alignment == 0 for item in requests
    ):
        conflicts = {
            item.name: {
                other.name
                for other in requests
                if item.name != other.name
                and (
                    item.persistent
                    or other.persistent
                    or item.first_event <= other.last_event
                    and other.first_event <= item.last_event
                )
            }
            for item in requests
        }
        offsets, _ = pack_disjoint([(item.name, item.extent) for item in requests], conflicts)
        try:
            check_buffer_allocation(requests, banks, offsets, reservations=reservations)
        except ValueError:
            pass
        else:
            result = BufferAllocation("sat", tuple(offsets.items()))
    if result is None:
        result = allocate_buffers(requests, banks, reservations=reservations)
    if result.status != "sat":
        raise ArenaPlanError(f"bounded arena allocation {result.status}: {result.reason}")
    offsets = dict(result.addresses)
    arena_bytes = max(
        [offset + sizes[name] for name, offset in offsets.items()]
        + [start + extent for start, extent in requirements.reserved],
        default=0,
    )
    memory = MemoryPlan(
        arena_bytes,
        offsets,
        sizes,
        _bounded_arena_stats(prog, requirements, requests, arena_bytes),
    )
    check_bounded_arena(prog, requirements, memory)
    return memory

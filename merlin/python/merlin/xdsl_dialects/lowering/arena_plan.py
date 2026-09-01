"""``merlin-arena-plan`` — compile-time static MEMORY PLAN for a dispatch program.

The current whole-model runtime materializes every intermediate with its own
``tensor.empty`` -> ``memref.alloc`` (~4391 mallocs on smolvla), the dominant share of the
dispatch overhead and the reason large models hit the RAM wall. This pass replaces that with
ONE arena: it computes each intermediate buffer's live range over the (topo-ordered) dispatch
DAG and greedy-colors non-overlapping buffers onto shared byte offsets, so the runtime binds
``arena_base() + offset`` with ZERO per-op allocation and bounded, reused memory.

Pure analysis over :class:`~.dispatch_program.DispatchProgram` (no IR mutation). NOT YET WIRED: no
caller consumes the resulting :class:`MemoryPlan`, and ``merlin_program.c`` -- named as its consumer
here and in two other docstrings -- does not exist in the tree. That is stated plainly because the
reverse impression is what makes a fail-open planner dangerous: every branch that cannot size a buffer
now raises :class:`ArenaPlanError` rather than substituting a plausible number, and offsets are aligned
to :data:`ARENA_ALIGN` (derived from the allocator the runtime actually uses), so whoever wires it gets
an honest refusal instead of a short arena.

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
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1,
    "index": 8, "ui8": 1, "ui32": 4,
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
        "narrower one, with no way to notice.")


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
                "produces a plan that looks complete and an arena that is too small.")
        if d == 0:
            return ARENA_ALIGN      # a legal empty buffer still needs a distinct, aligned address
        n *= d
    return max(n * _elem_bytes(dtype), 1)


def _align_up(n: int, a: int = ARENA_ALIGN) -> int:
    return ((int(n) + a - 1) // a) * a


@dataclass
class MemoryPlan:
    arena_bytes: int                       # total arena the runtime must provide
    offsets: dict[str, int]                # intermediate buffer id -> byte offset in arena
    sizes: dict[str, int]                  # buffer id -> byte size
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"arena_bytes": self.arena_bytes, "offsets": dict(self.offsets),
                "sizes": dict(self.sizes), "stats": dict(self.stats)}


def _live_ranges(prog: DispatchProgram) -> tuple[dict[str, int], dict[str, int]]:
    """first-def node index and last-use node index per buffer.

    A buffer is live across [def, last_use]. Args/consts are defined "before" node 0 (def=-1).
    Result buffers (and anything still consumed at the end) get last_use = len(nodes) so they
    survive the whole program (they are bound to the output buffer, not the arena, but we keep
    the conservative range so arena packing never reuses bytes that a result still needs).
    """
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
        last_use[bid] = n                  # lives to the end
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
    arena_bufs = [b for b in prog.buffers.values()
                  if b.kind == "intermediate" and b.id not in results]
    # Sizes are ALIGNED UP: the offsets are what the runtime adds to the arena base, so padding each
    # block is what keeps every buffer 64-byte aligned no matter what precedes it (see ARENA_ALIGN).
    sizes = {b.id: _align_up(_buf_bytes(b.shape, b.dtype)) for b in prog.buffers.values()}

    # process in definition order; tie-break larger-first so big buffers seat low and stable.
    order = sorted(arena_bufs, key=lambda b: (first_def.get(b.id, 0), -sizes[b.id]))

    # live allocations: list of (offset, size, dies_at_node).
    live: list[tuple[int, int, int]] = []
    offsets: dict[str, int] = {}
    arena_bytes = 0
    for b in order:
        d, u, sz = first_def.get(b.id, 0), last_use.get(b.id, 0), sizes[b.id]
        # free allocations whose last use is strictly before this buffer's definition.
        live = [(o, s, du) for (o, s, du) in live if du >= d]
        # first-fit: lowest offset with no overlap against still-live allocations.
        busy = sorted((o, o + s) for (o, s, _) in live)
        off = 0
        for lo, hi in busy:
            if off + sz <= lo:
                break                      # fits in the gap before this block
            if hi > off:
                off = _align_up(hi)        # push past this block, to the next aligned boundary
        offsets[b.id] = off
        assert off % ARENA_ALIGN == 0, (b.id, off)   # the invariant the C replay depends on
        live.append((off, sz, u))
        arena_bytes = max(arena_bytes, off + sz)

    arena_bytes = _align_up(arena_bytes)
    naive_total = sum(sizes[b.id] for b in arena_bufs)
    stats = {
        "n_intermediate_buffers": len(arena_bufs),
        "naive_total_bytes": naive_total,           # what per-op malloc would use (peak-naive)
        "arena_bytes": arena_bytes,                 # what the static plan uses
        "reuse_factor": round(naive_total / arena_bytes, 2) if arena_bytes else 0.0,
        "n_result_buffers": len(results),
        "align": ARENA_ALIGN,
    }
    return MemoryPlan(arena_bytes=arena_bytes, offsets=offsets, sizes=sizes, stats=stats)

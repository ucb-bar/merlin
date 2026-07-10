"""``merlin-arena-plan`` — compile-time static MEMORY PLAN for a dispatch program.

The current whole-model runtime materializes every intermediate with its own
``tensor.empty`` -> ``memref.alloc`` (~4391 mallocs on smolvla), the dominant share of the
dispatch overhead and the reason large models hit the RAM wall. This pass replaces that with
ONE arena: it computes each intermediate buffer's live range over the (topo-ordered) dispatch
DAG and greedy-colors non-overlapping buffers onto shared byte offsets, so the runtime binds
``arena_base() + offset`` with ZERO per-op allocation and bounded, reused memory.

Pure analysis over :class:`~.dispatch_program.DispatchProgram` (no IR mutation) — the C replay
(`merlin_program.c`) consumes the resulting :class:`MemoryPlan`. Args/weights are bound from the
model arg table; results are bound to the caller's output buffer; both stay OUT of the arena.
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


def _elem_bytes(dtype: str) -> int:
    d = dtype.strip()
    if d in _DTYPE_BYTES:
        return _DTYPE_BYTES[d]
    # tolerate spellings like "f32", "!llvm...", fall back to 4 (f32) with a marker.
    for k, v in _DTYPE_BYTES.items():
        if d.endswith(k):
            return v
    return 4


def _buf_bytes(shape: list[int], dtype: str) -> int:
    n = 1
    for s in shape:
        n *= int(s) if s and int(s) > 0 else 1
    return max(n * _elem_bytes(dtype), 1)


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
    sizes = {b.id: _buf_bytes(b.shape, b.dtype) for b in prog.buffers.values()}

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
                off = hi                   # push past this block
        offsets[b.id] = off
        live.append((off, sz, u))
        arena_bytes = max(arena_bytes, off + sz)

    naive_total = sum(sizes[b.id] for b in arena_bufs)
    stats = {
        "n_intermediate_buffers": len(arena_bufs),
        "naive_total_bytes": naive_total,           # what per-op malloc would use (peak-naive)
        "arena_bytes": arena_bytes,                 # what the static plan uses
        "reuse_factor": round(naive_total / arena_bytes, 2) if arena_bytes else 0.0,
        "n_result_buffers": len(results),
    }
    return MemoryPlan(arena_bytes=arena_bytes, offsets=offsets, sizes=sizes, stats=stats)

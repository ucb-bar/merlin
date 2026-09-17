---
title: "Design note: the static memory arena and the path we actually measure"
kind: design
status: current
owner: core
last_verified: 2026-09-04
related: [runtime_escape_audit, compiler_plane]
code_refs: [merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py, merlin/python/merlin/llvmlower/arena_bind.py, merlin/python/merlin/llvmlower/lower.py, merlin/python/merlin/xdsl_dialects/lowering/dispatch_program.py, merlin/python/merlin/runtime/program.py, merlin/python/merlin/llvmlower/pipeline.py]
---

# The static arena and the path we actually measure

## What is wired now

`lower_model(static_arena=True)` (or `MERLIN_STATIC_ARENA=1`) runs
`llvmlower.arena_bind.bind_arena` on the LLVM IR the pipeline returns, before it is written to
`model.ll` and compiled. That is the last seam the measured whole-model path still passes through as
text: `mining.k1.build_k1_binary` calls `lower_model_file` and compiles the `.ll` it gets back, so
nothing in `mining/k1.py` or `pipeline.py` had to change. Default OFF; unflagged, the `.ll` is
byte-identical to the baseline (verified by digest on all three int8 builds).

Measured on the host, int8, `blocking=False, features=frozenset()`, `hoist_static_allocs=False`:

| model | malloc | free | memrefCopy | llvm.memcpy | arena bytes | output |
|---|---|---|---|---|---|---|
| deepjscc | 112 → **0** | 112 → **0** | 11 → 11 | 53 → 53 | 23,020,288 | bit-identical |
| lstmnetvit | 309 → **0** | 309 → **0** | 8 → 8 | 123 → 123 | 20,047,616 | bit-identical |
| small_llama | 115 → **0** | 115 → **0** | 25 → 25 | 38 → 38 | 389,888 | bit-identical |

Every allocation is bound; none is refused. The copies are untouched — the arena removes
allocations, not copies, and it forces no new copy.

### Why it is safe

An arena that reuses bytes turns a lifetime mistake into wrong numbers rather than a crash, so the
argument is written out and tested rather than asserted:

1. The input program is already correct under `malloc`/`free`, so `[malloc, free)` is a sound
   over-approximation of each buffer's live range. Liveness is never guessed from the IR's shape.
2. A bound site runs at most once per call: its `malloc` block and its `free` block are required to
   lie in no CFG cycle.
3. Two buffers share bytes only when neither allocation happens inside the other's window
   (forward reachability from the malloc, stopping at the free).
4. Nothing outlives the call: the raw pointer's only uses are the alignment `ptrtoint` and its
   `free` (plus `insertvalue` into a memref descriptor, admitted only when every callee in the
   function is known non-capturing), and `@forward` returns void.

Anything that fails a condition stays a `malloc`/`free` pair and is counted as a refusal with a
reason. The placement is re-checked against the offsets actually emitted, and
`merlin/tests/runtime/test_arena_bind.py` proves that check can fail by mutating the conflict
relation — a guard that cannot fail is evidence of nothing.

The one fold that makes the analysis non-vacuous: the ownership passes emit every dealloc behind
`br i1 true, label %free, label %skip`. Keeping the never-taken `%skip` edge makes every free look
optional, so no buffer is ever provably dead. Before that edge was folded away the binder bound 95
of 112 allocations at 1.00x reuse on a program whose buffers do not all coexist.

## The footprint does not shrink, and here is exactly why

`arena_plan.plan_arena` reports 5.77x (deepjscc), 7.67x (lstmnetvit) and 11.73x (small_llama) reuse
over a `DispatchProgram`. The binder gets **1.00x** on all three. That is not a weaker planner: it is
a different liveness.

`plan_arena` ends a buffer's range at its last USE. The emitted binary frees every intermediate at
the END of `@forward` — measured on the deepjscc int8 build: the last `malloc` is at line 10,530 of a
12,112-line function and all 112 `free` calls are between lines 10,649 and 11,426, after the last
compute. Under `malloc`/`free` liveness every intermediate is therefore live for essentially the
whole inference, the peak already equals the naive total, and there is nothing to reclaim. The arena
is the same bytes in one block instead of N — no memory regression, no memory win.

This contradicts the note in `pipeline._dealloc_passes`, which records that
`func.func(optimize-allocation-liveness)` "moved nothing" and concludes "ownership-based deallocation
is already placing the frees tightly enough that there is no slack to reclaim". The frees are not
tight; they are all at the end. Whatever that experiment measured, the emitted IR says the slack is
there — 5.77x of it on deepjscc.

**The highest-value follow-up is therefore in the pass pipeline, not in the binder**: get the
deallocations placed at last use (`optimize-allocation-liveness` actually firing, or an equivalent),
and the same binder produces the `plan_arena` footprint with no change to its analysis. That edit is
in `llvmlower/pipeline.py`.

The alternative — teaching the binder to compute last-use liveness itself, from the transitive
closure of pointer-derived SSA values — was deliberately not taken. It replaces an argument that
rests on the input program's own correctness with one that rests on an alias analysis being complete,
and an incomplete closure produces exactly the silent corruption the rest of this design avoids.

## The lane the arena was originally written for

`outline_dispatches` splits `@forward` into `forward$kernel_<i>` plus a thin driver;
`build_dispatch_program` flattens the driver into buffers + nodes; `runtime.program.build_program`
plans the arena over it. Its consumers are `passes.run_dialect_plane` (statistics and partitioning)
and `runtime.dispatch_runtime` (a host-Python executor over per-kernel `.so`s). Neither produces a
deployable binary, and the three things that lane still needs are unchanged:

1. **A per-kernel emission path.** `c_runtime.generate` unrolls a single `_mlir_ciface_forward` call.
2. **The replay engine.** `merlin/runtime/c/merlin_program.c` does not exist in the tree.
   (`merlin_bump_linux.c`, behind `MERLIN_BUMP_MALLOC`, is a measurement proxy: one bump arena,
   `free` is a no-op. It measures the allocator's share of wall; it plans nothing.)
3. **A static-shape precondition**, surfaced as a build diagnostic rather than an exception.

`merlin/runtime/c/merlin_hal.h` and `hal_linux.c` are still referenced by nothing. The binder does
not use them: it emits the arena as an `internal global` in `model.ll`, which keeps the change
link-neutral. Routing it through `merlin_hal_arena_base()` instead would let a board place the arena
outside `.bss` — worth doing when a target needs it, at the cost of a link change in `mining/k1.py`
and `codegen.build_host_shared`.

## Adjacent levers, and why they are not substitutes

`promote_buffers_to_stack` (registered, default OFF) takes the `small_llama_int8_consistent`
allocation count from 181 to 3 — but LLVM's loop-idiom pass then re-forms 177 `memcpy` calls out of
the now provably non-aliasing copy loops. The allocation-related levers trade against each other, so
they belong to the search space. The arena is not one of them: nothing about VLEN, the register file
or the ISA decides whether allocating 175 buffers per inference is worse than reusing one arena, so
it is a flag to be measured and then defaulted, not an axis to be searched.

One thing the arena gives up that `malloc` had: LLVM infers `noalias` on a `malloc` result and cannot
on a `getelementptr` into a shared global. The plan guarantees the bound buffers do not alias while
live, but the optimizer no longer knows it. That is a candidate explanation if the board measurement
comes back slower, and the fix would be emitting `!alias.scope`/`!noalias` metadata per buffer.

## What is already fixed

Two defects that would have made a wired `plan_arena` produce silently wrong offsets are closed, with
tests in `merlin/tests/runtime/test_dispatch_program_regions.py`:

- **Region-captured reads are recorded.** `op.operands` on an `scf.for` covers lb/ub/step/iter_args
  only, so a tensor the loop body read from the enclosing scope appeared in no node's `inputs`,
  liveness ended before the last read, and the planner handed those bytes to the next buffer.
  `build_dispatch_program` now walks the regions and folds the captures into `inputs`;
  `plan_arena` refuses a region-carrying node whose capture set was never established.
- **A kernel call hidden in a region is refused.** The top-level walk that pairs driver calls with
  the dispatch table cannot see a nested `func.call`, so every later call took the wrong table entry
  — wrong symbol and wrong `prov.region_id` — while `verify_program` still reported a valid DAG.

---
title: "Design note: what wiring the static memory arena to the measured path requires"
kind: design
status: current
owner: core
last_verified: 2026-09-04
related: [runtime_escape_audit, compiler_plane]
code_refs: [merlin/python/merlin/xdsl_dialects/lowering/arena_plan.py, merlin/python/merlin/xdsl_dialects/lowering/dispatch_program.py, merlin/python/merlin/runtime/program.py, merlin/python/merlin/llvmlower/c_runtime.py, merlin/python/merlin/mining/k1.py]
---

# The static arena and the path we actually measure

## The gap, stated plainly

`arena_plan.plan_arena` computes a single static arena and per-buffer offsets for a
`DispatchProgram`, the way ExecuTorch plans one arena for a whole model. It works, it is tested,
and **nothing on the measured path can call it** — not because it is unfinished, but because the
two lanes never meet.

Measured on `small_llama_int8_consistent` (183 dispatches, CPU-only, no board): 433 intermediate
buffers, 3,114,112 B if each is allocated on its own, **265,536 B planned** — 11.73x reuse. That
number is what the wiring would be worth; it is not what the binary does today.

## Two lanes that never meet

**The lane we measure** is monolithic. `mining.k1.build_k1_binary` runs
`zephyr_model.prepare_for_lowering` -> `llvmlower.lower.lower_model_file` -> `model.ll` -> `clang -c`
-> `model.o`, then `llvmlower.c_runtime.generate` emits `model_gen.h` / `model_io.h` /
`model_call.c` / `weights.bin`. There is exactly ONE entry symbol: `model_call.c` emits
`void merlin_invoke(void **d) { _mlir_ciface_forward(d[0], d[1], ...); }`. Every intermediate
allocation is created *inside* `model.o` by `one-shot-bufferize` (which turns each `tensor.empty`
into a `memref.alloc`) and released by the ownership-based deallocation passes. On K1 the build
passes `hoist_static_allocs=False`, so those stay heap allocations by design. Neither
`dispatch_program` nor `arena_plan` is imported anywhere along that chain.

**The lane the arena belongs to** is the dispatch program. `outline_dispatches` splits `@forward`
into `forward$kernel_<i>` plus a thin driver; `build_dispatch_program` flattens the driver into
buffers + nodes; `runtime.program.build_program` plans the arena over it. Its consumers are
`passes.run_dialect_plane` (statistics and partitioning only) and `runtime.dispatch_runtime` (a
host-Python executor over per-kernel `.so`s). Neither produces a deployable binary.

## What wiring actually requires

Three pieces, none of them a one-line call:

1. **A per-kernel emission path.** `c_runtime.generate` unrolls a single `_mlir_ciface_forward`
   call. Binding arena offsets requires the emitted C to invoke kernels one at a time with
   descriptors it builds itself, which means the K1 build must lower the OUTLINED module (driver +
   `forward$kernel_<i>`) rather than the monolithic `@forward`, and `c_runtime` must emit a kernel
   table instead of one call.
2. **The replay engine.** `merlin_program.c` — named as the consumer in three separate docstrings —
   **does not exist in the tree**. Something has to walk the node list, bind `arena_base()+offset`
   per buffer, evaluate the view nodes, and call the kernel table. `merlin/runtime/AGENT.md` already
   records its absence. (`merlin_bump_linux.c`, behind `MERLIN_BUMP_MALLOC`, is only a measurement
   proxy: one bump arena, `free` is a no-op. It measures the allocator's share of wall; it plans
   nothing.)
3. **A static-shape precondition.** `plan_arena` refuses any dynamic extent by design. Whichever
   models are to be planned must be checked to have fully static outlined buffers first, and the
   refusal surfaced as a build diagnostic rather than an exception at the wrong layer.

Only then does the arena bind to bytes the board executes. Until all three land, `plan_arena`
remains an analysis whose output nothing reads — which is the honest reading, and the reason its
own module docstring says so.

## Adjacent levers, and why they are not substitutes

`promote_buffers_to_stack` (registered, default OFF) takes the `small_llama_int8_consistent`
allocation count from 181 to 3 — but LLVM's loop-idiom pass then re-forms 177 `memcpy` calls out of
the now provably non-aliasing copy loops. The allocation-related levers trade against each other,
so they belong to the search space, not to a default. Wiring the arena does not retire that lever
and the lever does not retire the arena: one removes allocations from the emitted object, the other
removes them from the program the runtime replays.

## What is already fixed

Two defects that would have made a wired planner produce silently wrong offsets are closed, with
tests in `merlin/tests/runtime/test_dispatch_program_regions.py`:

- **Region-captured reads are recorded.** `op.operands` on an `scf.for` covers lb/ub/step/iter_args
  only, so a tensor the loop body read from the enclosing scope appeared in no node's `inputs`,
  liveness ended before the last read, and the planner handed those bytes to the next buffer.
  `build_dispatch_program` now walks the regions and folds the captures into `inputs`;
  `plan_arena` refuses a region-carrying node whose capture set was never established.
- **A kernel call hidden in a region is refused.** The top-level walk that pairs driver calls with
  the dispatch table cannot see a nested `func.call`, so every later call took the wrong table entry
  — wrong symbol and wrong `prov.region_id` — while `verify_program` still reported a valid DAG.

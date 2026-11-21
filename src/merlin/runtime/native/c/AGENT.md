# src/merlin/runtime/native/c

## Purpose

The **Merlin C runtime**: a generic, data-driven driver that executes a compiled whole model (`_mlir_ciface_forward`) by building MLIR memref descriptors from a generated argument table. Target-agnostic core; the same code runs on host (verification) and bare-metal spike/Zephyr.

## What belongs here

- `merlin_model.h/.c` — the descriptor builder + `merlin_run` (arg table + weights base + input pointers + output buffer → descriptors → `merlin_invoke`).
- `merlin_op_prof.c` — optional per-operation profiler for whole-model images.
- `merlin_bump_linux.c`, `merlin_alloc_guard_zephyr.c`, `merlin_fatal_zephyr.c`, `merlin_socinit_zephyr.c` — per-environment allocator, fatal-handler and SoC-init units.
- `libomp_pthread.{h,c}`, `libomp_zephyr.{h,c}`, `omp_static_schedule.h` — the minimal OpenMP runtime used by parallel builds.
- `row_major_k_window.h`, `finite_float_boundary.h` — portable helpers also read from this include directory by companion runtimes.

## What does not belong here

- Generated, model-specific files (`model_gen.h`, `model_io.h`, `model_call.c`, `weights.bin`) — those are emitted per model by `merlin/python/merlin/compiler/lower/model/c_runtime.py` into a build dir.
- Target-specific harness (crt/HTIF/malloc/linker) — that is `src/merlin/runtime/native/baremetal/<env>/`.
- Per-kernel dispatch logic (the per-dispatch outliner/dispatch-table runtime, when added) layers on top but the descriptor ABI stays here.

## Interfaces

- Input: the generated `merlin_arg_t[]` table (`MERLIN_ARGS`), the weight blob, `MERLIN_INPUT_PTR`, an output buffer.
- The descriptor struct layout matches MLIR's `memref<...>` lowering exactly: `{allocated, aligned, offset, sizes[rank], strides[rank]}`. Do not change it without matching `convert-memref-to-llvm`.
- Driven end to end by `merlin/python/merlin/runtime/backends/spike_model.py` (build → run → verify).

## Invariants

- **Target-agnostic**: no ISA assumptions here; the only target-specific code is in `baremetal/`/codegen flags.
- Row-major contiguous strides; weights referenced by byte offset into the blob (never copied).
- Output emitted as exact f32 bit patterns so `spike == host` is checkable up to FP reassociation (different ISAs reassociate; gate on cos≈1 / rel<1e-4, not bit-equality).

## Testing expectations

`merlin/python/tests/test_spike_model.py` — small_llama whole-model spike == host == torch (skips without the chipyard toolchain). Verified: cos 0.9999999.

## Notes for future agents

This monolithic-`forward` path is the correctness baseline; the per-dispatch outliner + dispatch-table runtime (for multicore + bounded memory) builds on the same descriptor ABI and `spike_model` build flow.

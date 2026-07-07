# AGENT.md — merlin/runtime

## Purpose
The **Merlin-owned, target-INDEPENDENT C runtime substrate** — the dependency-free code that drives a
compiled model via its MLIR C-runtime interface. Hand-authored source (cannot be generated).

## What lives here
- `c/` — `merlin_model.{h,c}` (build memref descriptors, drive the model), `merlin_host_main.c`
  (host verification driver), `merlin_bump_linux.c` (bump allocator). Target-agnostic core.
- `abi/` — `mlir_runtime.c` (`memrefCopy`, `rsqrtf`): the MLIR C-runtime symbols Merlin implements.
- `baremetal/spike/` — ONE execution environment's harness (crt/htif/libc_min/malloc/linker +
  `rvv_matmul_i8.S`). RVV/spike-flavored **by nature** (an isolated backend, not an overfit leak).

## What does NOT belong here
- Target-specific codegen (that's a target/backend concern). Compiled objects/ELFs → `build/`.

## Used by
`merlin.runtime.backends.{spike_model,zephyr_model}`, `merlin.rvvgen.k1`, `merlin.baselines.buddy`.

## Notes
`c/{merlin_hal.h,hal_linux.c,hal_baremetal_spike.c}` are **untracked** forward-scaffolding for a
not-yet-built lean replay runtime (`merlin_program.c`); they are not wired in yet — do not treat as
live code. Every subdirectory has an AGENT.md.

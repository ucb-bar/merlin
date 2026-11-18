# src/merlin/runtime/native

## Purpose
The **Merlin-owned, target-INDEPENDENT C runtime substrate** — the dependency-free code that drives a
compiled model via its MLIR C-runtime interface. Hand-authored source (cannot be generated).

## What lives here
- `c/` — `merlin_model.{h,c}` (build memref descriptors, drive the model), `merlin_bump_linux.c`
  (bump allocator). Target-agnostic core.
- `abi/` — `mlir_runtime.c` (`memrefCopy`, `rsqrtf`): the MLIR C-runtime symbols Merlin implements.
- `baremetal/spike/` — ONE execution environment's harness (crt/htif/libc_min/malloc/linker +
  `rvv_matmul_i8.S`). RVV/spike-flavored **by nature** (an isolated backend, not an overfit leak).

## What does NOT belong here
- Target-specific codegen (that's a target/backend concern). Compiled objects/ELFs → `build/`.

## Used by
`merlin.runtime.backends.{spike_model,zephyr_model}`, `merlin.runtime.backends.linux.remote_session`.

## Notes
Every subdirectory has an AGENT.md.

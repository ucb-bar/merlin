# AGENT.md — merlin/runtime/abi

## Purpose

Merlin-owned C implementations of the MLIR runtime symbols the lowered model calls — so the compiled `forward()` links the same on host and on bare-metal without depending on `libmlir_c_runner_utils`.

## What belongs here

- `mlir_runtime.c` — `memrefCopy` (the strided copy `memref.copy` lowers to) plus the non-standard math symbols the libm lowering needs (`rsqrtf`/`rsqrt`). Pure, freestanding C; no target assumptions.

## What does not belong here

- The model driver / descriptor builder (that is `merlin/runtime/c/`).
- Target-specific harness (crt/HTIF/malloc/linker — `merlin/runtime/baremetal/<env>/`).
- Anything MLIR-version-specific beyond matching the C-runtime symbol contract.

## Interfaces

Compiled and linked into both the host `.so` (`llvmlower/codegen.py`) and the bare-metal ELF (`runtime/backends/spike_model.py`). `memrefCopy`'s signature + descriptor layout must match upstream MLIR's `UnrankedMemRefType` exactly.

## Invariants

- Freestanding-safe (no libc beyond `string.h`/`math.h`); links for x86 and rv64gcv unchanged.
- If `convert-memref-to-llvm` or `convert-math-to-libm` starts emitting a new runtime symbol, add it here (the build will show the undefined reference).

## Testing expectations

Exercised transitively by `test_llvmlower.py` (host) and `test_spike_model.py` (spike).

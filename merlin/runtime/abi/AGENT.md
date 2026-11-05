# AGENT.md — merlin/runtime/abi

## Purpose

Merlin-owned C implementations of the MLIR runtime symbols the lowered model calls — so the compiled `forward()` links the same on host and on bare-metal without depending on `libmlir_c_runner_utils`.

## What belongs here

- `mlir_runtime.c` — `memrefCopy` (the strided copy `memref.copy` lowers to) plus the non-standard math symbols the libm lowering needs (`rsqrtf`/`rsqrt`). Pure, freestanding C; no target assumptions.
- `portable_math.h` (+ the generated `portable_math_tables.h`) — `merlin_erff`/`merlin_expf`/`merlin_tanhf`/`merlin_logf`, the portable transcendentals the `portable-transcendentals` numeric policy binds the lowered model to (`llvmlower/portable_math.py`). Included once, by `mlir_runtime.c`, so every link path carries them. Double-precision evaluation with every multiply-add an explicit fma: the bits do not depend on the compiler, its flags or the C library.

## What does not belong here

- The model driver / descriptor builder (that is `merlin/runtime/c/`).
- Target-specific harness (crt/HTIF/malloc/linker — `merlin/runtime/baremetal/<env>/`).
- Anything MLIR-version-specific beyond matching the C-runtime symbol contract.

## Interfaces

Compiled and linked into both the host `.so` (`llvmlower/codegen.py`) and the bare-metal ELF (`runtime/backends/spike_model.py`). `memrefCopy`'s signature + descriptor layout must match upstream MLIR's `UnrankedMemRefType` exactly.

## Invariants

- Freestanding-safe (no libc beyond `string.h`/`math.h`); links for x86 and rv64gcv unchanged.
- `portable_math.h` computes the same bits on every build: never compile it with `-ffast-math` (it refuses) or edit `portable_math_tables.h` by hand (regenerate with `build_tools/scripts/gen_portable_math_tables.py`). Its accuracy and the evidence are in `docs/design/portable_transcendentals.md`.
- If `convert-memref-to-llvm` or `convert-math-to-libm` starts emitting a new runtime symbol, add it here (the build will show the undefined reference).

## Testing expectations

Exercised transitively by `test_llvmlower.py` (host) and `test_spike_model.py` (spike).
The portable transcendentals: `merlin/tests/runtime/test_portable_transcendentals.py` (bit-identity across host and RISC-V compilers on spike, accuracy, special values) and `merlin/tests/rvv/test_portable_gelu_gate.py` (the W8A8 per-element trip in miniature).

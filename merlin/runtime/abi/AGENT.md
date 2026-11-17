# AGENT.md — merlin/runtime/abi

## Purpose

Merlin-owned C implementations of the MLIR runtime symbols the lowered model calls — so the compiled `forward()` links the same on host and on bare-metal without depending on `libmlir_c_runner_utils`.

## What belongs here

- `owned_result_views.{h,c}` validate returned logical memref views against
  the selected invocation allocator, before payload reads. Preserve negative
  strides, zero-access empty views, exact static extents, overflow and byte
  bounds, complete returned-owner closure and independent copy destinations.
  This is reusable C descriptor validation, not a model driver or a target
  placement policy. Caller metadata is compiler-owned; returned addresses and
  extents are untrusted. Allocation context and target linking remain explicit.

- `mlir_runtime.c` — `memrefCopy` (the strided copy `memref.copy` lowers to) plus the non-standard math symbols the libm lowering needs (`rsqrtf`/`rsqrt`). Pure, freestanding C; no target assumptions.
- `owned_results.c` is an explicitly selected host allocation runtime for
  returned memref results. Its thread-local synchronous invocation records
  actual capacities and charged heap bytes, including metadata/alignment.
  Preserve empty ownership, count limits and invalid/double-free refusal.
  Generic callbacks stay hidden and cleanup frees recorded owners only. This
  host selection is not linked into default or bare-metal builds.
  `MERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1` explicitly selects one nonconcurrent
  process-local invocation context; it grants no multihart/thread safety.
  `MERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1` delegates fatal diagnostics/exit
  to the selected executor without hosted stdio/TLS. Each selected macro must
  equal one. Default native TLS and diagnostics remain unchanged. A target must
  still bind its actual allocator, memory map, execution and failure contracts.
- `half_extend.h` supplies typed compiler-rt `__extendhfsf2`/`__truncsfhf2` ABIs when the
  selected C compiler supports binary16. Its pinned LLVM specialization retains
  all finite/subnormal/zero words. Narrowing rounds nearest ties to even and
  quiets computed NaNs while retaining their sign and truncated payload.
  Neither helper may recursively call a half conversion. Preserve the LLVM
  license and header source dependency in host/target compilation observations.

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

---
title: 6. Lower through scalar and OOT implementations
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [05-transformations, 07-building]
code_refs: [src/merlin/compiler/pipeline.py, src/merlin/compiler/backend/selected.py, src/merlin/compiler/backend/cabi.py, src/merlin/compiler/lower/regions.py, src/merlin/compiler/lower/scalar, src/merlin/compiler/lower/llvm, src/merlin/compiler/artifact.py]
---

# 6. Lower through scalar and OOT implementations

[Previous](05-transformations.md) | [Guide index](../README.md) | [Next](07-building.md)


Lowering turns a selected implementation into code or another admitted artifact.
This is where a region enters its OOT compiler. Shared scalar computation and
control lowering remain reusable Merlin components.

## Find the scalar code

`compiler/backend/scalar.py` exposes the installed `merlin.scalar` provider.
`compiler/lower/scalar/` owns the shared scalar pass sequence.
`compiler/lower/llvm/` owns terminal LLVM conversion and ABI handling.
The native invocation layer is `compiler/lower/native/`.

The small `merlin.layout` provider uses `compiler/lower/c/layout.py` for its
supported layout-only subset. It is useful for the quickstart, but does not
replace the scalar arithmetic compiler.

The historical `llvmlower/` family still contains active saved-model machinery
as well as compatibility imports. Follow the implementation links in
[the source map](../reference/repo_structure.md); do not assume every older
directory is either dead code or a complete second compiler.

## Find the OOT boundary

`PlannedCompiler` validates the selected regions and invokes their bound backend
services. `backend/selected.py` passes the chosen lowering recipe and checks the
returned artifacts against that exact selection. `lower/regions.py` connects
selected region emission to the existing device object builder.

The OOT repository owns its dialect, instruction selection, device kernels,
target resource descriptions, encoders and target ABI glue. Merlin owns shared
host transformations, orchestration, buffer ownership and build infrastructure.
An optimization that is independent of the accelerator belongs in the shared
library even if it was first measured on one target.

RVV implementation code belongs in `rvv-mlir`. A CPU with scalar and RVV
implementations can still be one shared hardware resource. Scalar fallback
must not be reported as evidence that vector or accelerator code executed.

## Outputs do not have to be LLVM

A backend may return compatible IR, an object file, a device binary, a command
stream or another declared artifact kind. `Compilation` records the artifacts
and their hashes. Atlas is not required to pass through LLVM to participate.

What matters is the next boundary: accepted representation, data layout, target
ABI, runtime requirements and invocation method. Textual interchange between
xDSL and native MLIR requires an admitted dialect/type/version subset. Separate
processes avoid a shared C++ ABI but do not remove IR compatibility requirements.

## Make the region callable

For a C-dispatch image, each region needs an admitted callable wrapper. The wrapper
describes its symbol, arguments, memory use and completion behavior. The device
backend supplies the target-specific call or launch; shared control calls that
wrapper. A raw device binary alone is not a callable host function.

`backend/cabi.py` defines the C call contract. `backend/stateful.py` handles
explicit context/workspace contracts. `build/control.py` emits shared control.
A synchronous wrapper must complete the work it promises before returning;
an asynchronous design needs explicit completion dependencies.

This boundary does not require an extra copy by itself. Packing or transfer is
required only when the selected layout, address space or ABI requires it, and
must remain visible in the plan and cost accounting.

## Open the implementation

| Responsibility | Source |
| --- | --- |
| Compile selected regions | [`pipeline.py`](../../src/merlin/compiler/pipeline.py) |
| Invoke the exact chosen lowering | [`backend/selected.py`](../../src/merlin/compiler/backend/selected.py) |
| Scalar provider | [`backend/scalar.py`](../../src/merlin/compiler/backend/scalar.py) |
| Scalar pass sequence | [`lower/scalar/pipeline.py`](../../src/merlin/compiler/lower/scalar/pipeline.py) |
| C wrapper contract | [`backend/cabi.py`](../../src/merlin/compiler/backend/cabi.py) |
| Artifact records | [`artifact.py`](../../src/merlin/compiler/artifact.py) |

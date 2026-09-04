# rvvhost CPU backend

This package is an out-of-tree, dependency-light compiler for the version-1
Merlin CPU capsule ABI.  The host compiler validates the capsule descriptor,
lowers it to a generated `rvvhost` plan, and emits a freestanding-compatible C
kernel.  Scalar, scalable-RVV, runtime-proved fixed-VLEN, and deterministic
multicore paths share the same scalar reference semantics.

The generated dialect description is under `include/`.  It records the legal
CPU-level operations used in `lowered.mlir`; the standalone compiler does not
require an MLIR installation at grading time.

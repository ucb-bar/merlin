---
title: "Source-bound compiler constants"
kind: reference
status: current
owner: ir
last_verified: 2026-11-09
related: [architecture, triton_frontend]
code_refs:
  - src/merlin/frontends/compiler_constants.py
  - src/merlin/frontends/compile_inputs.py
  - src/merlin/frontends/derived_compile_inputs.py
  - src/merlin/runtime/captured_constants.py
  - src/merlin/llvmlower/c_runtime.py
---

# Source-bound compiler constants

Model2MLIR may export literal tensors as function arguments. New captures store
their original dense bytes in `weights.safetensors`, alongside learned state,
and declare manifest entries with `kind: constant`. Each entry binds its actual
exported placeholder name, source target, dtype, shape, weight key and payload
SHA-256. The receipt's additive `m2m.compiler-constants.v1` profile identifies
the complete argument roster. Legacy evaluator `extra.npz` remains separate.

Compiler staging validates every binding against the original prepared graph's
`CONSTANT_TENSOR` placeholder, actual parsed function argument, manifest and
immutable safetensors payload. Duplicate, omitted, mistyped, renamed or stale
bindings refuse. Old lifted-constant captures without this contract remain
unbound; the compiler never recovers their values from evaluator archives.

Both raw model2MLIR and derived triton2MLIR packages emit a compiler-only
`compiler-constants.json` projection with schema `merlin.compiler-constants.v1`.
Derived intake checks the original parent correspondence and requires its
complete public function type to survive kernel binding. Optional evaluator
archives and their value digests remain excluded from staged members.

`CaptureConstantTable` validates the already owned immutable manifest/blob
once, then reads selected typed tensors under explicit byte budgets. The
streaming `verify_capture_constant` API retains its independent bounded-memory
behavior. Both reject invalid layouts and return exact payload bytes without
floating conversion or packing. Scalar and empty tensor shapes retain their
original meaning. This establishes source-bound byte custody; numerical,
effect, ownership, packing and target admission keep their separate gates.

The normal C runtime generator binds constants as weight arguments using the
existing exact blob/alignment path. Source constants cannot use parameter
stubs. Real triton2MLIR model tests qualify f16/bf16/f32/f64, Boolean and signed
i8/i16/i32/i64 carriers through native/Merlin execution in three memory
profiles. Copy kernels preserve signaling NaNs, signed zero and large i64
words; that result does not qualify arbitrary arithmetic on those values.

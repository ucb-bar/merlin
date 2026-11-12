---
title: "FP8 reader and constant bindings"
kind: reference
status: current
owner: ir
last_verified: 2026-11-11
related: [core_dialects, triton_frontend]
code_refs:
  - src/merlin/xdsl_dialects/fp8.py
  - src/merlin/common/dense_float_ir.py
  - src/merlin/frontends/linalg_mlir.py
  - src/merlin/llvmlower/loop_tensor_destinations.py
  - src/merlin/llvmlower/dealloc_lifetime.py
  - src/merlin/llvmlower/c_runtime.py
  - src/merlin/common/mlir_query.py
  - src/merlin/llvmlower/fp8_widening.py
  - src/merlin/llvmlower/fp8_narrowing.py
  - merlin/tests/ir/test_fp8_feature_composition.py
  - src/merlin/llvmlower/pipeline.py
---

# FP8 reader and constant bindings

Merlin's xDSL reader recognizes all four FP8 alternatives of Triton's
`TT_Float` at revision `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`:
`f8E4M3FN`, `f8E4M3FNUZ`, `f8E5M2` and `f8E5M2FNUZ`.
They retain their ordinary builtin MLIR spelling and distinct type identity.
The type parser works without an optional constant codec installed.

The `fp8` installation extra supplies `ml-dtypes>=0.6,<0.7` for constant
packing. Decimal scalar and dense constants round to the declared format.
Typed hex scalar/list constants and raw dense bytes preserve every original
encoding, including all NaN words and signed zeros. A float carrier retains
the raw word alongside its decoded Python value so re-encoding a NaN cannot
replace its source payload. Encoding from a different format uses numeric
conversion; the raw word is reused only for its original format.

The reader extends complete xDSL float constraints, including their flattened
dense-constant copies and the union captured by Linalg's abstract named-operation
builder. Custom FP8 fills therefore construct their actual scalar regions in
either reader, without depending on another producer's process-global imports.
Integer and restricted float constraints retain their
original meaning. This grammar/verification binding grants no permission to
execute a numeric operation or erase an effect. The parser registry is shared
with triton2MLIR when both independent readers occupy one interpreter, so
import order cannot produce different Python types for the same MLIR spelling.

`merlin/tests/ir/test_fp8_float_constraints.py` checks both reader entrances,
all four types, all 256 words in each format, scalar/dense serialization,
distinct finite/NaN/zero interpretations and nonfloat refusals. triton2MLIR
also checks both import orders and native MLIR parse/print byte preservation.
Attributed `tensor.yield` operations print in generic syntax, retaining their
values, types and complete provenance while avoiding xDSL's incompatible custom
attribute order. Native reconstruction of mixed-type named regions can then
re-read the whole module, including padded FP8 tensors.
Validation uses xDSL 0.68 and the existing native SDK's xDSL 0.65, with
ml_dtypes 0.6.0. No LLVM or Triton rebuild is needed.

The ordinary tensor/memref host ABI additionally transports original FP8 words
as one-byte storage. triton2MLIR archive carriers are explicitly `uint8`, with
their logical float format retained in MLIR and signature metadata. Native
LLVM conversion already handles FP8 loads, stores, bitcasts and value/control
transport. No custom pass or integer public ABI is introduced.

Loop destination and allocation-generation checks recognize the four FP8
formats as non-handle value types. Native alias/lifetime verification still
decides legality: invalid ownership/double release and unknown calls refuse.
Footprint accounting and generated C input arrays use one byte per element;
non-`uint8` FP8 input carriers refuse before numeric reinterpretation. C byte
storage checks preserve all 256 words and do not establish numeric C-output
interpretation or target execution.

The actual triton2MLIR integration qualifies all four formats in tensor,
buffer and guarded-memory profiles, including scalar events, masked tails,
runtime loops, empty dimensions and reused native Merlin libraries. Its saved
demo binds exact word references and keeps evaluator bytes/digests out of
compiler staging. The default converter still leaves direct FP8 extension
probes unresolved; the explicit widening feature below supplies those casts.

Numeric cast rounding/overflow, arithmetic/scaled contractions and staged
target execution still require executable implementations and evidence.
In particular, a parsed `arith.truncf` does not prove it matches Triton's
backend-dependent saturating FP8 conversions.

## Native FP8 widening

The default-off `lower_fp8_widening` feature implements scalar `arith.extf`
from all four FP8 formats to f16, BF16, f32 and f64. Portable input retains
the standard cast and typed storage. At the selected initial native source
seam, the feature validates all candidates before editing, then decodes the
standard sign/exponent/fraction fields using ordinary integer operations and
exact f32 arithmetic. Source locations and nonsemantic annotations survive.

Every finite value in these FP8 formats is exactly representable in all four
destination types. Subnormal decoding uses an exact fraction and power-of-two
scale; its f32 result is normal, including the smallest E5M2FNUZ value 2^-17.
E5M2 infinities, signed zeros, finite-only top binades and unsigned-zero NaN
sentinels are distinct cases. Computed NaN class is preserved without imposing
a payload/sign choice. Uncomputed source words remain unchanged.

The current LLVM source's `arith-expand` implements other narrow types but
does not expand these four FP8 extensions. The implementation follows the
format-field approach in pinned Triton's `test_conversions.py::upcast_emulated`,
adding the special values excluded from that test's finite input population.
Format definitions and extrema are independently checked against LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`, `BuiltinTypes.td`,
`APFloat.cpp` and `APFloatTest.cpp`. It introduces no lookup table, opaque
runtime function, changed tensor/memref ABI or physical target mapping.

Scalar narrowing/arithmetic/calls, nonneutral cast flags and tensor/vector
cast candidates refuse before edits. Those forms retain separate obligations;
the feature is not an FP8 arithmetic policy. Its source wrapper consumes one
marker and forwards every ownership/runner callback. BF16 arithmetic emulation
may follow it at its own source marker; BF16 word transport remains late.
The ordinary native conversion, allocation, alias and lifetime gates still run.

When narrowing and widening are both selected, their pipeline edits now retain
one initial narrowing marker followed by one widening marker regardless of
feature application order. Narrowing must implement the standard truncations
before the widening checker inspects remaining FP8 operations. The ordinary
sorted feature application previously put widening first and the native
runner rejected this valid composition. Duplicate or moved source markers
still refuse; loop destination planning can split subsequent stages while
consuming the cast markers only once and preserving every runner callback.
This fixes feature composition without changing cast arithmetic, source
numeric contracts, buffer ownership or lifetime admission. Native cast and
runner tests cover the complete 67-case selection; actual triton2MLIR target
execution additionally exercises mixed-policy loop kernels through this seam.

Twenty-five native feature/seam/refusal cases and the existing BF16 regressions
pass (48 combined cases). triton2MLIR additionally qualifies all 48
format/destination/memory-profile combinations over every source encoding,
with independent rational references, an adjacent exact-word result and reused
Merlin libraries over empty/ragged domains and runtime source loops. Sixteen
oracle cross-checks use the independent standard codec. This is host numeric
evidence; FP8 narrowing/saturation, equal-width casts, scaled contractions,
broader models and applicable target materialization remain open.

---
title: "FP8 reader and constant bindings"
kind: reference
status: current
owner: ir
last_verified: 2026-11-10
related: [core_dialects, triton_frontend]
code_refs:
  - src/merlin/xdsl_dialects/fp8.py
  - src/merlin/common/dense_float_ir.py
  - src/merlin/frontends/linalg_mlir.py
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
dense-constant copies. Integer and restricted float constraints retain their
original meaning. This grammar/verification binding grants no permission to
execute a numeric operation or erase an effect. The parser registry is shared
with triton2MLIR when both independent readers occupy one interpreter, so
import order cannot produce different Python types for the same MLIR spelling.

`merlin/tests/ir/test_fp8_float_constraints.py` checks both reader entrances,
all four types, all 256 words in each format, scalar/dense serialization,
distinct finite/NaN/zero interpretations and nonfloat refusals. triton2MLIR
also checks both import orders and native MLIR parse/print byte preservation.
Validation uses xDSL 0.68 and the existing native SDK's xDSL 0.65, with
ml_dtypes 0.6.0. No LLVM or Triton rebuild is needed.

These checks establish reader and constant support. FP8 host buffer ABIs,
numeric cast rounding/overflow, arithmetic/scaled contractions and staged
target execution require separate executable implementations and evidence.
In particular, a parsed `arith.truncf` does not prove it matches Triton's
backend-dependent saturating FP8 conversions.

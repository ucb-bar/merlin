---
title: "Length-aware literal diagnostics"
kind: reference
status: current
owner: ir
last_verified: 2026-11-10
related: [effect_kernel_intake, lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/length_aware_diagnostics.py
  - src/merlin/common/dense_float_ir.py
  - merlin/tests/ir/test_length_aware_diagnostics.py
---

# Length-aware literal diagnostics

The explicit host feature `lower_length_aware_diagnostics` lowers standard
`cf.assert` messages without a NUL-terminated string boundary. The portable
program retains its actual condition, literal message and effect; no diagnostic
answer or invocation reference supplies compiler semantics.

Upstream LLVM at `a47bddccec30255619bb8c37fa59700e661d4e66`,
`mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp`, uses a literal
global, `puts` and `abort`. Its `test/Conversion/ControlFlowToLLVM/assert.mlir`
checks that expansion. `puts` stops at an embedded NUL. This feature instead
runs immediately before the ordinary CF conversion, after bufferization,
ownership/lifetime checks and function/arithmetic conversion. It replaces each
scalar assertion with an internal LLVM helper whose actual body checks the
condition, iterates over the complete literal UTF-8 bytes plus linefeed using
`putchar`, then calls `abort`. A true condition returns without output.

Only private literal storage reaches the helper. No source buffer is borrowed,
allocated or escaped. The complete helper body remains in native LLVM; this is
an executable expansion rather than an opaque dependency accepted as portable
coverage. Fresh internal symbols prevent helper interposition/name collisions.
Incompatible definitions or types for the C runtime declarations refuse before
edits. Source locations and provenance attributes remain on the replacement
call. The feature supports the existing host C ABI and assumes usable stdout,
as the upstream diagnostic implementation does.

The feature is off by default. triton2MLIR selects it from actual standard
assertions containing embedded NUL, not annotations, reference streams or
example values. Its owned evaluator processes disable C stdout buffering so
diagnostics survive an ensuing abort. This does not set stdout policy in
general Merlin callers.

The common xDSL parser also normalizes escaped UTF-8 attribute tokens to
`StringAttr` only when decoding is lossless. Raw dense bytes and non-UTF-8
byte attributes retain their original representations. No floating-point bit
conversion or numerical rewrite follows from this serialization adaptation.

This qualifies literal source assertions at the host boundary. General Triton
print formatting, external implementations, synchronization and accelerator
diagnostic handling need their own executable contracts.

[Scalar/string materialization](scalar_diagnostics.md) now supplies a separate
explicit software profile for standard printing operations. Physical
replication, source formatting and broader effects remain separate contracts.

---
title: "Scalar and string diagnostic materialization"
kind: reference
status: current
owner: ir
last_verified: 2026-11-10
related: [literal_diagnostics, effect_kernel_intake, lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/scalar_diagnostics.py
  - src/merlin/xdsl_dialects/standard_vector.py
  - merlin/tests/ir/test_scalar_diagnostics.py
---

# Scalar and string diagnostic materialization

The explicit host feature `lower_scalar_diagnostics` materializes standard
`vector.print` scalar/string effects after the ordinary buffer ownership and
lifetime checks, immediately before CF-to-LLVM conversion. It includes the
[length-aware assertion expansion](literal_diagnostics.md); callers select this
feature instead of also selecting `lower_length_aware_diagnostics`.

The portable program contains actual standard printing operations. Prefixes,
coordinates, signedness/hex word decoding and source loop positions remain
executable value/control flow. This feature interprets neither provenance nor
an expected output stream as semantics.

The declared software profile prints signless integers in signed decimal and
floating values with six fixed decimal places. Narrow integer scalars widen
with their actual sign, with i1 widened as 0/1; f32 widens exactly to f64.
Frontend unsigned64 and hex word formatting remain standard integer
arithmetic, loops and literal/scalar printing. Direct vector, half/BF16, index
and unsigned scalar operands require their own adapter or explicit widening.

Private internal LLVM helpers retain their complete bodies. Literal UTF-8
uses byte counts and `putchar`, including embedded NUL and literal percent
characters. Numeric helpers pass only constant `%lld`/`%f` format strings and
typed scalar values to the C varargs `printf` ABI. No source buffer or formatted
prefix reaches that varargs boundary. Incompatible runtime declarations/bodies
refuse before edits; helpers use fresh internal symbols. Locations and
provenance remain attached to replacement calls. Native LLVM verification and
ordinary compilation/linking remain mandatory.

The host runtime requires usable C stdout and the declared numeric locale.
triton2MLIR's owned evaluator processes select the C numeric locale and disable
stdout buffering so bytes survive a subsequent assertion abort. General Merlin
callers retain their own runtime/stdio policy.

LLVM `a47bddccec30255619bb8c37fa59700e661d4e66` `VectorOps.td` defines the
standard human-readable stdout effect. Its VectorToLLVM scalar conversion and
CRunnerUtils implementation provide the reuse boundary: elementary scalar,
literal and punctuation output independent of vector layout. This explicit
software profile retains fixed decimal formatting and length-aware literals
without requiring a separately built runner-utils library.

`standard_vector` supplies the missing string/punctuation fields in xDSL's
binding, checking the native field constraints. This binding changes no
native operation semantics, tensor ABI or ownership facts. triton2MLIR's
logical source-print profile abstracts physical thread replicas and interleaving
separately; this feature chooses no GPU lane, warp or physical grid mapping.

Tests cover registered roundtrips and schema refusals, real C ABI execution,
literal NUL/Unicode/percent bytes, numeric boundaries, runtime declaration
refusals, fresh helper names and the single late native seam. Direct
triton2MLIR integration also qualifies all memory profiles, exact unsigned64
and floating words, rank-two tiles, logical multi-axis grids, repeated calls,
print-before-assert order and integer CAS composition. These host observations
grant no accelerator diagnostic implementation or arbitrary external effect.

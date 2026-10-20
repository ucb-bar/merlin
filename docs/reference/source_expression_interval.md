---
title: "Reference: source expression interval tables"
kind: reference
status: draft
owner: core
last_verified: 2026-10-06
related: [architecture, verified_compiler_roadmap]
code_refs:
  - src/merlin/llvmlower/source_expression_interval.py
  - src/merlin/llvmlower/source_expression_interval_llvm.py
  - src/merlin/llvmlower/bounded_rne_maps.py
  - merlin/tests/ir/test_source_expression_interval.py
  - merlin/tests/ir/test_source_expression_interval_llvm.py
---

# Source expression interval tables

`source_expression_interval.py` provides explicit compiler utilities, with no
default rewrite or automatic workload selection. Callers supply an actual typed
binary32 expression, its complete closed integer observation, an effects
contract, and a fixed raw-word partition with a storage budget. Source arithmetic
and literal bits determine the canonical expression identity. Source names and
provenance do not determine applicability or table sharing.

## Source and observation closure

`close_scalar_i8_observer` extracts a pure unary source DAG and proves its live
endpoint is followed by two original rounded binary32 multiplications and a
complete saturated ties-to-even signed-byte conversion. The second finishing
factor must be a finite positive nonzero source literal. The existing
`prove_scalar_bounded_rne` proves the clamp, truncate, fraction, parity, sign and
increment graph. Unknown consumers and additional floating escapes refuse.

The source witness retains every scalar block operation, captured literal,
operand, result type, use list and enclosing ownership/numerical context.
`validate_closed_scalar_observer` must pass again immediately before binding a
compiled implementation. This is a numerical use closure, not a proof of
physical aliases, tensor coordinate transformations, external call purity or
compiled-code identity. Those remain caller obligations.

The effects contract explicitly requires RNE, gradual underflow, nontrapping
arithmetic and unobserved floating exception flags and signed zero. The analysis
does not infer these permissions. Unsupported runtime data and rounding modes
must retain the original source implementation; a target's mode read and ABI
guard belong in its provider.

## Full-cell enclosure

Table construction requires explicit partition width and storage budget; the
lookup emitter also requires the width. Neither utility supplies a performance
or storage default.

The table partitions the entire binary32 word space by a caller-selected prefix
of 9 through 24 bits. All suffix words in an admitted cell are covered. The
partition includes the complete sign/exponent fields; exponent zero and 255
cells refuse. Negative cells reverse numerical endpoint order. Generation reads
only the source expression and partition, with no runtime operands or golden
output.

Cartesian interval extension preserves each original rounding boundary:

- A product of binary32 endpoints has at most 48 significant bits and is exactly
  representable in binary64, including subnormal binary32 operands.
- Add, subtract, FMA and positive-denominator division take adjacent binary64
  values around the computed endpoints before monotone binary32 RNE conversion.
  This encloses the original binary32 operation, including double-rounding ties.
  FMA retains its single rounding: its product is never rounded to binary32 first.
- Exact nonzero product/addend residuals lie on a dyadic lattice of at least
  2^-298, preventing binary64 underflow in the enclosure calculation. Binary32
  ratios remain in the finite normal binary64 range.
- Integer conversion, fixed-width addition/shift and bitcast execute only at
  defined point intervals. The integer word operations retain wrap semantics.
- Nonfinite, reversed, zero-touching or zero-crossing result bounds refuse.

These rules prove enclosure by induction over the complete source DAG. Forgetting
correlation can widen a bound but cannot remove a source value. Endpoint samples
and random tests are independent regression evidence, not the full-cell proof.
The generator checks the actual host rounding mode and gradual-underflow behavior
and refuses unsupported environments without changing them.

## Observed result and physical ownership

For unchanged finite runtime operands, each rounded finishing multiplication is
monotone or reverses order. The retained saturated RNE observer is monotone. If
the original observation at both interval endpoints is the same signed byte,
every possible enclosed source result has that observation. Returning an
unobserved endpoint carrier is therefore legal. If the two observations differ,
the original source expression must execute.

`emit_source_interval_lookup` emits a portable helper that retains original
compiled scalar expression and quantizer callbacks. It emits no target ISA. The
table is immutable compiler-produced data; it cannot be generated lazily outside
a measured region. Share physical table storage only for equal canonical source
arithmetic under the same numeric contract and layout, with complete source and
compiled identity binding. Different source DAGs require different tables or an
explicit budget refusal.

All lookup loads, addressing, guard/certificate work, original fallback and
finishing operations belong in complete performance measurements. Table capacity
and logical requests do not establish physical cache misses or a whole-program
speedup. Promotion requires independently compiled source/observation tests and
the original complete-model gate before any target hardware admission.

`emit_immutable_bytes_llvm` can bind compiler-produced bytes as one readonly
constant inside the normal host module. It preserves bytes without an implicit
endianness conversion or device placement policy. The consumer must prove its
decoding layout, alignment and lifetime. This keeps ordinary constant data in
the host object without relaxing device-catalog or provider-symbol rules.

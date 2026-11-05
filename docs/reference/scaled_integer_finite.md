---
title: "Reference: broadcast scaled integer finite guards"
kind: reference
status: draft
owner: core
last_verified: 2026-10-07
related: [source_expression_interval, architecture, verified_compiler_roadmap]
code_refs:
  - src/merlin/llvmlower/scaled_integer_finite.py
  - src/merlin/llvmlower/scaled_integer_finite_llvm.py
  - src/merlin/llvmlower/source_expression_interval.py
  - merlin/tests/ir/test_scaled_integer_finite.py
  - merlin/tests/ir/test_scaled_integer_finite_llvm.py
---

# Broadcast scaled integer finite guards

These explicit utilities retain the original signed integer conversion and
ordered binary32 constant products before a final broadcast scale multiply.
They select no workload, target, profitable schedule or default numeric policy.

`prove_broadcast_scaled_integer_finite` derives the full signed integer domain,
source constant words, exact tensor projection and broadcast repetition from
typed IR. Rational arithmetic bounds each original rounded intermediate
separately; it never factors or reassociates source products. A conservative
raw-word threshold admits finite scales whose final product cannot overflow.
NaN, infinity, overflowing constant chains and unsupported source contexts
refuse. Complete source operations and ancestor numeric contexts are retained.

`bind_finite_scale_helpers` closes those proofs against actual scalar LLVM
conversions, constant products, scale loads and plain borrowed pointer helpers.
It proves scale indices with zero-seeded increasing loop induction, dominating
header bounds and checked integer arithmetic. Unknown effects, pointer escapes,
input stores, output initializer reads, strict FP, fastmath and unbounded scale
loads refuse. The current grammar accepts static one-dimensional channel scales;
unsupported layouts require a separate proof and retain ordinary lowering.

## Ownership and execution contract

The caller must explicitly establish valid immutable input spans and a fresh
output disjoint from every input through the normal buffer ownership contracts.
Readonly inputs may alias each other. Source identity or pointer equality does
not supply these ownership facts. Inputs must remain unchanged until the
synchronous consumer finishes; no cache keyed by addresses is introduced.

`emit_finite_scale_helper` emits portable integer-bit scans followed by the
finite-input helper, or the unchanged original helper on refusal. Both scans,
all subsequent source computation and output publication belong in measured
costs. Scan code performs no floating operation or environment change. Stable
RNE and gradual underflow remain explicit caller/provider obligations; target
mode checks and ABI glue stay in the selected provider.

The optional `finite_inputs` argument of `emit_source_interval_i8_lookup`
requires validated helper proofs plus the explicitly supplied immutable
`finite_table` with identical source expression and partition. Selected route
factors must match unique typed observers. It removes only corresponding input
finite checks. Every table-validity check, original constant and rounded
finishing operation, ambiguity decision and cold source continuation is
retained. The lookup may only be called through its bound successful scans;
unsupported modes or refused scales retain the original complete source helper.
The existing nontrapping/unobserved-effect and closed integer observer contracts
remain mandatory. This grants no approximate arithmetic or floating escape.

Empty selection preserves the ordinary emitted lookup bytes. Actual scanner
traffic, register/frame changes and complete helper costs determine whether this
explicit preparation is profitable; saved checks alone do not predict cycles.

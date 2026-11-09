---
title: "Explicit approximate source roundoff"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [independent_scalar_lanes, prepared_operand_owners]
code_refs:
  - src/merlin/llvmlower/source_roundoff_policy.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/source_rms_roundoff_estimate.h
  - src/merlin/runtime/native/c/source_rms_point_products.h
---

# Explicit approximate source roundoff

`ApproximateSourceRoundoffPolicy` separately authorizes a fixed four-sigma
estimate for source ordered binary32 FMA roundoff. The normal source attention
emitter accepts it only through the explicit `source_roundoff_estimate` argument,
with complete encoded-row and exact source-point producer contracts. `None`
retains the original C bytes. Source semantics, types, constants and effects
still determine eligibility; no workload, captured input or golden selects it.

The estimate multiplies the deterministic source-roundoff factor by
`min(1, 4/sqrt(3*K))`. It retains deterministic subnormal allowance and original
absolute-prefix overflow safety. Only finite, exactly reconstructed source rows
qualify. Representation errors and input-interval uncertainty are never reduced:
unknown or unequal rows use the unchanged rigorous bound producer. Refused
providers must use the retained original source writer before publishing output.
Nonlinear/mask/reduction semantics and original source replay order are unchanged.

This statistical estimate is **not a rigorous enclosure**, exact consumer theorem,
probability guarantee or whole-output error budget. Correlated or biased errors
can fall outside it. The typed permission explicitly requires approximate output
to be permitted, independent original-output validation, stable RNE, nontrapping
execution, unobserved exception flags, retained source fallback and unscaled
representation error. Core does not select an accuracy threshold or deploy a
model automatically. Original model accuracy criteria remain authoritative.

## Normal provider binding

Select this policy through the ordinary portable source emitter. Bind the emitted
C/object identity, complete typed source DAG, numerical permission, effects and
retained source fallback using the existing explicit closed-group writer and
host-provider APIs. Record `policy.numerical_policy` as the approximate route.
Do not use `ConsumerObservedGroupPreparation`: that exact-observation binder
rejects this policy because estimates cannot prove every source integer/scale
observation. Target products, descriptor wrappers and ISA capability remain OOT.
No default model route is enabled by installing these utilities.

Independent runtime tests cover unequal dimensions, tails, exact/nonpoint
representation, zeros/subnormals, cancellation safety, nonfinite/overflow refusal
and unavailable rounding modes. Complete normal-emitter finite fixtures retain
source-output validation and dirty-buffer/input guards. Those finite passes and
historical experimental model gates do not establish an all-input theorem.
Price all preparation, norms, estimates, product readout, replay and output
traffic before using the alternative; local instruction or group timing results
do not predict whole-model performance.

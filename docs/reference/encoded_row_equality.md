---
title: "Encoder row equality"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/encoded_row_equality.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/encoded_row_equality.h
---

# Encoder row equality

The optional `prepare_encoded_rows` frontier feature requires admitted norm
requirements. It integrates proof generation into the mandatory reconstructed
binary32-to-binary64 write, rather than making a second equality pass.

For each contiguous row it records finite numeric source/reconstruction equality
and, for uncertain operands, equality of both source-envelope endpoints to that
source value. Signed zeros may compare equal because the reused quantities are
absolute-value norms and zero representation error. This does not prove ordered
source FMA exactness: its rounding bounds and replay remain mandatory.

The typed result binds exact source, reconstructed, lower and upper spans, row
length and flags. All are private disjoint owners, regenerated on every encoding,
unchanged until synchronous bounds consumption. The emitted grammar checks this
write/use chain. Pointer equality alone grants no cache or lifetime permission;
source mutation invalidates the proof and requires a new encoding.

Matching rows preserve the original source norm accumulation order and reuse its
L1 or complete admitted norm by value. The representation-error norm is exactly
zero. Nonfinite, inexact, uncertain or identity-mismatched rows retain the original
checked scans. Column proofs omit uncertainty spans and can skip only the zero
representation-error scan.

The option defaults off. Private workspace gains `query_rows + chunk` flag bytes
plus structure padding. Callers must query and bind the new compiled size; old
capacity must be refused, never silently reused. Complete measurements include
proof generation, flags initialization, branches, allocation and traffic.

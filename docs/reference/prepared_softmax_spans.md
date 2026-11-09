---
title: "Private softmax producer spans"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/llvmlower/prepared_softmax_spans.py
  - src/merlin/runtime/native/c/prepared_softmax_spans.h
---

# Private softmax producer spans

`prepare_softmax_spans=False` leaves the generated source unchanged. Opting in
requires both the prepared softmax domain and the admitted source-radius
producer. This carries evidence from successful source-bound interval production;
it does not change any interval, floating operation, source replay or observation.

The source-bound generator closes the complete mask gathering, two tile
production/copy, and immediate consumer region. Mask gathering rejects bytes
outside 0/1. Each successful bound output is finite and ordered:

* The checked gamma path calls `merlin_fma_bound_finish`, which refuses nonfinite
  or overflowing endpoints before outward binary32 conversion.
* The private product-row path admits finite immutable reconstructed centers,
  source norm/error bounds and a finite prefix/final envelope below `FLT_MAX`.
* The separable-radius path admits exact private products, nonnegative finite
  source norms and a finite complete row envelope below `FLT_MAX`.

The latter two rely on the existing exact product callback/overflow and private
storage contracts, not a newly trusted public flag. Their helper implementations
are included in the provider dependency seal. Every output address is covered
by the original row/tile copy. No successful early exit or direct extra bound
write is admitted by the generator. Unknown producer grammar is refused.

A fresh local epoch and a typed span value record the same lower/upper/mask
owners, dimensions and sequential tile completion. Incomplete, duplicated,
out-of-order, mismatched-address/epoch/size or reused evidence is refused. A
failed consumer match retains the original checked span scan. The evidence is
single-use, lives only across one synchronous head setup, and is never cached.
All private arrays are disjoint; product callbacks cannot mutate the mask or
published bound spans. Later exact-dot refinement remains checked inside its
original enclosing interval, preserving finite/order invariants. Stable source
RNE, nontrapping arithmetic and unobserved exception flags remain required.

The option saves a redundant scan only. It does not claim a hardware or whole
model speedup. Complete allocation-aware group cost and original model accuracy
qualification are separate release gates.

`prepared_softmax_spans.prepare_softmax_produced_spans` validates and rewrites
the owned producer region. The runtime header with the same name records and
consumes tile coverage. Neither layer accepts arbitrary caller-supplied intervals
as completed producer evidence.

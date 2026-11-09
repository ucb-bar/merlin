---
title: "Four-cell prepared polynomial scheduling"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/llvmlower/prepared_polynomial_batch.py
  - merlin/runtime/c/prepared_polynomial_batch.h
---

# Four-cell prepared polynomial scheduling

`emit_source_attention_frontier(..., polynomial_batch_four=True)` is an explicit,
default-off scheduling option for the prepared word-space softmax enclosure.
It requires the existing immutable source polynomial plan, source RNE,
nontrapping arithmetic and unobserved exception flags. Unsupported intervals or
plans use the existing scalar checked helper.

The portable helper stages four independent intervals and eight endpoints. Each
endpoint retains its source multiply, floor, subtract, three rounded Horner
FMAs, subtract, encoded FMA and integer conversion. Point/cutoff cases retain
scalar results; speculative discarded arithmetic is limited to the already
proved polynomial domain. It does not reassociate arithmetic or widen bounds.

The consumer computes these independent intervals after maximum refinement,
then processes cells in their original order. Exact replay, BF16 bin decisions,
per-lane denominator subsequences and the final tree remain unchanged. Tails
use only their active cells; dummy private lanes do not escape. Private scratch
is automatic storage, with no change to the caller workspace ABI.

Portable C expresses independence but does not promise target issue order.
Inspect actual compiler output and price the entire group, including scratch
and replay, before selecting this policy. A portable four-cell schedule does not establish an eight-cell cost policy;
selected compiler output and complete measurements must justify any wider
alternative.

The emitter adapter `prepare_polynomial_batch_four` checks the complete original
cell loop and tail before replacing it. The runtime arithmetic lives in
`prepared_polynomial_batch.h`; target-specific eight-endpoint FMA hooks, when
separately selected, remain provider capabilities rather than a new cell schedule.

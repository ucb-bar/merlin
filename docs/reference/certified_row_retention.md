---
title: "Retaining completed observation certificates"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - merlin/runtime/c/templates/source_attention_endpoint_prepared.c.in
---

# Retaining completed observation certificates

`retain_certified_rows=True` is an explicit source-attention executor option requiring `prepare_endpoint_rows=True`. The default generated source stays identical. It retains private endpoint values for rows whose complete quantized-output and escaping-scale observations have already been certified.

The generated lifecycle supplies the proof:

1. Every invocation resets its private row flags to zero, including reused workspace.
2. The first endpoint pass computes every row's lower/upper bound and carrier output before any flag becomes true.
3. Only a successful complete row observation certificate marks that row true. Flags never revert during the call.
4. Later refinement traversals first skip flagged rows. The denominator writer affects `den[row]`, source probabilities at `row * keys + key`, and the row's exactness flag. The partial writer affects `((tile * parts + part) * rows + row) * depth + column`. Neither writes another row. Alpha and center inputs remain immutable throughout certification.
5. Endpoint evaluation is row-local: its only reductions are the fixed source tile/part order within one row and column. It reads no other row or global accumulator. Skipping a certified row therefore preserves exactly the values repeated evaluation would produce.
6. Only complete success publishes the fresh public result. Refusal still invokes the caller's retained original source; no partial result is published. Reused workspace starts a new lifecycle.

The flag pointer is internal to the source-proved executor, not a public claim that arbitrary outputs are initialized. Generated structure checks refuse changed lifecycle/endpoint patterns. Native tests exercise retained initialized rows alongside independently updated unresolved rows, mixed masks, all-masked input, dirty reused workspace, descriptor errors, refusal, source oracle comparison, and explicit option validation. Complete original-group and whole-model evidence remains separately recorded. No hardware gain follows from the row census alone.

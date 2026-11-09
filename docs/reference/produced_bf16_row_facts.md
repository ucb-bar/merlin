---
title: "Owned BF16 producer facts"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [fused_encoded_witness, prepared_probability_bins, ordered_fma_certificates]
code_refs:
  - src/merlin/llvmlower/produced_bf16_row_facts.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/source_rms_produced_maximum.h
---

# Owned BF16 producer facts

`emit_source_attention_frontier` accepts a separate default-off
`ProducedBF16RowFactsContract` through `produced_bf16_row_facts`. The contract
requires complete finite BF16 producer writes, complete source/consumer use
closure, private disjoint storage, immutable source/metadata lifetime, explicit
prepared RHS epochs/consumer quotas, original encoded equality, original scalar
DAG/fallback and standard unobserved representation copies. Stable RNE,
gradual underflow, nontrapping arithmetic and unobserved exception flags remain
mandatory. These caller facts are not inferred from shapes or sampled values.

The original Q/RHS BF16 gather and both exact probability producers collect row
maximum/minimum magnitude words while performing their existing writes. Finite
BF16 word ordering matches widened absolute magnitude, including either sign,
signed zero and subnormals. The original radix step, coefficient RNE/saturation,
digits, reconstruction and equality flags remain; the selected private encoder
consumes these completed facts instead of scanning the row again. Unknown facts
retain the original scanning path. Unknown emitter grammar or insufficient dead
private storage refuses the modification.

The option requires exact probability point, prepared RHS and fused encoded
witness composition. Metadata occupies already dead private float storage;
dimensions must prove sufficient space. Runtime allocation queries remain the
actual compiled sizes. Physical RHS epochs, ordinals, quota, descriptor
invalidation and callback failure before public publication are retained. Epoch
addresses alone do not prove byte immutability or authorize fabricated metadata.

If the caller separately selected `ApproximateSourceRoundoffPolicy`, the original
RMS4 column/global maximum may consume the same immutable producer facts after
every original reconstructed equality check. False equality flags retain the
original rigorous fallback. The optional `source_rms_produced_maximum.h` adds
that consumer without changing the default public helper. No source radius,
overflow check, original arithmetic, numerical policy or observation is changed.
Producer-fact forwarding grants no approximate-output policy itself.

Complete integer product families, finite-point observations and exact BF16
integer observations retain their independent contracts and compose before fact
forwarding. All live integer output planes, source reconstruction and unchanged
BF16/scaling/product DAG remain required. The modifier does not choose target
commands, memory banks, kernels, model names or provenance IDs.

Producer/write/use/effect proofs, storage and source forwarding belong Merlin.
Target ISA, resource placement, commands and physical ABI belong the OOT dialect.
The current emitter option is explicit; automatic current-IR producer discovery
and profitability remain separate. Measure complete producer, metadata, packing,
norm/certificate, transfer, consumer and fallback costs under the original whole
output gate. Retired instructions and isolated helper savings do not establish
FPGA cycles or a whole-model gain.

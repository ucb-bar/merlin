---
title: Checked alignment for exact paired readouts
kind: design
status: current
owner: core
last_verified: 2026-11-09
related: [architecture]
code_refs:
  - src/merlin/llvmlower/enclosed_readout.py
---

# Checked alignment for exact paired readouts

The exact paired-readout decoder observes two immutable producer readout bytes
and updates the first span with the complete certified source result. Equal
eight-byte packets need no stores. The producer-domain certificate, complete
readout semantics, count bounds, disjoint spans and stable second input remain
caller obligations.

The explicit `checked_alignment=True` option requires `compiler_builtin` copies.
For a complete packet, actual pointer values must both be divisible by eight
before the emitter supplies a compiler alignment assumption. All other pointer
combinations retain byte-safe copies. Packet bounds and the original scalar
tail prevent accesses beyond count, including an inaccessible following page.
Copies into integer locals preserve byte-storage effective types; no integer
lvalue aliases the input arrays. The second span can reside in read-only memory.

No floating arithmetic, rounding-mode permission or new flags occur in the
decoder. Unknown pairs still trap. This is a reusable host code generation
option; target instruction/resource selection remains provider-owned. Default
emission is byte identical. An actual target object and complete producer,
readout, fence and decoder measurement determine profitability; an instruction
count reduction alone supplies no hardware or whole-model timing claim.

`enclosed_readout.prove` constructs the complete paired-observation certificate;
`emit_pair_scan` consumes it and emits the selected copy/alignment policy.
The emitted decoder does not establish that a physical device implements those
readout functions or that its commands have completed.

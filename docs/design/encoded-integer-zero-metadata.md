---
title: Exact zero metadata for encoded integer panels
kind: design
status: current
owner: compiler
last_verified: 2026-10-06
related: []
code_refs:
  - src/merlin/llvmlower/encoded_i8_zeros.py
  - merlin/tests/ir/test_encoded_i8_zeros.py
  - src/merlin/llvmlower/encoded_radix_reconstruct.py
  - merlin/tests/ir/test_encoded_radix_reconstruct.py
---

# Encoded byte zero summaries

An omitted integer product must be justified by the **stored encoded values**.
The original floating dtype, a quantization precision, source identity, or a
previous input cannot prove a current integer panel is zero.

Merlin supplies complete immutable reference summaries and C producer joins.
The consumer supplies a positive logical row block extent. Metadata is plane
major, with one boolean for each complete or partial logical row block. Both
row-major `[plane,row,K]` and transposed `[plane,K,row]` storage are supported.

A producer first clears every summary entry. While storing an encoded row, it
ORs unsigned casts of **every actual stored signed byte**, then joins whether
that OR is nonzero into the appropriate panel. Values such as `-128` are
nonzero. A zero summary thus proves all encoded bytes of that logical panel are
zero, including reduction and row tails. The independent verifier rescans
storage and rejects both stale input and changed metadata.

The producer and consumer must share the same complete extents/layout. Encoded
inputs and metadata remain immutable together during use and do not overlap
the writable output span. Nonoverflowing absolute-value conversion preserves
the zero summaries; it does not justify any additional numerical assumptions.

The provider owns hardware panel widths, resource placement, skipped command
emission and initialization. If all terms of an output tile are omitted, it
must still initialize and write every logical output. Dirty destinations and
previous accumulator contents cannot become an implicit zero seed.

This utility enables no automatic policy or model routing. Producer joins,
runtime branches, initialization and output readback must be included in
performance qualification. A lower command count alone proves no latency gain.

## Exact weighted reconstruction

An explicit companion emits support-aware binary64 updates for a canonical
radix product plan. A panel is supported when any of its integer product pairs
has nonzero panels on both sides. An unsupported panel's exact integer group is
zero, so the host can omit its source read and positive-zero weighted update.
Fully supported pairs retain the flat original traversal.

This requires RNE, a positive-zero seed, complete and exact i32 group outputs,
the signed-digit magnitude bounds, and at most the plan's reduction-length
terms. The existing absolute weighted bound proves every term and prefix is an
exact integer in binary64. Exact cancellation yields positive zero under RNE,
so omitted positive-zero additions preserve the raw result bits. This grants
no permission to eliminate zero additions from arbitrary floating code.

The emitter rederives the entire canonical range proof and refuses changed
plans. Tests compare every prefix as raw binary64 bytes, including unequal
logical block sizes, tails, all-zero groups, explicit cancellation, and the
largest reduction length accepted near the binary64 precision boundary.

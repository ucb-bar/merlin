---
title: "Fused canonical encoder witness"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/fused_encoded_witness.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/bf16_radix_pack.h
---

# Fused canonical encoder witness

The optional `fuse_encoded_witness` provider feature requires the existing
encoded-row source/point equality contract. It specializes the canonical radix
encoder's output consumer while preserving its finite BF16 scan, coefficient
rounding, planes, power-of-two step, and binary32 reconstruction multiplication.
That exact f32 value is widened immediately to the existing private double
buffer. Equality and optional point-envelope flags are derived while the source
word remains live. There is no new numeric approximation.

The helper is generated from the canonical encoder body and refuses changed
producer grammar. Finite reconstruction follows from at most21 coefficient bits
and the admitted step exponent: the maximum is strictly below2^128 and exactly
representable in binary32; accepted subnormals are integer multiples of2^-149.
Numeric equality preserves the prior treatment of opposite signed zeros. The
reconstructed value's own sign bits and all planes match the previous encoder.

Source and optional endpoint spans are immutable through synchronous encoding
and consumption; planes, reconstruction and flag storage are disjoint private
owners. Each call overwrites all rows before publishing its witness, which
cannot survive a new encoding epoch. The caller's original source fallback
remains mandatory on failure. Public checked encoder and widening APIs are
unchanged. Existing float scratch remains allocated but has no consumer in the
selected route, preserving the private workspace size and external ABI.

The selected rewrite removes redundant scanning and scratch traffic. Removing
a separate widening scope is not the same as removing all of its cost: f64
stores and point comparisons remain. Complete source-consumer and whole-model gates plus
measured full provider cost govern promotion.

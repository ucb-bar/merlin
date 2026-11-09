---
title: "Consumer-derived reconstruction norm requirements"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - merlin/runtime/c/dot_norm_requirements.h
  - merlin/runtime/c/prepared_dot_norms.h
---

# Consumer-derived reconstruction norm requirements

`emit_source_attention_frontier(..., prepare_required_norms=True)` is opt-in and requires producer-domain preparation. Defaults preserve emitted source and actual default target object/ELF bytes.

The emitted source has two reconstructed-LHS norm consumers. The representation-error term needs L2 only when the admitted RHS error L1 is nonzero. Producer-domain admission uses only reconstructed L1. Successful immutable RHS error admissions with L1, maximum and L2 all exactly zero therefore prove the first consumer is bypassed for every column. A distinct `merlin_l1_norm` accumulates the identical upward L1 sequence and is accepted only by the typed L1 producer-domain function; it cannot be passed to the full-norm product helper. Nonzero, unknown, or invalid RHS metadata retains full norm preparation. No partially initialized full norm object is published.

The actual error-producing rows, norm metadata and reconstructed products remain private, disjoint and immutable through the synchronous use. Source FMA order, gamma/overflow checks, representation-error branches, centers, final enclosure and consumer quantization are unchanged. Existing stable RNE/gradual-underflow and nontrapping, unobserved exception flags remain required; omitted square/root operations need not preserve exception flags.

Independent tests exercise exact and nonzero reconstruction error, uncertainty, overflow, invalid values, missing metadata, zero/signed-zero/subnormal domains and unsupported rounding modes. Complete captured-group and whole-source evidence is recorded separately; observed zero-error frequency never selects a production policy by workload identity.

---
title: Shared explicit tensor permutation proof
kind: reference
status: current
owner: core
last_verified: 2026-10-06
code_refs: [src/merlin/llvmlower/shared_permutation.py]
---

# Shared explicit tensor permutation proof

`prove_shared_transpose(inputs, logical_shape)` checks typed `linalg.transpose`
producers, positive static unencoded tensor shapes, destination and element
identity, and identical axis permutations across every operand. It returns the
physical shape, logical shape, and permutation as ordinary proof data.

The caller must separately prove uniform elementwise arithmetic. This helper
alone does not authorize moving a reduction, per-axis quantizer, or other
axis-dependent operation. It performs no IR mutation and never consumes
transpose fanout. It has no tile, target, rank, or integer-element restriction;
implementations retain their own ABI and resource restrictions.

Unknown producers, dynamic or empty axes, encoded tensors, mismatched element
types, and differing axis identities fail closed. Equal dimensions do not make
different permutations interchangeable. No pipeline enables this proof or a
rewrite by default.

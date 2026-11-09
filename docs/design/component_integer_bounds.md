---
title: Exact integer source bounds for independent component DAGs
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-11-09
code_refs:
  - src/merlin/experiments/phase0/component_integer_bounds.py
  - src/merlin/experiments/phase0/component_numerics.py
  - src/merlin/experiments/phase0/writer.py
---

# Integer DAG arithmetic admission

The component numerical reference implements standard unflagged MLIR integer
operations using modular projection. Previously, a component DAG could carry
a selected `bounded_exact` contract while its writer reported the partial-sum
bound as `not_applicable`. A final correct or canceled result did not establish
that the intermediate arithmetic satisfied the selected contract.

The normal generator now performs a pure source interval preflight before
capture, staging, palette realization or reference allocation. The writer
repeats that check before the ordinary builder. The reference replays the bound
against the actual typed source before constructing leaf tensors. Numerical
arithmetic, operation order, exact comparison and tolerances remain unchanged.

## Selected arithmetic

Direct integer DAGs require the selected independent `integer_reference`
engine and an explicit arithmetic choice:

- `bounded_exact`: every source node product, reduction prefix and result must fit.
- `modular_wrap`: the selected source explicitly admits modular integer arithmetic.

Saturation, unknown choices and an ambiguous legacy `wrap_internal_mac` without
a selected full-operation policy refuse. A separately declared
`bounded_exact_requires_each_partial_sum` policy also checks its explicit signed
operand and MAC result widths. Missing or unknown declared widths refuse.
These widths are selected numerical declarations; the proof does not establish
that the hardware implements them.

## Pure interval derivation

Input intervals come from the canonical stimulus range or the exact selected
palette alphabet. Only the alphabet is resolved; no shaped tensor is realized.
All input values must fit their actual signed source type. Shape and dtype
resolution use the existing checked component program analyzer.

For matmul, the four interval endpoint products bound every multiplication.
Multiplying this interval by the actual reduction extent bounds the completed
sum. Including zero gives a conservative bound on every possible reduction
prefix. Each product and prefix must fit the source result width and any
separately selected declared MAC width. Cancellation never discounts this bound.

Add and update propagate endpoint sums and check each resulting source width.
Copy, transpose and alias preserve the interval. Actual functionalized SSA
operands select the correct logical epoch, so later updates cannot rewrite the
bound of an earlier snapshot. A signed-width comparison uses integer bit
lengths and does not allocate an enormous shifted bound merely to check a type.

The proof is sufficient and conservative. It ignores correlations, exact tensor
positions and cancellation and can refuse a safe program. There is no success
fallback based on a small final result. Unsupported semantics remain unavailable
until their independent proof owner is implemented.

## Receipts and scope

The generated capsule and complete golden carry the same versioned proof, with
typed source and selected numerical-semantic commitments. Current generation
checks its presence. v2 coverage replay requires it and verifies that the
arithmetic choice still matches the protected selected software digest. Saved
proofs cannot authorize altered sources or an easier overflow policy.
Historical v1 records remain readable without a new bound claim; fresh Phase1
requires budgeted v2 evidence.

Regressions run the ordinary generation path: valid bounded DAGs publish all
independently recomputed outputs; an overflowing prefix is refused even when
the final sum is zero; a wide result cannot hide a narrower declared MAC
overflow; a later cancellation cannot hide an overflowing add. The same source
is accepted under an explicitly selected modular contract. Unknown arithmetic,
missing widths, changed policy and stale proof refuse before numerical data.

This proof establishes selected source arithmetic for the bounded input domain.
Hardware correspondence, physical arithmetic, device execution, synchronization,
large-shape compiler legality and timing require separate qualification. The
execution budget separately limits logical reference work and payload; it does
not establish numerical exactness by itself.

The current source analyzer also supports `reduce_sum`: the complete reduction
extent bounds every prefix, including zero, before checking the selected result
width. `concat` joins operand intervals; copy, transpose, reshape, alias, cast
and extract-slice propagate the checked source interval and still check the
result width. These paths do not infer physical hardware behavior or relax the
selected bounded-exact/modular distinction.

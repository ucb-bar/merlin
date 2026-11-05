---
title: "Host integer primitive source witnesses: what the shift and mask probe proved"
kind: design
status: current
owner: core
last_verified: 2026-10-08
related: [verified_compiler_roadmap]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_literal_arange.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_literal_arange_admission.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_host_source_dispatch.py
  - src/merlin/frontends/linalg_patterns.py
  - examples/workloads/host_control_math/loader.py
---

# Host integer primitive source witnesses

A host operation is admitted only through a closed source-body schema and its own mandatory
witness (`private_host_source_dispatch.py` routes each schema to one). A 2026-10-06 diagnostic
(commit `cb0d40a09`, `host_primitive_source_proof.py`) tried a wider witness for three i64
primitives on a frontend trace plus its lowered MLIR: `arange`, arithmetic right shift by a
literal (`x >> c`), and bitwise-and with a literal (`x & c`). It was never wired to an
admission path. This page records what it proved, what it left open, and why it was not kept.

## What it checked

* **Source side.** The frontend trace had to be complete. Its typed argument references had to
  equal the recorded edge roster exactly, with no cycles. Each selected node had to produce one
  positive rank-one i64 tensor. `arange` needed literal start, end and step with a nonzero step,
  a length that agrees with the result shape, and no element outside i64. The shift count had to
  lie in `[0, 64)`, the range where `arith.shrsi` is defined. The tensor operand of a shift or
  mask had to be produced by a typed original node of the same graph, dtype and shape.
* **Lowering side.** Each primitive had to join one-to-one, through `prov.origin_node_ids`, to a
  `linalg.generic` whose body is exactly the approved form. For `arange` that is `index`,
  `index_cast`, `constant`, `muli`, `constant`, `addi`, `yield`, with the same literals and no
  overflow flags. For a shift or a mask it is `constant`, then `shrsi` or `andi`, then `yield`,
  with the source literal. The maps had to be identity, iteration parallel, and no other
  semantic attribute present.
* **Dead sources.** A selected node with no lowering had to have an exact dead-use proof: an
  explicitly pure ATen operation all of whose users are themselves proved dead.

The probe workload added `arange_step` and `shift_mask` cases (the full i64 range, including
`-(2**63)` and `2**63 - 1`) and wide-range `sine_range` and `cosine_range` inputs up to `1e8`.

## What it left open

Its own result said `source_literal_body_verified_not_host_approved`, and it listed two
equivalences it did not prove:

* **original tensor to lowered operand.** A test swapped the outer input of a shift for another
  value of the same shape, and the witness still passed. It binds literals and local bodies, not
  data flow into the body.
* **lowered result to program output.** Nothing tied the lowered result to the value the program
  returns.

## Why it was not kept

* **`arange` is covered.** `private_literal_arange.py` checks the same literal i64 lowering body
  on the prepared graph, and `private_literal_arange_admission.py` joins it to admission.
* **Shift and mask have no admission path to serve.** The static pointwise schema
  (`linalg_patterns.py`) has no i64 `shrsi` or i64 `andi` with an in-body literal, and no target's
  host capabilities request one. A witness for them should be built as a closed source-body
  schema with its witness under the dispatch, and it has to prove the two open equivalences
  above. A diagnostic that cannot detect a swapped operand cannot admit anything.

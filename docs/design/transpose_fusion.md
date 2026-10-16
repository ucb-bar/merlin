---
title: Fuse a transpose into a contraction access map
kind: design
status: current
owner: compiler
last_verified: 2026-10-12
related: [whole_model_op_profile, normal_search_feedback, compiler_plane]
code_refs: [src/merlin/llvmlower/transpose_fuse.py, src/merlin/llvmlower/pipeline.py, src/merlin/llvmlower/impr_features.py]
---

# Fuse a transpose into a contraction access map

A contraction can consume a transposed operand without materializing a separate
buffer when its access map can express the same element selection. The optional
`fuse_transpose_b` feature applies this rewrite before native lowering.

## Rewrite and preconditions

For a supported `matmul(A, transpose(W, permutation))`:

1. Replace the contraction's transposed operand with the transpose source.
2. Compose its operand indexing map with the transpose permutation.
3. Erase the transpose only when it has no remaining users.

The source and replacement access maps must select the same logical elements.
Check rank, permutation, types and uses; do not extend admission to arbitrary
reshape chains, batching or unsupported layouts by analogy. An escaped consumer
keeps its materialization. The rewritten contraction remains an ordinary
`linalg.matmul` for subsequent target-independent passes.

The rewrite changes representation and movement, not scales, rounding, reduction
order or the permitted output error. Validate the unchanged numerical contract
through the selected complete outputs. Shape and indexing correspondence are
preconditions; they do not establish target execution or performance.

## Evaluate the effect

Inspect before/after IR and the actual linked artifact to confirm whether a
materialization disappeared. Measure both the contraction and complete program
boundary where available: changing access order can trade buffer traffic for
less favorable operand locality. Compare controlled alternatives and tails,
including an additional consumer, rather than selecting by model identity.

Profiling observations can identify the materialization's cost. They do not
prescribe fusion or prove profitability for another target or shape.

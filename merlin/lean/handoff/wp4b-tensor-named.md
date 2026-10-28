# Handoff: tensor ops and named linalg ops

> **Historical (2026-10-15).** A work-package handoff from before the code landed; its status
> lines are out of date. The current state is the code under `merlin/lean/` and the design doc.

Scope: `Sem/Ops/Tensor.lean`, `Sem/Ops/LinalgNamed.lean` (+ helpers under `Sem/Tensor/`,
`Sem/Named/`; tests `Tests/{Tensor,LinalgNamed}.lean`). Spec: design §4.3.

## Status

- **Implemented: nothing.** Both files are still the G0 stubs (every op `stuck`). Work was stopped
  during source research; no Lean was written, so nothing is proved and there are no tests.
- Named-op map table: not started.

## Research results (reuse these)

- MLIR pin: `docs/guides/llvm_toolchain.md` pins `a47bddccec30`; `~/Projects/llvm-project` is at the
  same commit. A matching `mlir-opt` is at `~/Projects/llvm-project/build/bin/mlir-opt` — use
  `--linalg-generalize-named-ops --mlir-print-op-generic` to cross-check each transcribed map.
- Ops present in this revision:
  - TD ops (`Linalg/IR/LinalgStructuredOps.td`): `generic`, `map`, `reduce`, `transpose`,
    `broadcast`, `elementwise`, `matmul`, `contract`, `batch_matmul`, `batch_reduce_matmul`.
    `matmul`/`batch_matmul` take an optional `indexing_maps` attribute (transpose/broadcast variants).
  - **`matmul_transpose_a/b` and their batch variants do not exist** at this revision.
  - OpDSL ops (`Linalg/IR/LinalgNamedStructuredOps.yaml`, from
    `python/mlir/dialects/linalg/opdsl/ops/core_named_ops.py`): `copy`, unary/binary elementwise
    (`exp log abs ceil floor negf reciprocal round sqrt rsqrt square tanh erf add sub mul div
    div_unsigned max min powf select`), `matvec vecmat batch_matvec batch_vecmat dot`,
    `quantized_matmul`, `quantized_batch_matmul`, `mmt4d batch_mmt4d`, `conv_1d/2d/3d`,
    `conv_1d_{nwc_wcf,ncw_fcw}`, `conv_2d_{nhwc_hwcf,nhwc_fhwc,nchw_fchw,ngchw_fgchw,ngchw_gfchw,
    nhwgc_gfhwc}` (+`_q`), `conv_3d_*`, `depthwise_conv_{1d,2d,3d}_*`, `pooling_{nhwc,nchw,nwc,ncw,
    ndhwc}_{sum,max,min}` (+`_unsigned`), `fill`, `fill_rng_2d`.
- Tensor ops (`Tensor/IR/TensorOps.td`): bitcast, cast, concat, dim, empty, extract, extract_slice,
  from_elements, gather, generate, insert, insert_slice, rank, reshape, expand_shape, collapse_shape,
  pad, parallel_insert_slice, scatter, splat, yield.
- Slice facts:
  - Dynamic sentinel is `ShapedType::kDynamic` = INT64_MIN in `static_offsets/sizes/strides`
    (exported as `Attr.denseArray`) → `stuck`.
  - Dropped dims: `getDroppedDims` (`lib/Dialect/Tensor/IR/TensorOps.cpp:142`) scans sizes from the
    back; a static-1 size is kept iff the current reduced-shape dim is 1, otherwise dropped. The
    verifier uses `computeRankReductionMask` (`lib/IR/BuiltinTypes.cpp:544`, greedy front scan).
    Semantically the choice is irrelevant: dropped dims have size 1 (index = offset), non-unit dims
    are all kept in order. Implement the exact `getDroppedDims` rule and also check kept sizes equal
    the result shape; a mismatch is `stuck`.
  - Static bounds: `verifyInBoundsSlice` (`lib/Interfaces/ViewLikeInterface.cpp:37`) requires
    `offset < dim` and `offset + (size-1)*stride < dim`; violation is invalid IR → `stuck`.
  - `insert_slice` uses the same mask against the source shape (`TensorOps.cpp:3219`).

## Next steps

1. Rebase to pick up WP-4a's `Linalg.genericSem` and shared helpers (`Sem/Support.lean`,
   `Sem/Typing.lean`) if present; otherwise write private helpers.
2. `Tensor.lean`: empty (all poison), extract/insert, from_elements/splat, dim, cast, bitcast,
   extract/insert_slice, collapse/expand_shape (row-major linearization), concat, pad/generate
   (evaluate region per index with `index` block args at `ctx.indexBits`). Dynamic → `stuck`.
3. `LinalgNamed.lean`: one table `name ↦ (maps, iterator kinds, source line)`; `sem` reads
   operands per `operandSegmentSizes`, honors `strides`/`dilations`, `indexing_maps` overrides on
   matmul/batch_matmul, `permutation`/`dimensions`, and calls `Linalg.genericSem` with the op's own
   region. Put the table in the module docstring for WP-5's cross-check.
4. `#guard` tests on small integer ASTs; then a progress lemma per family.

---
title: "Design note: whole-model transpose-b fusion (fuse_transpose_b)"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [whole_model_op_profile, expert_gap_attribution, compiler_plane]
code_refs: [merlin/python/merlin/llvmlower/transpose_fuse.py, merlin/python/merlin/llvmlower/pipeline.py, merlin/python/merlin/llvmlower/impr_features.py, merlin/python/merlin/kernels/action_catalog.py, merlin/python/merlin/kernels/cca_contract.py, build_tools/scripts/k1_op_profile.py]
---

# Whole-model transpose-b fusion

## The measurement that made this #1

The whole-model per-op profiler (`build_tools/scripts/k1_op_profile.py`, `llvmlower/op_profile.py`)
attributed, on the K1, `linalg.transpose` = **390 ms across 45 ops** in openvla — the single largest
non-contraction bucket, emitted **scalar** (`convert-linalg-to-loops`; not in the vectorized
`contraction` family). Every openvla matmul (26/26) and bitvla matmul (15/15) is a **transposed-B
GEMM**: the frontend tags it `prov.transposed_b = "true"` and emits a *standalone* weight transpose
feeding the matmul's B operand:

```mlir
%Bt = linalg.transpose ins(%W : tensor<NxKxf32>) outs(... : tensor<KxNxf32>) permutation = [1, 0]
%C  = linalg.matmul indexing_maps = [#A=(m,k), #B=(k,n), #C=(m,n)]
                    ins(%A, %Bt : tensor<MxKxf32>, tensor<KxNxf32>) outs(%C0) -> ...
```

That materializes a full transposed copy of the weight in DRAM **every forward** and reads it back —
pure overhead. BLAS/XNNPACK never do this: they read B transposed via the GEMM's own access pattern
(a "transpose-b" kernel). A kernel library is stuck at the call boundary; a compiler is not — this is
the flagship whole-model, cross-op case.

## What the feature does

LLVM-23 removed the `linalg.matmul_transpose_b` named op; transpose-b is now expressed as a plain
`linalg.matmul` whose B `indexing_map` reads the *un-transposed* weight `(n, k)` instead of `(k, n)`.
So the fusion is a pure operand + map rewrite, done in place before the pass manager runs
(`transpose_fuse.py`, spliced into the lowering runner, gated by `argv[5]`):

1. repoint the matmul's B operand from the transpose RESULT to the transpose's SOURCE `%W`;
2. permute the B `indexing_map` results by the transpose permutation (`new[j] = old[perm[j]]` — for
   the 2-D `[1,0]` case, the swap `(k,n) -> (n,k)`);
3. erase the transpose if it is now dead (single-use is the common case, and is what the captures
   emit).

The op stays `linalg.matmul` with valid contraction `indexing_maps`, so the **frozen** RVV transform
schedule (which matches `ops{["linalg.matmul"]}`) still tiles + vectorizes it — verified with
mlir-opt: the transpose-b matmul lowers through `tile -> vectorize -> lower_contraction` to the same
`vector.fma` chain. Net: the scalar weight transposes **disappear** (no op, no buffer), and B is read
`(n, k)` — contiguous along k in the row-major `[N, K]` weight, exactly the transpose-b access an
expert GEMM uses.

### Correctness

The rewrite is **value-identical by construction**: `B[k, n]` on the transposed weight equals
`W[n, k]` on the source, and the map change encodes precisely that. It is a **default-off** compiler
feature (`fuse_transpose_b`) so the frozen `hand_v0` control keeps a byte-identical lowering (the
transform schedule and pass pipeline are unchanged; the rewrite is runner-side and gated). It is
gated on the board with a **per-element** check (`fp32_rel = max|pred-ref| / max|ref|`), not only cos:
cos alone was measured to accept a kernel 1209 % wrong per-element.

## Measured whole-model result (K1, fp32, n=3, correctness-gated)

Un-instrumented control walls (the perturbation guard confirmed the profiler moved the wall < 0.8 %):

| model  | wall before | wall after | Δ wall        | transpose bucket | scalar frac | fwd RVV cov |
|--------|-------------|------------|---------------|------------------|-------------|-------------|
| openvla| 5995.7 ms   | 5604.3 ms  | **−391 ms / −6.53 %** | 390.0 → 0.0 ms | 6.7 % → 0.3 % | 0.229 → 0.255 |
| bitvla | 2492.9 ms   | 2414.9 ms  | **−78 ms / −3.13 %**  | 45.5 → 0.4 ms  | 3.4 % → 1.5 % | 0.219 → 0.247 |

Both deltas are well above the board noise floor (≈1.9 %). The matmul bucket is **unchanged** by the
rewrite (openvla 5602 → 5561 ms, within noise) — the contiguous `(n,k)` B read did not slow the
vectorized contraction, so the transpose elimination is pure gain. Correctness held at the baseline
level in every gated run: openvla cos 0.9999999 / rel 9.6e-7 (identical to baseline), bitvla cos
0.9999946 / rel 3.1e-3 (identical to baseline — a pre-existing property of that recapture, not a
fusion effect).

> Note on the 57 % framing: the task's "transpose = 57 % of openvla" was measured with a *fast* matmul
> kernel routed in, so transpose dominated relative to a fast GEMM. Here the profiles use our native
> RVV matmul (slow), so contraction dominates and transpose is 6.5 % of the *whole model* in absolute
> terms — but the ~390 ms of transpose is the same, and eliminating it is a 6.5 % whole-model win
> regardless of how fast the matmul is. With a fast matmul routed in, the same 390 ms would be an even
> larger fraction.

## CCA routing

The divergence — "the expert folds the transpose into the contraction's access; we materialize a
standalone transpose op+buffer" — is routed in `action_catalog.py` on a new
`layout.transpose_materialized` axis to the `impr_features:fuse_transpose_b` PASS
(`forkable_now=True`, with the measured effect recorded). The backing CCA *facet field* that would let
the CCA auto-discover this divergence needs a **graph-level (IR) lifter** — the pattern is a
whole-model graph property, not visible in a single contraction kernel's asm that `lift_asm` sees — so
that lifter is recorded as a deferred work-item in `cca_contract.KNOWN_OPEN["rvv"]["orphan_routes"]`
(the same honest treatment as `compute.mr_adapts_to_m`). The route and the measured win are real now;
only the auto-discovery lifter is deferred.

## Generality

The rewrite matches any `matmul(A, transpose(B, perm))` whose transpose permutation matches the B
map's arity — not an openvla-specific pattern. It fired on all 26 openvla and all 15 bitvla matmuls,
and the map-permutation rule (`new[j] = old[perm[j]]`) is general over the permutation. Batch-matmul
(attention) transpose-b was **not** targeted: those transposes feed reshape chains rather than the
batch_matmul directly, and are on tiny activation tensors (17×64), so the whole-model win is entirely
in the 2-D weight transposes.

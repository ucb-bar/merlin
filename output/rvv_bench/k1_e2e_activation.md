# Vectorized-activation feature on K1 — honest silicon result

The `vectorized_transcendental_activation` feature (commit b5c7450) rewrites `math.erf`/`exp`/`tanh`
to a compiler-emitted minimax polynomial (act_poly.py) and vectorizes the elementwise activation
`linalg.generic`. On **spike**, isolated, it closed our scalar gap ~6.2× (GELU) / ~3.2× (sigmoid).
This is the on-silicon (K1, VLEN=256, `rdtime`, N=3 min) re-measure of two questions:

1. **Whole-model e2e** (bitvla, openvla): does it lift the model? **No** — it falls back to scalar
   whole-model (a schedule-edit limitation), so it is a regression there.
2. **Isolated GELU/sigmoid** (the op the feature targets): does the compiler polynomial now match
   XNNPACK on silicon? **It vectorizes and is 2.3–3.7× faster than our scalar, but still ~3.6–4.6×
   behind XNNPACK's hand polynomial** — it narrows the gap from ~11–18× to ~3.6–4.6×, not closed.

## VALIDATION 1 — whole-model e2e (N=3 min wall, cos vs host golden)

### bitvla (golden (1,32,1024))

| config | feature | lowering | min wall | cos | vs baseline |
|---|---|---|---|---|---|
| baseline | — | vectorized (fixed-width SIMD, matmul) | 2.528 s | 0.999995 | 1.00× |
| **act_alone** | `vectorized_transcendental_activation` | **scalar_fallback** | **3.327 s** | 0.999995 | **0.76× (REGRESSION)** |
| matmul_only (context) | `fused_vfmacc_tiled` | vectorized (vfmacc) | 0.274 s | 0.999995 | 9.24× |
| act_plus_matmul | act + `fused_vfmacc_tiled` | **not_run** | — | — | **CompositionError (both schedule_replace)** |

### openvla (golden (1,20,512))

| config | feature | lowering | min wall | cos | vs baseline |
|---|---|---|---|---|---|
| baseline | — | vectorized (fixed-width SIMD, matmul) | 5.884 s | 1.000000 | 1.00× |
| **act_alone** | `vectorized_transcendental_activation` | **scalar_fallback** | **8.706 s** | 1.000000 | **0.68× (REGRESSION)** |
| matmul_only (context) | `fused_vfmacc_tiled` | vectorized (vfmacc) | 1.617 s | 1.000000 | 3.64× |
| act_plus_matmul | act + `fused_vfmacc_tiled` | **not_run** | — | — | **CompositionError (both schedule_replace)** |

**Whole-model lowering verdict (measured via `lower_and_characterize`, fmuladd/fixedvec in the .ll):**
the activation feature's `_ACT_POLY_SCHEDULE` tiles+vectorizes EVERY `linalg.generic [16]`, which fails
on the model's *non-activation* generics → `PipelineError` → `build_k1_binary` silently falls back to
the SCALAR (vectorize=False, no-feature) lowering. So `act_alone` whole-model runs FULL scalar — it
loses even the baseline's matmul fixed-width SIMD, hence the 0.68–0.76× regression. The exact errors:
- bitvla:  `"too many tiles provided, expected at most 0 found 1"` (schedule line 10, a rank-collapsed generic)
- openvla: `"Attempted to vectorize, but failed"` (a non-activation generic the [16] tile can't take)

**Composition verdict (V1c):** `vectorized_transcendental_activation` and `fused_vfmacc_tiled` are
BOTH `schedule_replace=True` full-schedule features, so `apply_schedule` raises `CompositionError` by
design (one would clobber the other). They do **NOT** compose — recorded `not_run`, honest. The
matmul feature (`fused_vfmacc_tiled`) is the whole-model win (9.24× bitvla / 3.64× openvla); the
activation feature contributes nothing whole-model on these two models (scalar fallback). The
activation feature is a **kernel-level** feature — its real payoff is isolated (V2 below).

## VALIDATION 2 — isolated GELU / sigmoid on K1 (rdtime ticks, N=3 min; cos/abs-verified)

Unlike whole-model, in the isolated single-activation workload the feature lowers CLEANLY and DOES
vectorize (characterized: `<N x float>` SIMD count 0→96 GELU / 0→61 sigmoid, libm `erff`/`expf` call
1→0). The prior matrix's `ours_vectorized` column was lowered with `vectorize=True` but `features=[]`
(the feature did not exist when built) → still scalar libm; it is renamed `ours-vec (no-feat)`. The
genuine polynomial is the new `ours-vectorized (poly)` column:

| op | N | XNNPACK | ours-scalar | ours-vec (no-feat) | **ours-vectorized (poly)** | poly vs scalar | poly vs XNN |
|----|---|---------|-------------|--------------------|----------------------------|----------------|-------------|
| GELU | 1K | 270 | 2765 | 2929 | **983** | 2.81× faster | 3.64× slower |
| GELU | 16K | 2656 | 45171 | 47400 | **12189** | 3.71× faster | 4.59× slower |
| GELU | 256K | 44718 | 714815 | 775349 | **201720** | 3.54× faster | 4.51× slower |
| sigmoid | 1K | 139 | 1277 | 1445 | **559** | 2.28× faster | 4.02× slower |
| sigmoid | 16K | 2111 | 23359 | 25363 | **8097** | 2.88× faster | 3.84× slower |
| sigmoid | 256K | 35305 | 375784 | 415575 | **139149** | 2.70× faster | 3.94× slower |

**Accuracy (approximation, NOT bit-exact):** every poly row PASSES the driver's verify (errors=0 at
2e-3 abs tolerance) vs the scalar `erff`/`expf` reference. (Spike measured cos≈1.0, max rel-err
~5e-8…~1.2e-7 vs libm; the K1 driver gate is the abs band, all PASS.)

**Headline:** the compiler-emitted polynomial **MATCHES neither nor BEATS** XNNPACK's hand polynomial
on silicon — it still **TRAILS** by ~3.6–4.6×. But it is a genuine on-silicon win over our own scalar
(~2.3–3.7×) and narrows the XNNPACK gap from the previously-reported ~11–18× (which was scalar
mislabeled as vectorized) to ~3.6–4.6×. The residual is that our polynomial lowers to separate
`vfmul.vv`+`vfadd.vv` fixed-width SIMD (fmuladd=0 — no fused vfmacc) and carries more ops/element than
XNNPACK's tuned rational-12-10 / rr2-p5 kernels with their merge/select tricks.

## Conclusion (honest)
- **Whole-model:** the activation feature does NOT lift bitvla/openvla — it falls back to scalar
  (schedule edit not robust to non-activation generics) and is a 0.68–0.76× regression. It does not
  compose with the matmul vfmacc feature (CompositionError, by design). The whole-model win remains
  `fused_vfmacc_tiled` (matmul). The activation feature is correctly default-off.
- **Isolated (the op it targets):** it works — vectorizes, 2.3–3.7× faster than our scalar, accuracy
  within the approximation band — but still ~3.6–4.6× behind XNNPACK's hand polynomial. The spike
  closure (6.2×/3.2× vs OUR scalar) translates qualitatively to silicon (2.3–3.7× vs OUR scalar); it
  does NOT close the gap to XNNPACK.

Data: `output/rvv_bench/k1_e2e_activation.json` (both models),
`output/kernels/ceiling/{ours_vectorized_ops_k1.jsonl, cross_framework_ops_k1.{jsonl,md}}`,
plot `output/kernels/ceiling/op_coverage.png`. Board left clean.

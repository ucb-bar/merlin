# Vectorized-activation feature on K1 — honest result (does NOT yet vectorize on real workloads)

The `vectorized_transcendental_activation` feature (commit b5c7450) rewrites `math.erf`/`exp`/`tanh`
to a compiler-emitted minimax polynomial (vfmacc Horner). On **spike**, for a synthetic isolated
activation op, it vectorizes (vfmacc>0) and closes the gap vs our scalar baseline ~6.2× (GELU) / ~3.2×
(sigmoid). **On real K1 workloads it does NOT vectorize** — a schedule-edit bug makes it fall back to
scalar. This is the honest, measured state; a fix is in progress.

## Whole-model bitvla on K1 (N=3 min wall, cos vs host golden)

| config | feature | lowering | min wall | cos | vs baseline |
|---|---|---|---|---|---|
| baseline | — | vectorized (fixed-width SIMD) | 2.528 s | 0.999995 | 1.00× |
| **activation feature** | `vectorized_transcendental_activation` | **scalar_fallback** | **3.327 s** | 0.999995 | **0.76× (REGRESSION)** |
| matmul (context) | `fused_vfmacc_tiled` | vectorized | 0.274 s | 0.999995 | 9.24× |

The activation feature raises a lowering **PipelineError** —
`"too many tiles provided, expected at most 0 found 1"` — when its schedule edit tries to tile the
activation `linalg.generic`, and the model silently falls back to **scalar** (slower than the
fixed-width-SIMD baseline). So whole-model it is currently a **regression**, not a win.

## Isolated GELU / sigmoid on K1 (rdtime ticks; ours WITH the feature vs XNNPACK)

| op | size | XNNPACK | ours (feature on) | gap |
|---|---|---|---|---|
| GELU | 1K / 16K / 256K | 270 / 2656 / 44718 | 2929 / 47400 / 775349 | ~10.8× / ~17.8× / ~17.3× behind |
| sigmoid | 1K / 16K / 256K | 139 / 2111 / 35305 | 1445 / 25363 / 415575 | ~10.4× / ~12.0× / ~11.8× behind |

The "ours" ticks are **identical to the scalar baseline** measured earlier — i.e. the feature fell
back to scalar here too (same schedule-edit bug). So on the board the activation gap to XNNPACK is
**not yet closed**; the spike closure was on a path that lowers cleanly only in isolation.

## Conclusion (honest)
- The polynomial math is correct (spike: cos≈1.0, vfmacc forms in the synthetic case).
- But the feature's **schedule edit is not robust** to real activation generics — it raises
  `too many tiles ... expected at most 0 found 1` and the pipeline falls back to scalar.
- **Net board effect today: regression (0.76× whole-model) and still ~11–18× behind XNNPACK.**
- **Fix required:** make the activation vectorization apply without the bogus tile spec (vectorize
  the elementwise generic directly / drop the tile count for rank-collapsed activation ops), then
  re-measure. Until then, this feature is NOT a board win and is correctly default-off.

(openvla activation e2e and the post-fix re-measure are pending; the board agent stalled mid-run but
the bitvla + isolated data above already establish the result.)

# Experiment `beam_rvv_v2` — faithful whole-model beam re-run

**Run:** `mined_knowledge/rvv/beam_rvv_v2_20260619T132435/` (manifest + per-model rankings + summary + raw).
**Supersedes** `beam_rvv_v1` (single-op `matmul_f32_64x64x64`, 7 candidates, predated v3 — a single small
GEMM mis-ranks whole-model winners). This run ranks the **current** candidate optimizations **whole-model**
on real driving examples (bitvla, openvla) on the **SpacemiT K1**, by min wall (N=3) + fp32 cos vs host
golden. Baseline = frozen `hand_v0`. Figure: `output/kernels/ceiling/beam_rvv_v2_ranking.png`.

## Result — best kernel is per-model

| rank | bitvla | speedup | openvla | speedup |
|---|---|---|---|---|
| 1 | **v3_plus_lmul** | **16.77×** | **accum_wholemodel** | **4.97×** |
| 2 | microkernel_v3 | 16.73× | tiled_plus_lmul | 3.64× |
| 3 | tiled_plus_lmul | 9.20× | matmul_only (tiled) | 3.63× |
| 4 | matmul_only (tiled) | 9.12× | v3_plus_lmul | 2.39× |
| 5 | accum_wholemodel | 8.10× | microkernel_v3 | 2.39× |
| 6 | accum_ntail | 7.75× | accum_ntail | 1.35× |
| 7 | lmul_widen | 1.04× | lmul_widen | 1.05× |
| 8 | baseline | 1.00× | baseline | 1.00× |
| 9 | act_alone | 1.00× | act_alone | 0.96× |
| 10 | **vfmacc_contraction** | **0.75× (scalar fallback)** | vfmacc_contraction | 0.68× (scalar fallback) |
| — | v3_plus_act, act_plus_matmul | blocked (CompositionError) | (same) | blocked |

All run configs cos ≥ 0.99999.

## What the faithful beam found (that the manual comparison missed)
1. **New openvla winner — `accum_wholemodel` (4.97×)** — beats tiled (3.63×) *and* v3 (2.39×). The beam
   discovered this by measuring every candidate whole-model; the earlier manual pass had only compared
   tiled vs v3 and reported 3.65×.
2. **bitvla winner = v3** (16.7×, beats XNNPACK 13.65×); `+lmul` adds nothing (16.77 vs 16.73 = noise).
3. **`lmul_widen_n` is a near-no-op whole-model** (1.04–1.05×); every `*_plus_lmul` combo ≈ its base feature.
4. **`vfmacc_contraction` regresses** (0.75× / 0.68×, scalar fallback) — it is kernel-sized-only, not
   whole-model-safe (the full-unroll breaks on the model's op mix).
5. **`vectorized_activation` is neutral** (1.00× / 0.96×) — correct + vectorized, but activations are
   negligible vs matmul at whole-model scale.
6. **Composition limit (honest):** v3+activation and activation+matmul are blocked — both are
   full-schedule-replacement features. Getting matmul-v3 *and* vectorized activation together needs a
   unified schedule (future work).

## Takeaway
The compiler now holds a **portfolio of whole-model-safe optimizations**, and the faithful beam selects
the right one **per model**: v3 (beats XNNPACK) on bitvla, accum-resident-wholemodel on openvla. This is
the reproducible, versioned form of "how the beam search performed" — vs `beam_rvv_v1`'s stale single-shape
run. Re-run: `scripts/k1_e2e_activation.py --configs <candidates> --models <model>` per the manifest.

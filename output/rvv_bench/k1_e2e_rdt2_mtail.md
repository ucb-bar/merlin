# rdt2 whole-model on K1: matmul M-tail clamp + composed whole-model-safe vfmacc

Closes the two whole-model-safety / composition gaps the K1 real-model validation surfaced, flowed
through the mining -> action-catalog -> impr_features pipeline. General (evidence-derived), default-OFF,
baseline FROZEN. No hand kernels as results.

## The two gaps

1. **matmul M-tail (WORK-ITEM 1).** rdt2/smolVLA decode is dominated by matmuls with leading **M=1**
   (one token row). The accumulator-resident schedule's `MR=4` register tile writes a `vector<4xNR>`
   into a `tensor<1xNR>` C tile -> a masked `vector.transfer_write` LLVM-23 rejects (multi-op
   `vector.mask` PipelineError) -> silent **scalar fallback** (no vfmacc). This is the M-side analog
   of the batch_matmul N-tail (N<NR) already fixed in `accumulator_resident_ntail`.
   Reproduced on rdt2: the un-clamped `accumulator_resident_microkernel` -> `lowering=scalar_fallback`,
   `pipeline_error_op=vector.mask`, `fmuladd=0`.

2. **Composition (WORK-ITEM 2).** `fused_vfmacc_tiled` and `accumulator_resident_ntail` are both FULL
   schedule replacements (their `edit_schedule` ignores its input). `apply_schedule` iterated
   `sorted(features)`, so enabling both let the last-sorted one CLOBBER the other's clamp.

## Design chosen

**Inherent-clamp (not additive-edit) + composition guard.** The transform schedules differ
structurally (tile vs pack vs bufferize_to_allocation), so additive text edits do not layer cleanly.
Instead:

- The tail clamps are made **inherent to one parameterized schedule**:
  `_accumulator_resident_pre_schedule(MR, NR, KC, NR_bmm, MR_mm)` now clamps the matmul `MR=min(MR,M)`
  (`MR_mm`) AND the batch_matmul `NR=min(NR,N)` (`NR_bmm`). A tile that adapts on BOTH the M side
  (matmul M<MR) and the N side (batch_matmul N<NR) — general, not a memorized shape.
- New default-OFF features:
  - `accumulator_resident_mtail` — matmul `MR_mm=1` (M-tail only).
  - `accumulator_resident_wholemodel` — the COMPOSED config: `MR_mm=1` (M-tail) AND `NR_bmm=8`
    (N-tail) in ONE schedule, so the best single config is whole-model-safe by construction.
- A **composition guard**: each full-replacement feature is flagged `schedule_replace=True`;
  `apply_schedule` refuses (`CompositionError`) to apply more than one, instead of silently picking a
  winner. Additive edits (`schedule_replace=False`, e.g. `lmul_widen_n`) still layer on top.

Routing: `action_catalog.py` gets `compute.mr_adapts_to_m` as a typed forkable **HEURISTIC**
(target_seam `schedule:MR=min(MR,M)` -> `impr_features:accumulator_resident_mtail`), consistent with how
`compute.nr_is_vsetvlmax` was routed.

## Spike bit-exact across the shape spread (correctness authority)

`accumulator_resident_wholemodel`, built through the real RVV compiler + run on spike, gated cos vs golden:

| shape | kind | build | spike cos | vfmacc | vfmul |
|---|---|---|---|---|---|
| matmul 64x64x64 | normal cube | ok | 1.0 | >0 | 0 |
| matmul 1x64x64 | **M=1 token-decode** | ok | 1.0000001 | >0 | 0 |
| matmul 96x48x160 | **non-cube** | ok | 1.0 | >0 | 0 |
| batch_matmul B4,M32,N8,K32 | **N=8 attention** | ok | 1.0000001 | >0 | 0 |

All four vectorize to vfmacc in ONE schedule (no scalar fallback, no `vector.mask` PipelineError) and
are bit-exact — proves general, not M=1-overfit. (The M-tail-only feature passes cube/M=1/non-cube;
the N=8 batch_matmul needs the composed feature, confirming both clamps are required.)

## K1 board (real silicon, rdtime wall proxy, cos vs HOST golden), N=3, min wall

rdt2 has 23 matmuls with mixed leading M (M=1 decode: `1x1024`, `1x9216`, `1x256`, `1x2048`;
plus M=28, M=256) and no batch_matmul — a pure M-tail whole-model case.

| config | lowering | ll fmuladd | min wall (s) | cos vs host | vlen |
|---|---|---|---|---|---|
| baseline (hand_v0, FROZEN) | vectorized | **0** (vfmul+vfadd) | 73.71 | 1.0000001 | 256 |
| `accumulator_resident_wholemodel` | vectorized | **57** (vfmacc) | **31.41** | 1.0 | 256 |

**Speedup 2.35x**, cos=1.0. The rdt2 M=1 matmuls now vectorize to **vfmacc** (`fmuladd` 0 -> 57)
instead of the masked-transfer_write scalar fallback. Baseline frozen; the win comes entirely from the
default-OFF composed feature.

Honest scope: K1 `rdtime` is the real-silicon wall proxy (`cycle_accurate=false`); spike is the
cycle-accurate correctness authority. cos vs host golden verified (=1.0) before reporting the speedup.

Raw: `output/rvv_bench/k1_e2e_rdt2_mtail.json`.

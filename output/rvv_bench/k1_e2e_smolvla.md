# Validating the GENERAL compiler features (commit `e5dd143`) on REAL models on the K1 board

Goal: EVIDENCE that the general transforms that just landed (`accumulator_resident_ntail`,
the vfmacc fusion family, conv/attention vectorization) help on real whole models running on
SpacemiT K1 silicon — and an honest account of where they do NOT — **not** a tuning loop against
specific shapes. Baseline is the FROZEN `hand_v0` schedule; the new features are default-OFF and
enabled only on `impr` forks.

- Board: SpacemiT K1, Bianbu Linux, VLEN=256 (`vlenb`*8 read at run time), 8 cores, 3.4 GB RAM.
- Timing: whole-model `wall_ns` from `clock_gettime(CLOCK_MONOTONIC)` (ground truth); `time_ticks`
  = delegated `rdtime` (24 MHz). `cycle_accurate=false` (userspace `rdcycle` traps). N=3 runs, min reported.
- Correctness: fp32 cosine of the board output vs the **host-captured** `golden.npy`
  (`zephyr_model._gate`) — verified BEFORE any speedup is quoted. Honest `not_run` over fabrication.
- Harness REUSED unchanged: `merlin.rvvgen.k1.run_on_k1` / `build_k1_binary` (the bitvla 9.35x
  precedent path). The driver `scripts/k1_e2e_general_validate.py` only adds measurement: a
  whole-model lowering probe (`vectorized` vs `scalar_fallback` + the exact `PipelineError` op) and
  `.ll` fma/fixed-vector counts.

## TL;DR

1. **smolVLA whole-model on K1 = honest `not_run`** — blocked by a **SpacemiT-toolchain backend
   crash** that hits even the FROZEN baseline (not a feature regression). Separately, at the
   lowering level the N-tail feature would NOT have vectorized smolVLA's attention anyway: smolVLA
   has 237 `M=1` token-row matmuls that trip a *different* `vector.mask` failure (a matmul **M**-tail,
   not the batch_matmul **N**-tail the feature fixes). Both findings are precise and reproducible.
2. **The N-tail fix is general and does what it claims** on the models where it lowers: it
   vectorizes attention `batch_matmul` to **vfmacc** where the older `fused_vfmacc_tiled` silently
   fell back to **scalar**. Proven at the lowering level on tiny_llama / small_llama, and at the
   **board** level on bitvla (7.73x) and openvla (1.33x), both numerically correct.
3. **Cross-model board evidence (not one-shape-overfit):** the SAME general features speed up TWO
   distinct real models with attention — **bitvla** (re-confirmed 9.22x with `fused_vfmacc_tiled`,
   7.73x with `accumulator_resident_ntail`) and **openvla** (3.61x / 1.33x). Both stay correct
   (cos ≥ 0.9999999).

---

## 1. smolVLA full e2e (the explicit ask) — honest blocker

Model `output/smolvla_fp32_consistent` (1.2 GB weights, golden `(1,50,32)`, finite/no-NaN,
264 attention `batch_matmul` + 302 `matmul`). The before/after build was attempted exactly like the
bitvla run. Result for every config, including the frozen baseline:

```
fatal error: error in backend: Incomplete scavenging after 2nd pass
clang: error: clang frontend command failed with exit code 70
clang version 19.1.7   (SpacemiT spacemit-toolchain-linux-glibc v1.1.2)
Target: riscv64-unknown-linux-gnu
```

The SpacemiT cross-clang (LLVM 19.1.7) **crashes in its RISC-V register scavenger** while compiling
smolVLA's whole-model translation unit (`model.o` + `main_linux.c` link step). This is a
**toolchain limitation on a large model**, independent of the compiler features under test — the
FROZEN `hand_v0` baseline crashes identically. So there is no valid smolVLA board number to report:
**`not_run` (SpacemiT clang backend crash)**. Not forced.

### What the lowering probe shows for smolVLA (the feature-level answer to "did attention vectorize")

Even setting the toolchain crash aside, the headline N-tail question is answered at the MLIR level
(this is what `build_k1_binary` would have compiled):

| config | whole-model lowering | `PipelineError` op | `.ll` fmuladd | `.ll` fixedvec | attention |
|---|---|---|---|---|---|
| baseline (hand_v0) | **vectorized** | — | 0 | 56,975 | fixed-width SIMD (`vfmul.vv`+`vfadd.vv`), **no fused vfmacc** |
| optimized `accumulator_resident_ntail` | **scalar_fallback** | `vector.mask` | 0 | (scalar) | **scalar — vfmacc path fell back whole-model** |
| `fused_vfmacc_tiled` + `accumulator_resident_ntail` ("combined") | scalar_fallback | `vector.mask` | 0 | (scalar) | scalar (see composition note) |

So on smolVLA the N-tail fix does **not** let attention vectorize — it raises `PipelineError`, and
`build_k1_binary` silently falls back to a fully-scalar contraction (worse than the baseline's
fixed-width SIMD). **Root cause (distinct from the llama N-tail case):** smolVLA contains **237
matmuls whose leading M dimension is 1** (single-token decode rows, e.g. `tensor<1x960xf32>`). The
ntail schedule tiles matmul `[MR=4, NR=16, KC=16]`; with `M=1 < MR=4` the M tile is masked, producing
a `vector.mask` that wraps **two** ops (a `transfer_write` to `tensor<1x16xf32>` *and* a
`tensor.cast`), which LLVM-23 rejects (`'vector.mask' op expects only one operation to mask`). The
feature clamps `NR` for small-N **batch_matmul** (the attention N-tail) but does not clamp `MR` for
small-M **matmul**; smolVLA's M=1 rows are an unaddressed *matmul M-tail*. Concrete follow-up
work-item: an analogous `MR=min(MR,M)` clamp (or single-op-mask lowering) for M<MR matmuls.

### Composition note (important, and contrary to the naive "combine the two features")

`fused_vfmacc_tiled` and `accumulator_resident_ntail` are each a **full schedule replacement**
(their `edit_schedule` ignores the incoming text and returns a complete transform schedule).
`apply_schedule` folds features in `sorted()` order, so enabling BOTH means `fused_vfmacc_tiled`
(sorts after `accumulator_resident_ntail`) **overwrites** the ntail schedule and the N=8 batch_matmul
N-clamp is LOST — the "combined" config is byte-identical to `fused_vfmacc_tiled` alone
(verified: combined batch_matmul tile `[1,4,16,16]` == tiled-alone, whereas ntail-alone is
`[1,4,8,…]`). The config that actually delivers the attention N-tail payoff is
**`accumulator_resident_ntail` ALONE**. This validation therefore uses ntail-alone as the optimized
config and records the clobbered "combined" only to make the non-composition explicit.

---

## 2. Cross-model evidence the general transform isn't one-shape-overfit

### 2a. bitvla — re-confirmed on the board with the current code (`k1_e2e_bitvla_reconfirm.json`)

`output/bitvla_fp32_consistent` (9 MB weights, golden `(1,32,1024)`, finite). All configs vectorize
whole-model and stay correct.

| config | lowering | `.ll` fmuladd | attention | min wall (N=3) | fp32 cos | speedup vs baseline |
|---|---|---|---|---|---|---|
| baseline (hand_v0) | vectorized | 0 | fixed-width SIMD, no fma | 2.525 s | 0.9999946 | 1.00x |
| **`accumulator_resident_ntail`** | vectorized | 78 | **vfmacc** | 0.327 s | 0.9999946 | **7.73x** |
| `fused_vfmacc_tiled` | vectorized | 1217 | **vfmacc** | 0.274 s | 0.9999946 | **9.22x** |

Re-confirms the prior 9.35x precedent (9.22x now; run-to-run jitter). The N-tail feature ALSO works
whole-model on bitvla — it vectorizes attention to vfmacc (7.73x) with a more conservative tile.

### 2b. openvla — a SECOND real VLA model with attention, runnable + numerically valid (`k1_e2e_openvla.json`)

`output/openvla_fp32_consistent` (29 MB weights, golden `(1,20,512)`, finite, 27 attention
`batch_matmul`). Both vfmacc configs lower whole-model and stay correct.

| config | lowering | `.ll` fmuladd | attention | min wall (N=3) | fp32 cos | speedup vs baseline |
|---|---|---|---|---|---|---|
| baseline (hand_v0) | vectorized | 0 | fixed-width SIMD, no fma | 5.848 s | 0.9999999 | 1.00x |
| **`accumulator_resident_ntail`** | vectorized | 239 | **vfmacc** | 4.409 s | 1.0 | **1.33x** |
| `fused_vfmacc_tiled` | vectorized | 3330 | **vfmacc** | 1.619 s | 0.9999999 | **3.61x** |

The SAME general features help a second, structurally different real model — the N-tail feature
vectorizes openvla's attention to 239 vfmacc with no loss of correctness. Evidence: not bitvla-overfit.

### 2c. The N-tail fix vectorizes the llama-family attention where the older feature could not (lowering-level)

Agent A's note recorded that `fused_vfmacc_tiled` falls back to scalar on the llama family because
its attention `batch_matmul` has N=8 < NR=16 (the `vector.mask` N-tail). `accumulator_resident_ntail`
is exactly the fix for that. Confirmed at the lowering level on both llama captures:

| model | `fused_vfmacc_tiled` | `accumulator_resident_ntail` |
|---|---|---|
| tiny_llama (`fp8_consistent`, all-f32 graph) | scalar_fallback (`vector.mask`, N=8) | **VECTORIZED whole-model, 82 vfmacc** |
| small_llama (`fp8_consistent`) | scalar_fallback (`vector.mask`, N=8) | **VECTORIZED whole-model, 78 vfmacc** |

So the N-tail fix is **general** — it closes the exact whole-model attention regression Agent A
flagged. These two are NOT reported as board numbers because, per Agent A and re-verified here, they
are not valid hardware targets: **tiny_llama** OOMs the board (embedded-fp32 RW segment ~398 MB vs
the board's ~1.9 GB commit limit) and **small_llama** is numerically all-NaN on both spike and K1
(broken fp8→f32 capture). Reporting their wall would be meaningless; the durable, honest evidence is
the lowering-level vectorization above plus the two correct board models (bitvla, openvla).

---

## Honest summary table (all real models surveyed)

| model | attention `batch_matmul` | board run | N-tail lowers whole-model? | result |
|---|---|---|---|---|
| **bitvla** | 12 | ✅ correct | yes (vfmacc) | ntail 7.73x, tiled 9.22x, cos 0.9999946 |
| **openvla** | 27 | ✅ correct | yes (vfmacc) | ntail 1.33x, tiled 3.61x, cos ≥ 0.9999999 |
| **smolVLA** | 264 | ❌ `not_run` | no (`vector.mask`, matmul M=1 tail) | SpacemiT clang backend crash (hits frozen baseline too) |
| tiny_llama | (N=8) | ❌ OOM | **yes (82 vfmacc)** | board OOM; ntail FIXES the lowering vs tiled |
| small_llama | (N=8) | ❌ NaN capture | **yes (78 vfmacc)** | capture numerically broken; ntail FIXES the lowering vs tiled |
| rdt2 | 24 | not attempted | no (`vector.mask`, M-tail) | same matmul M-tail class as smolVLA |

## Caveat respected
The transform-path accumulator-resident microkernel is still ~19x off the hand `ours_intrinsic_gemm_driver.c`
**CEILING REFERENCE** (it spills the accumulator to stack in the K-loop). No hand-kernel number is
presented here as a compiler result; every speedup above is the compiler-emitted lowering measured
on real silicon.

## Files
- `scripts/k1_e2e_general_validate.py` — measurement-only driver (reuses the frozen harness).
- `output/rvv_bench/k1_e2e_smolvla.json` — smolVLA characterization + `not_run` blockers.
- `output/rvv_bench/k1_e2e_bitvla_reconfirm.json` — bitvla board re-confirm (current code).
- `output/rvv_bench/k1_e2e_openvla.json` — openvla board before/after.

## Constraints honored
Baseline FROZEN: `git diff --name-only` touches none of `RVV_TRANSFORM_SCHEDULE`,
`build_rvv_pipeline`, `RVV_CFLAGS`, `impr_features.py`, `k1.py`, `cca.py`, `action_catalog.py`,
`hand_v0` (only the new script added). `pytest merlin/python/tests/test_impr_features.py` = 12 passed.
Board left clean (all deployed binaries/weight blobs removed; RAM freed). No push.

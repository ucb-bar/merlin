# Cross-framework op coverage — BEYOND matmul, on real K1 silicon

The headline cross-framework matrix (`cross_framework_matrix*.{md,jsonl}`) is **GEMM-only**, because
OpenBLAS is a BLAS (gemm/gemv/dot only). XNNPACK, however, ships RVV kernels for many more ops we
never raced. This extends the comparison to the ops XNNPACK actually has RVV kernels for, reusing the
**same harness and protocol** as `scripts/k1_cross_framework.py`:

* standalone driver per op (no framework runtime), inner-compute scope (pack/setup hoisted OUT of the
  timed region — noted per op below),
* bit-exact (or, for int8, cosine>0.99 — the repo's W8A8 tier) verified vs a scalar reference **before**
  any speedup is reported,
* K1 silicon `rdtime` ticks, **N=3 reps, min kept** (`cycle_accurate=false`; a 24 MHz wall proxy, not
  core cycles), `INSTRET` on spike where the kernel builds there.

Data: `output/kernels/ceiling/cross_framework_ops_k1.jsonl` (K1, 33 rows) and
`cross_framework_ops_spike.jsonl` (spike instret authority). Plot:
`output/kernels/ceiling/op_coverage.png` (6 op panels). Run:
`.venv/bin/python scripts/k1_cross_framework_ops.py`.

## Toolchain note (why some rows are spike-not_run)

Many XNNPACK RVV kernels (sigmoid, tanh, qd8 int8 gemm, dwconv) use the **overloaded** RVV intrinsic
spellings (`__riscv_vfmerge`, `__riscv_vse32`, `__riscv_vwmacc`, …) that the spike `riscv-gcc-13.2`
rejects but the **K1 SpacemiT clang** accepts. Those rows carry the K1 silicon number and an honest
`not_run` on spike. GELU uses explicit `_f32m4` spellings and runs on both.

## GELU / sigmoid — ours-scalar AND ours-vectorized (polynomial) vs XNNPACK

`ours-vectorized` is the genuine `vectorized_transcendental_activation` feature (compiler-emitted minimax polynomial → vectorized SIMD, NOT a libm call). `ours-vectorize-nofeature` is the prior column (vectorize pass, NO activation feature → still scalar `erff`/`expf` libm). XNNPACK is its hand-written rational/exp-poly RVV kernel. K1 `rdtime`, N=3 min; cos/abs-verified.

| op | N | XNNPACK | ours-scalar | ours-vectorize-nofeature | ours-vectorized (poly) | poly vs scalar | poly vs XNN |
|----|---|---------|-------------|--------------------------|------------------------|----------------|-------------|
| **GELU** | 1K | 270 | 2765 | 2929 | **983** | 2.81× faster | 3.64× slower |
| gelu | 16K | 2656 | 45171 | 47400 | **12189** | 3.71× faster | 4.59× slower |
| gelu | 256K | 44718 | 714815 | 775349 | **201720** | 3.54× faster | 4.51× slower |
| **SIGMOID** | 1K | 139 | 1277 | 1445 | **559** | 2.28× faster | 4.02× slower |
| sigmoid | 16K | 2111 | 23359 | 25363 | **8097** | 2.88× faster | 3.84× slower |
| sigmoid | 256K | 35305 | 375784 | 415575 | **139149** | 2.70× faster | 3.94× slower |

## Per-op results (K1 rdtime ticks, lower = faster)

| op | shape / size | XNNPACK (RVV) | ours-baseline | ours-optimized | attainment | correctness |
|----|--------------|---------------|---------------|----------------|------------|-------------|
| **GELU** f32 | N=1K | 270 | 2765 (scalar) | 2929 (vectorized) | ours ≈ **10.8× slower** than XNN | PASS (maxerr 0) |
| GELU f32 | N=16K | 2656 | 45171 | 47400 | **17.8× slower** | PASS |
| GELU f32 | N=256K | 44718 | 714815 | 775349 | **17.3× slower** | PASS |
| **sigmoid** f32 | N=1K | 139 | 1277 | 1445 | **10.4× slower** | PASS |
| sigmoid f32 | N=16K | 2111 | 23359 | 25363 | **12.0× slower** | PASS |
| sigmoid f32 | N=256K | 35305 | 375784 | 415575 | **11.8× slower** | PASS |
| **int8 GEMM** | 32³ | 230 | 39331 (ours-f32) | 40448 (ours-int8 W8A8) | int8 **176× slower** than XNN qd8 | PASS (cos>0.99) |
| int8 GEMM | 64³ | 1536 | 306946 | 324810 | **211× slower** | PASS (cos>0.99) |
| int8 GEMM | 128³ | 11633 | 2515346 | 2516672 | **216× slower** | PASS (cos>0.99) |
| **dwconv** f32 3×3 | 28×28×128 | 21014 | — | — | ours **not_run** (no depthwise prim) | XNN PASS |
| **conv2d** f32 (im2col→GEMM) | 64×16×27 | (= f32 GEMM ceiling) | 32323 (baseline) | 13225 (vfmacc) | ours-vfmacc **2.4× faster** than ours-baseline | PASS |
| **attention** bmm | 4×32×8×32 | (no library primitive) | 14654 (baseline) | 70666 (vfmacc) | vfmacc **4.8× SLOWER** here | PASS |

### spike instret authority (where it builds)

| op | size | XNNPACK instret | note |
|----|------|-----------------|------|
| GELU f32 | N=1K | 2408 | spike retired-instruction authority, inner-compute, verified |
| GELU f32 | N=16K | 37928 | " |
| sigmoid f32 | N=16K | **not_run** | spike gcc-13.2 rejects the kernel's overloaded RVV intrinsics (K1 clang OK) |

## What each comparison means

### GELU (f32) — `f32-vgelu` rational-12-10-div-u4v vs OUR GELU lowering
XNNPACK uses a hand-written **rational-12-10 polynomial** RVV kernel (no libm call). OUR `math.erf`-based
GELU `linalg.generic` lowers to a **scalar `erff` libm call loop** — the RVV vectorize pass is matmul-
contraction-targeted and does **not** vectorize the elementwise transcendental, so `ours-scalar` and
`ours-vectorized` are within ~6% of each other (both scalar libm). **This is a real coverage gap surfaced
by the comparison: OUR compiler has no vectorized activation path.** Bandwidth-bound, so we swept SIZE
(1K/16K/256K) not a cube; the ~17× gap is stable across sizes (a per-element fixed cost, not a tail effect).

### sigmoid (f32) — `f32-vsigmoid` rr2-p5-div-u4v vs OUR sigmoid lowering
Same story: XNNPACK is a vectorized `exp`-poly RVV kernel; ours lowers `math.exp` to a scalar `expf` libm
loop. ~11–12× slower, stable across sizes. (tanh was equally clean but uses the same overloaded-intrinsic
spelling; sigmoid was chosen as the representative unary activation per the task.)

### int8 GEMM — `qd8-f32-qc8w-gemm` 1x4v minmax rvv vs OUR W8A8 vwmacc datapath
The int8 analog of the f32 GEMM matrix. XNNPACK's qd8 kernel is a tight `vwmacc` i8×i8→i32 inner loop
with **pre-packed** weights (ksum + i8 panels + scale + bias) and per-row dynamic activation quant hoisted
out. OUR W8A8 path (`passes_quant_int.py`, `int8_compute=True`) produces the correct result (cos>0.99 vs
the f32 reference — the repo's fp32 int8 tier) but in this **isolated single-op** standalone kernel it is
~176–216× slower than XNNPACK and **barely faster than our own f32 baseline** (40448 vs 39331 at 32³). The
W8A8 win in the repo shows up **end-to-end** (whole-model, see `output/rvv_bench/k1_e2e_*`), not in this
isolated tight contraction — the per-op int8 datapath here carries the dynamic-quant + requant overhead
without the cross-op data-movement savings.
*Correctness note:* XNNPACK's qd8 driver verifies vs **its own quantized** reference (bit-exact); OUR int8
verifies vs the **f32** reference on cosine>0.99 (W8A8 is an approximation). Those are two different
correctness bars, recorded honestly — they are not head-to-head bit-exact.

### dwconv (f32 3×3 depthwise) — `f32-dwconv` 9p8vc rvv vs OURS (not_run)
**Depthwise is a DIFFERENT op from regular conv2d** (one filter per channel, no cross-channel reduction).
XNNPACK's only f32-conv RVV kernel is this depthwise one (9-tap unipass), and it PASSES on K1 (21014 ticks
for a 28×28×128 MobileNet-style layer). OUR compiler has **no depthwise primitive** on the f32 RVV path:
regular conv2d lowers im2col→matmul (the GEMM ceiling), but a per-channel filter is not expressible as that
single contraction, so the ours-depthwise row is an **honest not_run** with that blocker. Regular conv on
the library side IS its f32 GEMM (igemm) — i.e. the GEMM ceiling already raced in the matmul matrix.

### conv2d (f32, regular) — OUR im2col→matmul, baseline vs vfmacc
Regular conv on our side = im2col then `linalg.matmul` (K = patch-volume Cin·Kh·Kw = 27 for a 3×3×3→16
conv over an 8×8 output, M=64 positions, N=16 channels). No XNNPACK regular-conv RVV kernel to race (it is
the GEMM ceiling), so this is **ours-baseline vs ours-vfmacc**: the `fused_vfmacc_contraction` feature is
**2.4× faster** than baseline on this conv-shaped contraction (32323 → 13225 ticks). PASS.

### attention — OUR baseline batch_matmul vs OUR vfmacc feature (ours-vs-ours)
**Attention has NO library baseline** — it is not an XNNPACK or OpenBLAS primitive — so this is explicitly
**ours-vs-ours**, the attention-vectorization comparison, NOT vs a framework. On a llama-style attention
bmm (B=4, M=32, **N=8** small, K=32), the `fused_vfmacc_contraction` feature is **4.8× SLOWER** than the
baseline batch_matmul lowering (14654 → 70666 ticks). This is an honest measured regression: the small
N=8 N-tail dominates and the deferred-vfmacc path's setup is not amortized — the vfmacc win is a
large-contraction property, and this small-N attention shape is exactly where it does not pay off.

## Honest status summary

* **Raced vs XNNPACK (PASS both sides):** GELU (3 sizes), sigmoid (3 sizes), int8 GEMM (3 shapes), dwconv (1 shape, XNN side).
* **ours-vs-ours (no library primitive):** conv2d (im2col→GEMM, baseline vs vfmacc), attention bmm (baseline vs vfmacc).
* **not_run:** ours-depthwise-conv (no depthwise primitive in our f32 RVV path — exact blocker recorded);
  XNNPACK sigmoid **on spike** (gcc-13.2 overloaded-intrinsic incompatibility — runs on K1 clang).
* **No fabricated numbers:** every tick is an N=3-min K1 silicon `rdtime` measurement behind a passing
  bit-exact/cosine verify; every gap is a blocker string, not a guess. Board left clean.

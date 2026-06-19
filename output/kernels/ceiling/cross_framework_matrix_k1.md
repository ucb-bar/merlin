# Cross-framework fp32 GEMM ceiling matrix — REAL K1 silicon (rdtime)

Same five columns + scope as the spike matrix (`cross_framework_matrix.md`), but measured on the **SpacemiT K1 board** (Bianbu Linux, VLEN=256, X60 @ ~1.6 GHz), `mode=inner_compute`, bit-exact verified vs a scalar reference. Timer = the board's delegated **`rdtime`** counter (24 MHz fixed timebase; K1 traps userspace `rdcycle`), so the cells are real-silicon **rdtime ticks** — `cycle_accurate=false` (spike/FireSim remain cycle-accurate authorities). Experts reuse the SAME driver C the spike matrix used (`openblas_sgemm_driver.c` / `xnnpack_gemm_driver.c`), compiled UNCHANGED with the SpacemiT clang for `riscv64-linux -march=rv64gcv -mabi=lp64d -O3 -ffast-math`, with `k1_harness/util.h` mapping `read_csr(mcycle)`→`rdtime` and printf→glibc. Ours = the frozen `hand_v0` schedule + the named default-off impr feature, lowered exactly as the runner does.

## rdtime ticks (inner-compute; lower = faster; 1 tick ≈ 41.7 ns @ 24 MHz)

| shape (M=N=K) | OpenBLAS | XNNPACK | ours-baseline | ours-vfmacc | ours-tiled |
|---|---|---|---|---|---|
| 32^3 | 393 | 242 | 40,191 | 3,860 | 2,730 |
| 64^3 | 3,552 | 1,966 | 307,870 | 57,230 | 20,376 |

## Ordering vs spike — DOES IT RE-RANK? (the interesting question)

**YES — the two experts SWAP on real K1 silicon.**

| shape | spike: faster expert | K1: faster expert | flip? |
|---|---|---|---|
| 32^3 | OpenBLAS (11,039 < 13,289 XNN) | **XNNPACK (242 < 393 OB)** | YES |
| 64^3 | OpenBLAS (84,483 < 101,705 XNN) | **XNNPACK (1,966 < 3,552 OB)** | YES |

On the **functional spike** the cycle proxy is just retired-instruction count (IPC=1), where OpenBLAS's tight 8x8 register kernel retires fewer instructions than XNNPACK and so 'wins'. On the **real K1 VPU** the ordering FLIPS: XNNPACK's `xnn_f32_gemm_ukernel_1x4v__rvv` adapts its N-tile to the wider VLEN (NR=`vsetvlmax_e32m4` = **32 lanes at VLEN=256**, vs 16 at spike's VLEN=128) and runs at LMUL=4, so on a real vector pipeline it is ~1.6-1.8x faster than OpenBLAS's fixed `zvl128b` 8x8 kernel (which is tuned for VLEN=128 and leaves half the K1's 256-bit lanes idle). This is exactly the 'a real VPU may re-rank vector-heavy kernels' effect the spike instruction-count proxy cannot see — the spike ranking by instruction count is NOT a faithful predictor of K1 wall-time ordering between the two experts.

The OURS ordering is stable across substrates: baseline >> vfmacc-contraction > tiled (tiled fastest of ours) on BOTH spike and K1. On K1 the tiled vfmacc is ~15.1x faster than ours-baseline @ 64^3 (307,870 → 20,376 ticks) — the same lever that gave the 9.35x whole-model bitvla e2e speedup.

## Attainment vs the best expert (K1 ticks)

| shape | best expert (XNNPACK) | ours-best (tiled) | best-expert / ours-best |
|---|---|---|---|
| 32^3 | 242 | 2,730 | 0.0886x (ours 11.3x slower) |
| 64^3 | 1,966 | 20,376 | 0.0965x (ours 10.4x slower) |

Ours-tiled trails the best K1 expert by ~11x (32^3) and ~10x (64^3) — a real-silicon gap (the experts pre-pack operands into contiguous panels + run at higher LMUL; ours emits strided per-tile transfers at the fixed [4,16,16] tile). This is the same residual the spike matrix attributed to packing (mined `packed_rhs_policy`), now confirmed on silicon.

## Caveats (read before trusting the numbers)

- **rdtime is coarse + noisy at small counts.** At 24 MHz one tick ≈ 41.7 ns, so the small expert counts (242-3552 ticks) carry quantization + Linux-scheduler noise. The table reports a single run; a 3x repeat at 64^3 gives OpenBLAS min 2,288 (range 2,288-3,552) and XNNPACK min 1,960 (range 1,960-3,185), so **the XNNPACK-beats-OpenBLAS flip holds on the min and the typical** (XNNPACK ~1.2-1.6x faster), and the ours-vs-expert gaps are >10x — well outside the noise. Do not over-read a single tick. `INSTRET` is 0 on K1 (the userspace `minstret` CSR is not delegated — no retired-instruction count on the board).
- **Same substrate / scope for all five columns.** Every column is a standalone `riscv64-linux` binary, inner-compute timed with the same `rdtime` path, operand pack / memref-descriptor build hoisted OUT of the timed region, bit-exact verified. Ours times `_mlir_ciface_forward` (compiler-emitted `linalg.fill` + `linalg.matmul`) — GEMM + a thin wrapper, NOT the whole-model runner number.
- **K1 real-silicon vs the spike PROXY.** This matrix is the real-VPU companion to the spike matrix; where they disagree (the expert flip), the K1 is the silicon truth for wall-time ordering and spike is only an instruction-count proxy.

---
title: "Design note: where the whole-model codegen-vs-hand-C gap lives (it is the emitted matmul)"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [expert_gap_attribution]
code_refs: [build_tools/scripts/k1_codegen_vs_handc.py, merlin/python/merlin/mining/k1.py, merlin/python/merlin/llvmlower/impr_features.py, merlin/runtime/backends/ours_board/ours_gemm_rvv_shim.c]
---

# Where the whole-model codegen-vs-hand-C gap lives

## The unexplained number

On rdt2 whole-model, same runtime for every row, our COMPILER
(`accumulator_resident_wholemodel_vf`) was **30,235 ms** while our own hand-written C RVV GEMM
shim (`runtime/backends/ours_board/ours_gemm_rvv_shim.c`) was **20,154 ms** — our codegen ~1.5x
slower than plain hand-C, a ~10 s gap nobody had attributed. Both compute the same matmuls in the
same runtime, so the gap is either (a) our emitted matmul being worse than the hand-C kernel, or
(b) the `vf` feature making the non-matmul ~80% of the model worse. Prior work could not separate
these: our matmul is inlined into the one monolithic `_mlir_ciface_forward`, with no call boundary
to time.

## Method: a 2x2 factorial that creates the missing boundary

`kernel_backend="ours"` rewrites the routable f32 `linalg.matmul` ops to shim calls on the
*prepared* MLIR, **before** `lower_model_file(..., features=...)` applies the compiler feature. So
the two knobs compose:

| | matmul = our codegen | matmul = hand-C shim (rdtime-timed) |
|---|---|---|
| non-matmul = baseline | A | C (hand-C arm) |
| non-matmul = vf | B (codegen arm) | **D (the arm nobody ran)** |

In both C and D every matmul is routed to the same shim (routing is total: `n_routed` == the
model's `linalg.matmul` count), so C and D differ *only* in what `vf` does to the non-matmul model.
With `dispatch_bucket = wall − matmul_bucket` and `matmul_bucket` measured by the shim's own
`-DMERLIN_DISPATCH_TIMING` rdtime bracket, every term is measured, not attributed:

    non_matmul_delta = D.dispatch − C.dispatch          # vf's cost on the non-matmul 80%
    codegen_matmul   = B.wall − D.dispatch              # our emitted matmul, isolated at last
    handc_matmul     = C.matmul_bucket                  # the shim's rdtime bracket

Driver: `build_tools/scripts/k1_codegen_vs_handc.py` (cos-gated >= 0.9999 per arm; a `ctrl` arm
re-runs C's config for the campaign noise floor).

## Result: the whole gap is the emitted matmul

Measured on the SpacemiT K1 (VLEN=256), two models:

| term | rdt2 (n=3) | bitvla (n=5) |
|---|---|---|
| gap B − C | **10.05 s (1.50x)** — reproduces the headline | 105 ms (1.59x) |
| non-matmul delta (D.dispatch − C.dispatch) | **−0.65 s (−6.5% of gap)** | −58 ms (−55% of gap) |
| codegen matmul vs hand-C matmul | **14.04 s vs 3.33 s = 4.21x (106.5% of gap)** | 195 ms vs 32.5 ms = 6.0x |
| control vs C (noise floor) | −0.07% | −0.41% |
| shim reproducibility (D vs C matmul bucket) | 0.07% | 0.2% |

Both models agree: **the entire codegen-vs-hand-C gap is the emitted matmul.** The `vf` feature's
non-matmul lowering is actually slightly *faster* than baseline (negative delta), so the non-matmul
80% of the model is refuted as the cause.

## Why the emitted matmul is 4-6x slower: the MR (A-reuse) clamp

The `vf` feature clamps `MR_mm=1` for whole-model safety. That gives one B-row load per single
`vfmacc.vf` — 2.0 loads/useful-FMA, **zero A-operand reuse**. The hand-C shim defaults to MR=4,
sharing one B-row load across 4 `vfmacc.vf` into 4 resident accumulators (1.25 loads/FMA). An MR
sweep on the *same shim kernel* (dynamic VL, so VLEN is held constant) isolates this lever on the
matmul bucket alone (bitvla):

| MR | matmul bucket | speedup vs MR=1 |
|---|---|---|
| 1 | 117.6 ms | 1.00x |
| 2 | 62.2 ms | 1.89x |
| 4 | 34.1 ms | **3.45x** |
| 7 | 24.2 ms | 4.87x |

MR=1 -> MR=4 is 3.45x on the matmul bucket — the dominant term of the 4-6x codegen-matmul penalty.
A second, smaller term is the VLEN pin (below).

## Secondary finding: the VLEN pin does not reach the whole model

`codegen_march()` pins the board's real VLEN (`_zvl256b`), which turns a fixed `vector<16xf32>`
from `e32,m4` (512b sized for the RVV *minimum* zvl128b — vl=16 in an m4 group is **half the
VLEN=256 datapath idle**) into `e32,m2` (fully utilized, half the register pressure). Verified from
the same `model.ll`: `-march=...zvl128b` emits `e32,m4`; `-march=...zvl256b` emits `e32,m2`.

But as of commit 54312e6 the whole-model `model.o` compile in `build_k1_binary` passes
**`K1_MARCH`** (zvl128b), not `codegen_march()` — so the emitted `forward` runs half-idle
(`model.o` `.riscv.attributes` show `zvl128b`, and the disassembled `forward` K-loop is
`vsetivli zero, 0x10, e32, m4`). Whole-model A/B (bitvla, applying the pin to `model.o`):
**~3-5% at the min, ~3% on medians, vs a 2.66% control band** — i.e. small and near the noise
floor, NOT the 1.60x the pin delivers on a standalone 128^3 GEMM. The pin only helps the matmul
inner loop, and matmul is a minority of whole-model time, so the kernel-only 1.60x is diluted away.
An uncommitted edit in the main working tree already switches this line to `codegen_march()`;
committing it is correct but recovers only a few percent whole-model.

## What would close it, with evidence

The dominant, cheap, safe lever is MR (A-operand reuse):

- **bitvla (M=32, every matmul M%4==0): enabling MR=4 codegen already works.** The existing
  `accumulator_resident_wholemodel_vf_mr4` feature runs whole-model, cos-identical
  (0.9999946), **272 ms -> 143 ms = 1.90x**, landing next to the hand-C shim arm (D = 119 ms).
- **rdt2 is NOT safe under a global MR=4.** It has M=1 matmuls (the 2048x9216 projections); MR=4
  on M=1 trips the LLVM-23 masked-`transfer_write` PipelineError and the whole model falls back to
  a scalar matmul (`lower_scalar`) — confirmed by build probe. So a single global MR cannot serve
  rdt2.

The general fix is **per-matmul MR selection** — MR = 4 where M%4==0, MR=1 (or a padded tail)
otherwise — which is exactly how the hand-C shim gets MR=4 everywhere: it pads M up to MR
internally (`round_up_mr`) in plain C, so there is no MLIR masked write to legalize. Emitting that
(via `MERLIN_PAD_M`-style M-padding before tiling, or an op-level MR chosen from each matmul's
static M) gives the codegen the shim's kernel and closes the ~10 s rdt2 gap at its dominant term.

## What was refuted

- **"The non-matmul ops / dispatch glue are the gap."** No — the non-matmul term is negative on
  both models (vf's non-matmul lowering is slightly faster than baseline).
- **"The VLEN pin is the whole-model win."** No — it is ~3-5% whole-model (near the noise floor),
  not the kernel-only 1.60x; and the committed path does not even apply it to `model.o`.
- **"Our matmul kernel is fundamentally worse than the expert's."** No — our own hand-C shim (same
  inlined v3 micro-kernel) at MR=4 is within 8.5% of XNNPACK; the codegen just fails to emit the
  MR>1 register block the shim uses.

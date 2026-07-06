# AGENT.md — merlin/python/merlin/baselines/exo_kernels

## Purpose

EXO kernel sources for the K1-RVV whole-model baseline arm.

## Modules

- `gemm.py` — EXO f32 RVV GEMM kernel (vfmacc.vf, 8-wide) for the K1 whole-model glue runtime.
- `igemm.py` — EXO int8 RVV GEMM kernel (vwmacc.vx widening i16×i16→i32, 16-wide) for the W8A8 path.
  `build_igemm(ku, U)`: **U = output-register blocking** — U distinct 16-wide i32 accumulators
  (`Yr0..Yr{U-1}`) share ONE scalar `A[m,k]` load per k (U vwmacc.vx per load), the RVV-ceiling
  lever (U=1→8 lifts kernel RVV 0.17→0.31). `ku` = k-unroll (u=1 path). EXO note: RVV vector C types
  are *sizeless* so `vint32m2_t Yb[U]` (an arrayed register) is illegal C — the U-blocked schedule
  unrolls the tile loop first, then stages each tile into its own named register.
- `glue_ops.py` — EXO f32 RVV elementwise kernels: `residual_add_rvv` (vfadd.vv) + `ewise_mul_rvv`
  (vfmul.vv) — the glue's residual-add and SwiGLU product moved off the scalar path.
- `rvv256.py` — RVV register classes + intrinsics for the SpacemiT K1 X60 (VLEN=256): 8-wide f32
  (`RVV256`, +vfadd/vfmul) and the int8 widening path (`RVV256_I16` inputs, `RVV256_I32` m2
  accumulator, vwmacc).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Non-Python kernel sources

- `llama_glue.c` — fp32 whole-model TinyLlama glue runtime (calls `gemm_nt_ref` per Linear).
- `llama_glue_int8.c` — int8 W8A8 whole-model glue. Hand-RVV pre-pass: `quant_row_rvv` (per-token
  abs-max reduce + scale + f32→i32 convert + narrow-to-i16) and `transpose_widen_rvv` (widening i8→
  i16 load + strided `vsse16` store) — the previously-scalar "other" region. GEMM via `igemm_nt_ref`
  vwmacc; residual-add + SwiGLU product via the `glue_ops` RVV kernels; scalar requant. Emits
  `MERLIN_E2E`/`MERLIN_REGION` (incl. separate `quant`/`transpose` regions); norm/RoPE/attention/
  SiLU-sigmoid-exp + requant remain scalar glue, labeled as `ScalarFallback`.
- `igemm_bench.c` — standalone int8-GEMM micro-benchmark for the on-board U-blocking autotune (one
  `BENCH_TICKS` line per candidate U).

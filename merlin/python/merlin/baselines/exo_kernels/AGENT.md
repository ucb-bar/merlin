# AGENT.md — merlin/python/merlin/baselines/exo_kernels

## Purpose

EXO kernel sources for the K1-RVV whole-model baseline arm.

## Modules

- `gemm.py` — EXO f32 RVV GEMM kernel (vfmacc.vf, 8-wide) for the K1 whole-model glue runtime.
- `igemm.py` — EXO int8 RVV GEMM kernel (vwmacc.vx widening i16×i16→i32, 16-wide) for the W8A8 path;
  `build_igemm(ku)` k-unrolls the reduction by `ku` (the autotune knob).
- `glue_ops.py` — EXO f32 RVV elementwise kernels: `residual_add_rvv` (vfadd.vv) + `ewise_mul_rvv`
  (vfmul.vv) — the glue's residual-add and SwiGLU product moved off the scalar path.
- `rvv256.py` — RVV register classes + intrinsics for the SpacemiT K1 X60 (VLEN=256): 8-wide f32
  (`RVV256`, +vfadd/vfmul) and the int8 widening path (`RVV256_I16` inputs, `RVV256_I32` m2
  accumulator, vwmacc).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Non-Python kernel sources

- `llama_glue.c` — fp32 whole-model TinyLlama glue runtime (calls `gemm_nt_ref` per Linear).
- `llama_glue_int8.c` — int8 W8A8 whole-model glue (per-token activation quant → `igemm_nt_ref`
  vwmacc → scalar requant; residual-add + SwiGLU product via the `glue_ops` RVV kernels). Both emit
  `MERLIN_E2E`/`MERLIN_REGION` markers; norm/RoPE/attention/SiLU-sigmoid + int8 quant/requant remain
  scalar glue, labeled as `ScalarFallback`.
- `igemm_bench.c` — standalone int8-GEMM micro-benchmark for the on-board k-unroll autotune (one
  `BENCH_TICKS` line per candidate `ku`).

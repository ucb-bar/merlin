# bareMetalC corroboration report (capsule_bench_v0)

Real Gemmini reference programs (the canonical primitives the bareMetalC corpus uses: the
upstream `mvin_mvout` movement test + `tiled_matmul_auto`, the library matmul called by
`conv_perf.c`/`tiled_matmul_ws.c`) built with the IDENTICAL toolchain/flags as our package
ELFs and run on the SAME spike + verilator. Inputs use the same deterministic formula as our
capsule leaves, so the real-Gemmini output must equal our `Tensor` golden (== capsule golden).
These are external reference ORACLES only — never copied/called inside any submission.

**The anchor:** `real_gemmini_output == our_Tensor_golden == capsule_golden`.

| anchor | reference specimen | equiv capsule | feature | built | spike match (cyc) | verilator match (cyc) | status |
|---|---|---|---|---|---|---|---|
| mvin_mvout | bareMetalC/mvin_mvout.c (upstream, instrumented) | A1_mvin_mvout | MVIN/MVOUT movement (identity, i8) | yes | yes (None) | yes (None) | pass |
| ref_matmul_16 | tiled_matmul_auto (canonical Gemmini lib), WS | A2_single_tile_matmul | single-tile i8xi8->i32 matmul | yes | yes (None) | yes (None) | pass |
| ref_matmul_k32 | tiled_matmul_auto, K=32 (K-accumulation) | A3_k_accumulation | K-accumulation (Kt>1) | yes | yes (None) | yes (None) | pass |
| ref_matmul_relu | tiled_matmul_auto + RELU | A5_relu_epilogue | relu epilogue | yes | yes (None) | yes (None) | pass |
| ref_acc_scale_i8 | tiled_matmul_auto + acc_scale->i8 | A4_acc_scale_i8 | acc_scale (f32) + saturating i8 readout | yes | yes (None) | yes (None) | pass |

**5/5 anchors corroborated.**

## Interpretation

- A passing anchor means our golden engine (and thus the capsule it backs) reproduces real
  Gemmini hardware output bit-exactly for identical inputs — closing the 'goldens are only
  our own engine' gap for movement, single-tile matmul, K-accumulation, relu, and
  acc_scale→i8.
- conv2d corroboration is deferred: spike's Gemmini ISS does not run conv, and a verilator
  conv anchor is future work (recorded honestly, not silently omitted).
- These reference programs are NOT part of any submission; the integrity scan + ABI boundary
  forbid copying/calling Gemmini library kernels in a graded backend.

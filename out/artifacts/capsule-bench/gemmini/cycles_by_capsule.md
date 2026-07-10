# Cycles by capsule (gemmini_capsule_bench_v0)

Per-capsule status + **L2 spike** / **L3 verilator (cycle-accurate RTL)** cycle counts, from each run's `capsule_result.json`. L5 FireSim columns are added once the FPGA backfill runs.

## L3 (verilator) cycle matrix

| capsule | merlin/pilot_merlin_0001 | raw_ba/dry_0002 | raw_ba/rb_pilot_0001 | raw_ba/rb_pilot_0002 | raw_ba/rb_pilot_cpp_01 | raw_ba/rb_pilot_rep_01 |
|---|---|---|---|---|---|---|
| A0_config_smoke | 298 | · | fail | 311 | 299 | 296 |
| A2_single_tile_matmul | 298 | 308 | fail | 311 | 299 | 296 |
| A4_acc_scale_i8 | 261 | · | fail | 274 | 278 | 275 |
| B0_quantized_linear_i8 | 328 | · | fail | 318 | 340 | 339 |
| B3_conv2d_im2col_i8 | · | 1011 | · | · | · | · |
| H0_matmul_hidden | 298 | 308 | fail | 311 | 299 | 296 |
| H1_acc_scale_hidden | 261 | · | fail | 274 | 278 | 275 |
| H2_k_accum_hidden | 364 | · | fail | 344 | 375 | 377 |

## merlin_assisted/pilot_merlin_0001 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A0_config_smoke | public | pass | 50 | 298 |
| A2_single_tile_matmul | public | pass | 50 | 298 |
| A4_acc_scale_i8 | public | pass | 51 | 261 |
| B0_quantized_linear_i8 | public | pass | 67 | 328 |
| H0_matmul_hidden | hidden | pass | 50 | 298 |
| H1_acc_scale_hidden | hidden | pass | 51 | 261 |
| H2_k_accum_hidden | hidden | pass | 65 | 364 |

## raw_baseline/dry_0002 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A2_single_tile_matmul | public | pass | 47 | 308 |
| B3_conv2d_im2col_i8 | public | pass | 94 | 1011 |
| H0_matmul_hidden | hidden | pass | 47 | 308 |

## raw_baseline/rb_pilot_0001 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A0_config_smoke | public | fail | None | None |
| A2_single_tile_matmul | public | fail | None | None |
| A4_acc_scale_i8 | public | fail | None | None |
| B0_quantized_linear_i8 | public | fail | None | None |
| H0_matmul_hidden | hidden | fail | None | None |
| H1_acc_scale_hidden | hidden | fail | None | None |
| H2_k_accum_hidden | hidden | fail | None | None |

## raw_baseline/rb_pilot_0002 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A0_config_smoke | public | pass | 52 | 311 |
| A2_single_tile_matmul | public | pass | 52 | 311 |
| A4_acc_scale_i8 | public | pass | 54 | 274 |
| B0_quantized_linear_i8 | public | pass | 65 | 318 |
| H0_matmul_hidden | hidden | pass | 52 | 311 |
| H1_acc_scale_hidden | hidden | pass | 54 | 274 |
| H2_k_accum_hidden | hidden | pass | 63 | 344 |

## raw_baseline/rb_pilot_cpp_01 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A0_config_smoke | public | pass | 50 | 299 |
| A2_single_tile_matmul | public | pass | 50 | 299 |
| A4_acc_scale_i8 | public | pass | 52 | 278 |
| B0_quantized_linear_i8 | public | pass | 61 | 340 |
| H0_matmul_hidden | hidden | pass | 50 | 299 |
| H1_acc_scale_hidden | hidden | pass | 52 | 278 |
| H2_k_accum_hidden | hidden | pass | 59 | 375 |

## raw_baseline/rb_pilot_rep_01 — per-capsule L2/L3

| capsule | phase | status | L2 spike | L3 verilator |
|---|---|---|---|---|
| A0_config_smoke | public | pass | 50 | 296 |
| A2_single_tile_matmul | public | pass | 50 | 296 |
| A4_acc_scale_i8 | public | pass | 52 | 275 |
| B0_quantized_linear_i8 | public | pass | 61 | 339 |
| H0_matmul_hidden | hidden | pass | 50 | 296 |
| H1_acc_scale_hidden | hidden | pass | 52 | 275 |
| H2_k_accum_hidden | hidden | pass | 59 | 377 |

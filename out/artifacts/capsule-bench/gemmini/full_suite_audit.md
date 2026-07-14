# Full-suite audit (capsule_bench_v0) — all 25 capsules, RTL oracle

Corpus: `/path/to/merlin/bench_contract/capsules` · 25 capsules · tiers ['L2', 'L3'] · 8 parallel workers. Cycle counts are **L3 verilator (cycle-accurate RTL)**. Backends were built against the 4-capsule pilot only — failures on unimplemented classes (conv, attention) are expected and reported honestly, not hidden.

## Headline

| backend | lang | passed (all) | public | hidden | tier | audit wall(s) | sim_active(s) | oracle_wait(s) | speedup |
|---|---|---|---|---|---|---|---|---|---|
| rb_abc4 | cpp | 25/25 | 20/20 | 5/5 | L3 | 834.3 | 5540.582 | 0.084 | 6.67 |

## Coverage by workload class

| class | n | rb_abc4 |
|---|---|---|
| attention | 5 | 5/5 |
| conv | 3 | 3/3 |
| matmul | 7 | 7/7 |
| matmul+acc_scale | 3 | 3/3 |
| matmul+relu | 3 | 3/3 |
| mlp | 2 | 2/2 |
| movement | 2 | 2/2 |

## Per-capsule matrix (status · L3 cycles)

| capsule | label | class | rb_abc4 |
|---|---|---|---|
| H0_matmul_hidden | hidden | matmul | pass · 299cyc |
| H1_acc_scale_hidden | hidden | matmul+acc_scale | pass · 250cyc |
| H2_k_accum_hidden | hidden | matmul | pass · 371cyc |
| H3_movement_hidden | hidden | movement | pass · 142cyc |
| H4_conv_hidden | hidden | conv | pass · 775cyc |
| A0_config_smoke | public | matmul | pass · 299cyc |
| A1_mvin_mvout | public | movement | pass · 142cyc |
| A2_single_tile_matmul | public | matmul | pass · 299cyc |
| A3_k_accumulation | public | matmul | pass · 371cyc |
| A4_acc_scale_i8 | public | matmul+acc_scale | pass · 250cyc |
| A5_relu_epilogue | public | matmul+relu | pass · 299cyc |
| A6_resident_reuse | public | matmul | pass · 441cyc |
| A7_edge_padding | public | matmul | pass · 488cyc |
| B0_quantized_linear_i8 | public | matmul+acc_scale | pass · 335cyc |
| B1_linear_relu_i8 | public | matmul+relu | pass · 371cyc |
| B2_linear_acc_scale_relu_i8 | public | matmul+relu | pass · 335cyc |
| B3_conv2d_im2col_i8 | public | conv | pass · 775cyc |
| B4_conv2d_relu_i8 | public | conv | pass · 775cyc |
| C0_mlp_linear1 | public | mlp | pass · 1209cyc |
| C1_mlp_activation_linear2 | public | mlp | pass · 1209cyc |
| C2_attention_q_projection | public | attention | pass · 509cyc |
| C3_attention_k_projection | public | attention | pass · 509cyc |
| C4_attention_v_projection | public | attention | pass · 509cyc |
| C5_attention_qk_matmul | public | attention | pass · 299cyc |
| C6_attention_pv_matmul | public | attention | pass · 299cyc |

_Legend: cycles = L3 verilator RTL cycles (rdcycle-bracketed). oracle_wait(s) is time blocked on a queue/FPGA slot (≈0 for local verilator; nonzero only for queued VCS/FireSim). speedup = sum(active_sim)/wall under parallel workers._

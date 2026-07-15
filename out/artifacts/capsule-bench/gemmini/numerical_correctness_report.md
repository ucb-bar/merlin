# Numerical correctness report (capsule_bench_v0)

Integer capsules use **exact equality** three-way: golden == reference(cb) == simulate(cb) == oracle. No tolerance.

| capsule | policy | numeric | mismatch_count | max_abs_diff |
|---|---|---|---|---|
| A0_config_smoke | exact_int | pass | 0 | 0 |
| A1_mvin_mvout | exact_int | pass | 0 | 0 |
| A2_single_tile_matmul | exact_int | pass | 0 | 0 |
| A3_k_accumulation | exact_int | pass | 0 | 0 |
| A4_acc_scale_i8 | exact_int | pass | 0 | 0 |
| A5_relu_epilogue | exact_int | pass | 0 | 0 |
| A6_resident_reuse | exact_int | pass | 0 | 0 |
| A7_edge_padding | exact_int | pass | 0 | 0 |
| B0_quantized_linear_i8 | exact_int | pass | 0 | 0 |
| B1_linear_relu_i8 | exact_int | pass | 0 | 0 |
| B2_linear_acc_scale_relu_i8 | exact_int | pass | 0 | 0 |
| B3_conv2d_im2col_i8 | exact_int | pass | 0 | 0 |
| B4_conv2d_relu_i8 | exact_int | pass | 0 | 0 |
| C0_mlp_linear1 | exact_int | pass | 0 | 0 |
| C1_mlp_activation_linear2 | exact_int | pass | 0 | 0 |
| C2_attention_q_projection | exact_int | pass | 0 | 0 |
| C3_attention_k_projection | exact_int | pass | 0 | 0 |
| C4_attention_v_projection | exact_int | pass | 0 | 0 |
| C5_attention_qk_matmul | exact_int | pass | 0 | 0 |
| C6_attention_pv_matmul | exact_int | pass | 0 | 0 |
| H0_matmul_hidden | exact_int | pass | 0 | 0 |
| H1_acc_scale_hidden | exact_int | pass | 0 | 0 |
| H2_k_accum_hidden | exact_int | pass | 0 | 0 |
| H3_movement_hidden | exact_int | pass | 0 | 0 |
| H4_conv_hidden | exact_int | pass | 0 | 0 |

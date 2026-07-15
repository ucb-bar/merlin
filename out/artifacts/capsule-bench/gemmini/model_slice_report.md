# Model-slice report (capsule_bench_v0)

MLP + attention Gemmini-relevant matmul slices (no softmax). Q/K/V projections, QK^T (K pre-transposed as the resident weight leaf), PV — all weight-stationary matmuls.

| slice | semantic | status | L2(cyc) | L3(cyc) |
|---|---|---|---|---|
| C0_mlp_linear1 | mlp_linear1 | pass | 175 | 1888 |
| C1_mlp_activation_linear2 | mlp_relu_linear2 | pass | 175 | 1888 |
| C2_attention_q_projection | attn_q_proj | pass | 72 | 587 |
| C3_attention_k_projection | attn_k_proj | pass | 72 | 587 |
| C4_attention_v_projection | attn_v_proj | pass | 72 | 587 |
| C5_attention_qk_matmul | attn_qk | pass | 47 | 308 |
| C6_attention_pv_matmul | attn_pv | pass | 47 | 308 |

# C4_attention_v_projection

Model-slice capsule (attn_v_proj). Single weight-stationary matmul [16x64] x [64x16] -> output_dtype=i32. Golden: torch_unavailable (golden source = merlin_tensor_int).

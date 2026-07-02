# M07_tiny_llama_lm_16x2048x5632_i8

Model-slice capsule (model_matmul_model:tiny_llama/lm/matmul_7). Single weight-stationary matmul [16x2048] x [2048x5632] -> output_dtype=i8. Golden: torch_unavailable (golden source = merlin_tensor_int).

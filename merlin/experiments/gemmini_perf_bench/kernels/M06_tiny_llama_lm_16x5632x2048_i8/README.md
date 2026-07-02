# M06_tiny_llama_lm_16x5632x2048_i8

Model-slice capsule (model_matmul_model:tiny_llama/lm/matmul_9). Single weight-stationary matmul [16x5632] x [5632x2048] -> output_dtype=i8. Golden: torch_unavailable (golden source = merlin_tensor_int).

# M05_tiny_llama_lm_16x2048x32000_i8

Model-slice capsule (model_matmul_model:tiny_llama/lm/matmul_19). Single weight-stationary matmul [16x2048] x [2048x32000] -> output_dtype=i8. Golden: torch_unavailable (golden source = merlin_tensor_int).

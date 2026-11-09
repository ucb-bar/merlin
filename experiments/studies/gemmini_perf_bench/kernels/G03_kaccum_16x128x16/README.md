# G03_kaccum_16x128x16

Model-slice capsule (golden_matmul_baremetalc:tiled_matmul_ws.c). Single weight-stationary matmul [16x128] x [128x16] -> output_dtype=i32. Golden: torch_unavailable (golden source = merlin_tensor_int).

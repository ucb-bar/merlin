# RP3_matmul_bf16_k32_pt

RP3_matmul_bf16_k32_pt: matmul over W[32, 16]:bf16, A0[16, 32]:bf16, authored from PyTorch bf16 GEMM, K=32 (K-accumulate).

kind=isa label=public op=matmul modes={'k_accumulate': True}

# R2_gemm_bf16

R2_gemm_bf16: matmul over W[16, 16]:bf16, A0[16, 16]:bf16, authored from SIMT bf16 GEMM tile (bf16 operands, fp32 accumulate).

kind=isa label=public op=matmul modes={}

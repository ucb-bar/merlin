# R1_gemm_fp16

R1_gemm_fp16: matmul over W[32, 16]:fp16, A0[16, 32]:fp16, authored from SIMT fp16 GEMM, K=32 (fp16 operands, fp32 accumulate).

kind=isa label=public op=matmul modes={'k_accumulate': True}

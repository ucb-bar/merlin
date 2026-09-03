# M4_mxfp4_k_accum

M4_mxfp4_k_accum: matmul over W[64, 32]:mxfp4, A0[32, 64]:mxfp4, authored from mxfp4 K=64 -> 2 block-scale groups, bf16 K-accumulate.

kind=isa label=public op=matmul modes={'k_accumulate': True}

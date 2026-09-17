# M1_mxfp8_k_accum

M1_mxfp8_k_accum: matmul over W[64, 16]:mxfp8, A0[16, 64]:mxfp8, authored from mxfp8 K=64 -> 2 block-scale groups (per-group E8M0), bf16 K-accumulate.

kind=isa label=public op=matmul modes={'k_accumulate': True}

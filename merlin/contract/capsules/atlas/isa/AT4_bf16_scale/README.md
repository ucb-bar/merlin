# AT4_bf16_scale

AT4_bf16_scale: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from matmul + bf16 output scale (VPU requant precursor).

kind=isa label=public op=matmul modes={'acc_scale': True}

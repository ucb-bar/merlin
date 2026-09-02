# PF04_matmul_m32k64n32

PF04_matmul_m32k64n32: matmul over W[64, 32]:fp8_e4m3, A0[32, 64]:fp8_e4m3, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=32; K=64, M=32, N=32; op=matmul).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

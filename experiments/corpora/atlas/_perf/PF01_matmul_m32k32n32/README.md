# PF01_matmul_m32k32n32

PF01_matmul_m32k32n32: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=32; K=32, M=32, N=32; op=matmul).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}

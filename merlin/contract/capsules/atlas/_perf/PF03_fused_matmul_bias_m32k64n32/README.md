# PF03_fused_matmul_bias_m32k64n32

PF03_fused_matmul_bias_m32k64n32: fused_matmul_bias over W[64, 32]:fp8_e4m3, A0[32, 64]:fp8_e4m3, B[32]:bf16, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=32; K=64, M=32, N=32; op=fused_matmul_bias).

kind=model_slice label=dev op=fused_matmul_bias modes={'relu': False, 'acc_scale': False}

# PF00_fused_matmul_bias_m16k16n16

PF00_fused_matmul_bias_m16k16n16: fused_matmul_bias over W[16, 16]:i8, A0[16, 16]:i8, B[16]:i32, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=16; K=16, M=16, N=16; op=fused_matmul_bias).

kind=model_slice label=dev op=fused_matmul_bias modes={'relu': False, 'acc_scale': False, 'i8': False}

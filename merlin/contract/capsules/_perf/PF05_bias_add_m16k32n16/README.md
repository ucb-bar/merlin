# PF05_bias_add_m16k32n16

PF05_bias_add_m16k32n16: bias_add over X[16, 16]:i32, B[16]:i32, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=16; K=32, M=16, N=16; op=bias_add).

kind=model_slice label=dev op=bias_add modes={}

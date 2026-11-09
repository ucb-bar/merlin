# PF05_bias_add_m32k64n32

PF05_bias_add_m32k64n32: bias_add over X[32, 32]:bf16, B[32]:bf16, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=32; K=64, M=32, N=32; op=bias_add).

kind=model_slice label=dev op=bias_add modes={}

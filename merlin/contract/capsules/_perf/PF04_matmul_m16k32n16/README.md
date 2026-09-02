# PF04_matmul_m16k32n16

PF04_matmul_m16k32n16: matmul over W[32, 16]:i8, A0[16, 32]:i8, authored from shared epilogue-fusion group: one fused matmul+bias against the matmul and the bias add it replaces, at one identical shape per group (tile=16; K=32, M=16, N=16; op=matmul).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

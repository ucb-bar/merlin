# OPR0_narrow_m_1

OPR0_narrow_m_1: matmul over W[64, 32]:i8, A0[1, 64]:i8, authored from M=1 vecmat: a prior integration packed a single row to a full tile; also the shape the workload census found both numerous and cheap.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

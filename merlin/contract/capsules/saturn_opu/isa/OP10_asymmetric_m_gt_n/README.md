# OP10_asymmetric_m_gt_n

OP10_asymmetric_m_gt_n: matmul over W[24, 4]:i8, A0[16, 24]:i8, authored from M != N, so an operand swap in the accumulate is visible.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

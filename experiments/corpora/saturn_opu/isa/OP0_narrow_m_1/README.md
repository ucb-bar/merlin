# OP0_narrow_m_1

OP0_narrow_m_1: matmul over W[64, 16]:i8, A0[1, 64]:i8, authored from M=1 vecmat: a prior integration packed this to a single row of a full tile.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

# OPR1_narrow_both_1

OPR1_narrow_both_1: matmul over W[64, 1]:i8, A0[1, 64]:i8, authored from both parallel extents 1: a rank-1 accumulate degenerating to a dot product, where a broadcast move-in that assumes a full tile writes past the operand.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

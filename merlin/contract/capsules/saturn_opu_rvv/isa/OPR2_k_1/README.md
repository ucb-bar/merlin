# OPR2_k_1

OPR2_k_1: matmul over W[1, 32]:i8, A0[32, 1]:i8, authored from a single rank-1 update: the accumulate fires exactly once, so a kernel that folds the readout into the loop tail emits it in the wrong place.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}

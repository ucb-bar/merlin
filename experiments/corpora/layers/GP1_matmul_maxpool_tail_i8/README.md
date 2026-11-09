# GP1_matmul_maxpool_tail_i8

GP1: 25x16x17 int8 matmul; the commit pools a ragged 5x5 plane 2x2/2 down to 2x2.

kind=layer label=public op=matmul modes={'i8': False, 'relu': False, 'acc_scale': False}

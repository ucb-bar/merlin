# GP0_matmul_maxpool_i8

GP0: 16x16x16 int8 matmul whose commit fuses a 2x2/2 max-pool over the 4x4 output plane.

kind=layer label=public op=matmul modes={'i8': False, 'relu': False, 'acc_scale': False}

# GP2_conv2d_maxpool_i8

GP2: 3x3 int8 conv over 8x8x4 -> 6x6x16, with a 2x2/2 max-pool fused onto the store.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

# GQ0_conv2d_requant_relu_i8

GQ0: 3x3 int8 conv over 8x8x4 -> 6x6x16 with bias/acc_scale/relu fused on the store path.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True, 'i8': True, 'acc_scale': True, 'relu': True}

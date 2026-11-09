# GQ2_conv2d_pad_requant_relu_i8

GQ2: 3x3 same-padded int8 conv over 6x6x4 -> 6x6x16 with the quantized epilogue fused.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True, 'i8': True, 'acc_scale': True, 'relu': True}

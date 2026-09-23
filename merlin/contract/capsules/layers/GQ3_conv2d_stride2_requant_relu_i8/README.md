# GQ3_conv2d_stride2_requant_relu_i8

GQ3: 3x3 stride-2 int8 conv over 8x8x8 -> 3x3x16 with the quantized epilogue fused.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True, 'i8': True, 'acc_scale': True, 'relu': True}

# GC8_conv2d_stride2_i8

GC8_conv2d_stride2_i8: conv2d over W[36, 8]:i8, IFM[1, 8, 8, 4]:i8, authored from im2col conv2d int8 at stride 2: consecutive output positions share no input tap.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

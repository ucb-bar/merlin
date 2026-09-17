# GC7_conv2d_pad_i8

GC7_conv2d_pad_i8: conv2d over W[36, 8]:i8, IFM[1, 8, 8, 4]:i8, authored from im2col conv2d int8 at same-padding: the out-of-bounds taps must read as zero rather than as whatever the staging buffer last held.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

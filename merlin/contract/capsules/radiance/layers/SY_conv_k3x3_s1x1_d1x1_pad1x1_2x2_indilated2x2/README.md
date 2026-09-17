# SY_conv_k3x3_s1x1_d1x1_pad1x1_2x2_indilated2x2

SY_conv_k3x3_s1x1_d1x1_pad1x1_2x2_indilated2x2: conv2d over W[36, 16]:fp16, IFM[1, 3, 3, 4]:fp16, authored from synthesized for the convolution-window axis: window k3x3/s1x1/d1x1/pad1x1_2x2/indilated2x2, recovered structurally from 2 region(s) of ['deepjscc_int8']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

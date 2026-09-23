# SY_conv_k7x7_s4x4_d1x1_pad3x3

SY_conv_k7x7_s4x4_d1x1_pad3x3: conv2d over W[196, 16]:fp16, IFM[1, 13, 13, 4]:fp16, authored from synthesized for the convolution-window axis: window k7x7/s4x4/d1x1/pad3x3, recovered structurally from 4 region(s) of ['lstmnetvit_fp8', 'lstmnetvit_int8', 'lstmnetvit_int8_pretransposed', 'lstmnetvit_int8_w8a8']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

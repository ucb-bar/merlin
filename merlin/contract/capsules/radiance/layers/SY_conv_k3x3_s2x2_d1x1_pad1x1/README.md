# SY_conv_k3x3_s2x2_d1x1_pad1x1

SY_conv_k3x3_s2x2_d1x1_pad1x1: conv2d over W[36, 16]:fp16, IFM[1, 7, 7, 4]:fp16, authored from synthesized for the convolution-window axis: window k3x3/s2x2/d1x1/pad1x1, recovered structurally from 6 region(s) of ['deepjscc_int8', 'lstmnetvit_fp8', 'lstmnetvit_int8', 'lstmnetvit_int8_pretransposed', 'lstmnetvit_int8_w8a8']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

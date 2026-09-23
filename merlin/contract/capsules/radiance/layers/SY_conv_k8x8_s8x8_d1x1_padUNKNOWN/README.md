# SY_conv_k8x8_s8x8_d1x1_padUNKNOWN

SY_conv_k8x8_s8x8_d1x1_padUNKNOWN: conv2d over W[256, 16]:fp16, IFM[1, 32, 32, 4]:fp16, authored from synthesized for the convolution-window axis: window k8x8/s8x8/d1x1/padUNKNOWN, recovered structurally from 8 region(s) of ['lstmnetvit_fp8', 'lstmnetvit_int8', 'lstmnetvit_int8_pretransposed', 'lstmnetvit_int8_w8a8']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer. THE PADDING WAS NOT READABLE in the capture -- it is applied by an index gather, i.e. a reflection pad -- so this member tests the WINDOW and its stepping at zero padding and asserts nothing about a padding identity that is not zero.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

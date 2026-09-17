# SY_conv_k5x5_s1x1_d1x1_padUNKNOWN

SY_conv_k5x5_s1x1_d1x1_padUNKNOWN: conv2d over W[100, 16]:i8, IFM[1, 8, 8, 4]:i8, authored from synthesized for the convolution-window axis: window k5x5/s1x1/d1x1/padUNKNOWN, recovered structurally from 1 region(s) of ['deepjscc_int8']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer. THE PADDING WAS NOT READABLE in the capture -- it is applied by an index gather, i.e. a reflection pad -- so this member tests the WINDOW and its stepping at zero padding and asserts nothing about a padding identity that is not zero.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

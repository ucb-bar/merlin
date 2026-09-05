# SY_conv_k16x16_s16x16_d1x1_padUNKNOWN

SY_conv_k16x16_s16x16_d1x1_padUNKNOWN: conv2d over W[1024, 16]:i8, IFM[1, 8, 8, 4]:i8, authored from synthesized for the convolution-window axis: window k16x16/s16x16/d1x1/padUNKNOWN, recovered structurally from 6 region(s) of ['smolvla', 'smolvla_denoise_step_fp32_app', 'smolvla_int8', 'smolvla_pretransposed', 'spectformer_int8_full', 'spectformer_int8_full_pretransposed']. torch-mlir emits im2col, so a captured convolution carries no padding/stride/dilation attribute at all and the geometry comes from the gather's affine map and its padding producer. THE PADDING WAS NOT READABLE in the capture -- it is applied by an index gather, i.e. a reflection pad -- so this member tests the WINDOW and its stepping at zero padding and asserts nothing about a padding identity that is not zero.

kind=layer label=public op=conv2d modes={'conv2d': True, 'k_accumulate': True}

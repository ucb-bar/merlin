# OPM3_patch_embed

OPM3_patch_embed: linear over W[768, 196]:i8, X[256, 768]:i8, authored from the patch-embedding contraction; N=196 is not a whole number of tiles.

kind=model_slice label=public op=linear modes={'relu': False, 'acc_scale': False, 'i8': False}

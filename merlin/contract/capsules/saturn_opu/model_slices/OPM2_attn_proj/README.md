# OPM2_attn_proj

OPM2_attn_proj: linear over W[256, 768]:i8, X[196, 256]:i8, authored from the fused QKV projection (8 instances); N=768 is a whole number of tiles.

kind=model_slice label=public op=linear modes={'relu': False, 'acc_scale': False, 'i8': False}

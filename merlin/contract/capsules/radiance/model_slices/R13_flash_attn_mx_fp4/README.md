# R13_flash_attn_mx_fp4

R13_flash_attn_mx_fp4: attention_mx over Q[32, 32]:mxfp4, K[32, 32]:mxfp4, V[32, 32]:mxfp4, authored from MX flash attention, mxfp4 (e2m1 nibble, E8M0): softmax(Q@K^T/sqrt(H))@V.

kind=model_slice label=public op=attention_mx modes={}

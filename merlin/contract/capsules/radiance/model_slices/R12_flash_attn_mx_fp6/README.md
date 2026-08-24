# R12_flash_attn_mx_fp6

R12_flash_attn_mx_fp6: attention_mx over Q[32, 32]:mxfp6, K[32, 32]:mxfp6, V[32, 32]:mxfp6, authored from MX flash attention, mxfp6 (e3m2 nibble+LUT, E8M0): softmax(Q@K^T/sqrt(H))@V.

kind=model_slice label=public op=attention_mx modes={}

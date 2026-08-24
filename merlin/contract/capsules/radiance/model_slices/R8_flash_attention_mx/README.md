# R8_flash_attention_mx

R8_flash_attention_mx: attention_mx over Q[16, 32]:mxfp8, K[32, 32]:mxfp8, V[32, 16]:mxfp8, authored from MX flash attention (mxfp8, E8M0): softmax(Q@K^T/sqrt(H))@V — radiance-kernels flash_attention_mx.

kind=model_slice label=public op=attention_mx modes={}

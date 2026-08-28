# R10_flash_attn_mx_gemma

R10_flash_attn_mx_gemma: attention_mx over Q[16, 32]:mxfp8, K[32, 32]:mxfp8, V[32, 16]:mxfp8, Q_scale[1, 16]:e8m0, K_scale[1, 32]:e8m0, V_scale[1, 16]:e8m0, P_scale[1, 16]:e8m0, authored from MX flash attention with Gemma-2 logit soft-cap (cap=50) — gemma attention variant.

kind=model_slice label=public op=attention_mx modes={}

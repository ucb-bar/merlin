# R9_flash_attn_mx_gqa

R9_flash_attn_mx_gqa: attention_mx over Q[16, 32]:mxfp8, K[64, 32]:mxfp8, V[64, 32]:mxfp8, Q_scale[1, 16]:e8m0, K_scale[1, 64]:e8m0, V_scale[2, 32]:e8m0, P_scale[2, 16]:e8m0, authored from MX flash attention, GQA shape (wider KV: Skv=64, Dv=32) — grouped-query attention tile.

kind=model_slice label=public op=attention_mx modes={}

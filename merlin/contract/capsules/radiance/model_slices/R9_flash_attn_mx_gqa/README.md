# R9_flash_attn_mx_gqa

R9_flash_attn_mx_gqa: attention_mx over Q[16, 32]:mxfp8, K[64, 32]:mxfp8, V[64, 32]:mxfp8, authored from MX flash attention, GQA shape (wider KV: Skv=64, Dv=32) — grouped-query attention tile.

kind=model_slice label=public op=attention_mx modes={}

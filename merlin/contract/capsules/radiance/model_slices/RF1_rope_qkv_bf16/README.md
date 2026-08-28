# RF1_rope_qkv_bf16

RF1_rope_qkv_bf16: rope_qkv over X[16, 16]:bf16, Wqkv[16, 32]:bf16, authored from fused QKV+RoPE: Y = rope(X@Wqkv) bf16 (radiance-kernels rope_qkv_fused).

kind=model_slice label=public op=rope_qkv modes={'rope': True}

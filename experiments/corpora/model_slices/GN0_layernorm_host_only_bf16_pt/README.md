# GN0_layernorm_host_only_bf16_pt

GN0_layernorm_host_only_bf16_pt: layernorm over X[16, 32]:bf16, W[32]:bf16, B[32]:bf16, authored from PyTorch LayerNorm bf16 — the negative lane: this target declares no normalization capability, so every region must land on the host and none may reach the mesh.

kind=model_slice label=public op=layernorm modes={}

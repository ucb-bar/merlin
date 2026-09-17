# SY_host_lane_attention_bf16

SY_host_lane_attention_bf16: attention_full over Q[16, 32]:bf16, K[16, 32]:bf16, V[16, 32]:bf16, authored from synthesized for the host lane: real captures contain 3 'attention' region(s) at bf16, and this target's manifest admits 'attention' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=attention_full modes={}

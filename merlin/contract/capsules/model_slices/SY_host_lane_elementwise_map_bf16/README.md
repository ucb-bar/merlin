# SY_host_lane_elementwise_map_bf16

SY_host_lane_elementwise_map_bf16: gelu over X[16, 32]:bf16, authored from synthesized for the host lane: real captures contain 8 'elementwise_map' region(s) at bf16, and this target's manifest admits 'elementwise_map' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=gelu modes={}

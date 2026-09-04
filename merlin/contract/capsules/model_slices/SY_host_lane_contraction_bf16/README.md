# SY_host_lane_contraction_bf16

SY_host_lane_contraction_bf16: depthwise_conv2d over X[1, 1, 8, 8]:bf16, W[1, 1, 2, 2]:bf16, authored from synthesized for the host lane: real captures contain 5 'contraction' region(s) at bf16, and this target's manifest admits 'contraction' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=depthwise_conv2d modes={}

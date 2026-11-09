# SY_host_lane_contraction_f32

SY_host_lane_contraction_f32: depthwise_conv2d over X[1, 1, 8, 8]:f32, W[1, 1, 2, 2]:f32, authored from synthesized for the host lane: real captures contain 2049 'contraction' region(s) at f32, and this target's manifest admits 'contraction' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=depthwise_conv2d modes={}

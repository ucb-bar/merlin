# SY_host_lane_softmax_bf16

SY_host_lane_softmax_bf16: softmax over X[32, 64]:bf16, authored from synthesized for the host lane: real captures contain 1 'softmax' region(s) at bf16, and this target's manifest admits 'softmax' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=softmax modes={}

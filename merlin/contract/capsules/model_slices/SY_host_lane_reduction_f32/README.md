# SY_host_lane_reduction_f32

SY_host_lane_reduction_f32: reduce_sum over X[16, 32]:f32, authored from synthesized for the host lane: real captures contain 2166 'reduction' region(s) at f32, and this target's manifest admits 'reduction' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=reduce_sum modes={}

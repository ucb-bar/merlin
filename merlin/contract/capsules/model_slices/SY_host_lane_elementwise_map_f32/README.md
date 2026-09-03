# SY_host_lane_elementwise_map_f32

SY_host_lane_elementwise_map_f32: gelu over X[16, 32]:f32, authored from synthesized for the host lane: real captures contain 8096 'elementwise_map' region(s) at f32, and this target's manifest admits 'elementwise_map' at no such dtype -- so every one of them must be placed on the host. A corpus with no capsule here cannot tell a compiler that routes them correctly from one that does not.

kind=model_slice label=public op=gelu modes={}

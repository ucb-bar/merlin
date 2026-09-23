# SY_host_only_elementwise_map

SY_host_only_elementwise_map: gelu over X[32, 64]:f32, authored from synthesized for host-only family 'elementwise_map': real captures contain it and this target's capability manifest admits no capability for it, so the compiler must leave it on the host lane. dtype f32 is the one the captures carry for this family.

kind=model_slice label=public op=gelu modes={}

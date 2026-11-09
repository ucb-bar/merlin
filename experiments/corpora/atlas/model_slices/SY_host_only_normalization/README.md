# SY_host_only_normalization

SY_host_only_normalization: layernorm over X[32, 64]:f32, W[64]:f32, B[64]:f32, authored from synthesized for host-only family 'normalization': real captures contain it and this target's capability manifest admits no capability for it, so the compiler must leave it on the host lane. dtype f32 is the one the captures carry for this family.

kind=model_slice label=public op=layernorm modes={}

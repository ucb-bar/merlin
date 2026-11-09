# SY_host_only_reduction

SY_host_only_reduction: reduce_sum over X[32, 64]:f32, authored from synthesized for host-only family 'reduction': real captures contain it and this target's capability manifest admits no capability for it, so the compiler must leave it on the host lane. dtype f32 is the one the captures carry for this family.

kind=model_slice label=public op=reduce_sum modes={}

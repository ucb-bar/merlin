# SY_host_only_attention

SY_host_only_attention: attention_full over Q[16, 32]:f32, K[16, 32]:f32, V[16, 32]:f32, authored from synthesized for host-only family 'attention': real captures contain it and this target's capability manifest admits no capability for it, so the compiler must leave it on the host lane. dtype f32 is the one the captures carry for this family.

kind=model_slice label=public op=attention_full modes={}

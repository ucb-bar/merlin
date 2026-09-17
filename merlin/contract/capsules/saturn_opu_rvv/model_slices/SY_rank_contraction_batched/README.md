# SY_rank_contraction_batched

SY_rank_contraction_batched: gemv_batched over W[2, 64, 32]:i8, A0[2, 32, 64]:i8, authored from synthesized for the rank axis: this target's capability manifest declares 'contraction' handles a batched region, and a (family, dtype, alignment) cell cannot demand one. Shape and layout come from capability_probes.

kind=model_slice label=public op=gemv_batched modes={}

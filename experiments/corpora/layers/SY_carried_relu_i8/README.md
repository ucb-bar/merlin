# SY_carried_relu_i8

SY_carried_relu_i8: resident_reuse over W[32, 16]:i8, A0[16, 32]:i8, A1[16, 32]:i8, authored from synthesized for the carried-state axis: 'relu' is configuration a unit stays in (evidenced by ['manifest_composed_with']), so the command after it, which does not ask for it, is where a backend that does not restore its configuration shows.

kind=layer label=public op=resident_reuse modes={'resident_reuse': True}

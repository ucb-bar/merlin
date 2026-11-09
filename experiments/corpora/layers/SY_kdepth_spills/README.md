# SY_kdepth_spills

SY_kdepth_spills: matmul over W[16384, 16]:i8, A0[16, 16384]:i8, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'spills' residency regime admits (1024 accumulation passes, 2.0 of the operand store). Too large to certify cycle-accurately, so it is L2-only and extends SY_kdepth_certified, which carries the functional guarantee this perf-facing depth rests on.

kind=layer label=public op=matmul modes={}

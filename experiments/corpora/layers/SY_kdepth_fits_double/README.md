# SY_kdepth_fits_double

SY_kdepth_fits_double: matmul over W[4096, 16]:i8, A0[16, 4096]:i8, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'fits_double' residency regime admits (256 accumulation passes, 0.5 of the operand store). Too large to certify cycle-accurately, so it is L2-only and extends SY_kdepth_certified, which carries the functional guarantee this perf-facing depth rests on.

kind=layer label=public op=matmul modes={}

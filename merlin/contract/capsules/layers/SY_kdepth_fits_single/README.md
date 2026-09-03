# SY_kdepth_fits_single

SY_kdepth_fits_single: matmul over W[8192, 16]:i8, A0[16, 8192]:i8, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'fits_single' residency regime admits (512 accumulation passes, 1.0 of the operand store). Too large to certify cycle-accurately, so it is L2-only and extends SY_kdepth_certified, which carries the functional guarantee this perf-facing depth rests on.

kind=layer label=public op=matmul modes={}

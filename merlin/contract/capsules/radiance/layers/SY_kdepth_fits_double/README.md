# SY_kdepth_fits_double

SY_kdepth_fits_double: matmul over W[1024, 16]:bf16, A0[16, 1024]:bf16, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'fits_double' residency regime admits (64 accumulation passes, 0.5 of the operand store). The reduction moves the operands and not the result, so this certifies at one output tile.

kind=layer label=public op=matmul modes={}

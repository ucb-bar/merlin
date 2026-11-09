# SY_kdepth_fits_single

SY_kdepth_fits_single: matmul over W[2048, 16]:bf16, A0[16, 2048]:bf16, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'fits_single' residency regime admits (128 accumulation passes, 1.0 of the operand store). The reduction moves the operands and not the result, so this certifies at one output tile.

kind=layer label=public op=matmul modes={}

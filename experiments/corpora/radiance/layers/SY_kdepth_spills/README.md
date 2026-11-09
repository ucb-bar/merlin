# SY_kdepth_spills

SY_kdepth_spills: matmul over W[4096, 16]:bf16, A0[16, 4096]:bf16, authored from synthesized for the accumulation-depth axis: the deepest reduction the 'spills' residency regime admits (256 accumulation passes, 2.0 of the operand store). The reduction moves the operands and not the result, so this certifies at one output tile.

kind=layer label=public op=matmul modes={}

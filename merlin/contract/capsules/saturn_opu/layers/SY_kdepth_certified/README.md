# SY_kdepth_certified

SY_kdepth_certified: matmul over W[32, 16]:i8, A0[16, 32]:i8, authored from synthesized for the accumulation-depth axis: 2 accumulation passes, the shallowest reduction that writes the accumulator more than once. Emitted only because this target's residency regimes yield no depth of their own; where they do, they produce deeper capsules and this would duplicate one. Costs 81.0s against a 300.0s budget -- priced on the OUTPUT tile, which the reduction depth does not move.

kind=layer label=public op=matmul modes={}
